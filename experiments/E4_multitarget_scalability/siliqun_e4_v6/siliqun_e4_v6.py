"""
siliqun_e4_v6.py
================
SAC-based quantum state preparation on SiliQun — Silicon Spin Qubit Simulator.

This script is completely standalone and has NO dependency on QUASAR, ANDROMEDA,
or any other project. It uses the `siliqun` package (siliqun.envs.SiliQunEnv)
as its physics environment.

Changes from v5:
  - Replaced `from quasar_v9.envs import QuasarEnv` with `from siliqun.envs import SiliQunEnv`
  - Renamed all QuasarEnv → SiliQunEnv throughout
  - Updated all log messages, result metadata, and PBS script references
  - Removed quasar_v9 package cache path
  - Version bumped to v6

Author: Raad Al-Shehri | KAU FCIT PhD
Version: 6.0.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Standalone SiliQun environment import — NO QUASAR dependency
# ---------------------------------------------------------------------------
try:
    from siliqun.envs import SiliQunEnv
    print("[SiliQun v6] siliqun.envs.SiliQunEnv imported successfully", flush=True)
except ImportError as exc:
    print(f"[SiliQun v6] ERROR: Could not import SiliQunEnv: {exc}", file=sys.stderr)
    print("[SiliQun v6] Install the siliqun package: pip install -e /path/to/siliqun_package", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# SAC components (self-contained, no external DRL framework required)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

LOG_STD_MAX =  2
LOG_STD_MIN = -20
EPSILON     =  1e-6


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h        = self.net(obs)
        mu       = self.mu_head(h)
        log_std  = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std         = log_std.exp()
        dist        = Normal(mu, std)
        x_t         = dist.rsample()
        y_t         = torch.tanh(x_t)
        action      = y_t * math.pi          # scale to [-π, π]
        log_prob    = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob    = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0
        self.obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),        dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),        dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     float,
    ) -> None:
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.FloatTensor(self.obs[idx]),
            "next_obs": torch.FloatTensor(self.next_obs[idx]),
            "actions":  torch.FloatTensor(self.actions[idx]),
            "rewards":  torch.FloatTensor(self.rewards[idx]),
            "dones":    torch.FloatTensor(self.dones[idx]),
        }


class SACAgent:
    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        hidden_dim: int   = 256,
        lr_actor:   float = 3e-4,
        lr_critic:  float = 3e-4,
        alpha:      float = 0.2,
        tau:        float = 0.005,
        gamma:      float = 0.99,
        buffer_cap: int   = 1_000_000,
        batch_size: int   = 256,
        device:     str   = "auto",
    ):
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            device if device != "auto" else "cpu"
        )
        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size

        self.actor  = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_cap, obs_dim, act_dim)

        # Automatic entropy tuning
        self.target_entropy = -act_dim
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt      = optim.Adam([self.log_alpha], lr=lr_actor)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if deterministic:
                mu, _ = self.actor(obs_t)
                action = torch.tanh(mu) * math.pi
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().flatten()

    def update(self) -> Optional[Dict[str, float]]:
        if self.buffer.size < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        obs      = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        actions  = batch["actions"].to(self.device)
        rewards  = batch["rewards"].to(self.device)
        dones    = batch["dones"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        new_actions, log_pi = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
        }


# ---------------------------------------------------------------------------
# SLM correction (Spectral Landscape Mapping)
# ---------------------------------------------------------------------------

def slm_correction(
    agent: SACAgent,
    env:   SiliQunEnv,
    n_probe: int = 20,
) -> bool:
    """
    Probe the landscape around the current best policy and nudge the actor
    toward higher-fidelity regions.

    Returns True if a correction was applied.
    """
    best_F   = 0.0
    best_obs = None

    for _ in range(n_probe):
        obs, _ = env.reset()
        ep_best = 0.0
        for _ in range(50):
            action  = agent.select_action(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(action)
            if info["fidelity"] > ep_best:
                ep_best  = info["fidelity"]
                best_obs = obs.copy()
            if term or trunc:
                break
        if ep_best > best_F:
            best_F = ep_best

    if best_obs is not None and best_F > 0.1:
        obs_t = torch.FloatTensor(best_obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            mu, _ = agent.actor(obs_t)
            target_action = torch.tanh(mu) * math.pi
        # Add high-fidelity transitions to buffer
        obs_tmp, _ = env.reset()
        for _ in range(10):
            act_np = target_action.cpu().numpy().flatten()
            next_obs, rew, term, trunc, info = env.step(act_np)
            agent.buffer.add(obs_tmp, act_np, rew, next_obs, float(term or trunc))
            obs_tmp = next_obs
            if term or trunc:
                break
        return True
    return False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_cell(
    n_qubits:       int,
    target_state:   str,
    seed:           int,
    noise_stage:    int    = 5,
    max_steps:      int    = 500_000,
    hidden_dim:     int    = 256,
    lr_actor:       float  = 3e-4,
    lr_critic:      float  = 3e-4,
    alpha:          float  = 0.2,
    tau:            float  = 0.005,
    batch_size:     int    = 256,
    slm_interval:   int    = 50_000,
    reward_weights: Tuple  = (1.0, 0.1, 0.05, 0.01),
    result_dir:     str    = "results/siliqun_v6",
    log_interval:   int    = 5_000,
    warmup_steps:   int    = 10_000,
) -> Dict:
    """Train SAC on a single (n_qubits, target_state, seed) cell."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SiliQunEnv(
        n_qubits     = n_qubits,
        target_state = target_state,
        noise_stage  = noise_stage,
        max_steps    = 200,
        seed         = seed,
        reward_weights = reward_weights,
    )

    agent = SACAgent(
        obs_dim    = env.obs_dim,
        act_dim    = env.act_dim,
        hidden_dim = hidden_dim,
        lr_actor   = lr_actor,
        lr_critic  = lr_critic,
        alpha      = alpha,
        tau        = tau,
        batch_size = batch_size,
    )

    obs, _ = env.reset(seed=seed)
    best_F          = 0.0
    rolling_F_sum   = 0.0
    rolling_count   = 0
    plateau_steps   = 0
    plateau_limit   = 500_000
    slm_count       = 0
    start_time      = time.time()

    print(
        f"[SiliQun v6] Starting: {n_qubits}Q/{target_state}/s{seed} | "
        f"noise_stage={noise_stage} | max_steps={max_steps:,}",
        flush=True,
    )

    for step in range(1, max_steps + 1):
        if step < warmup_steps:
            action = np.random.uniform(-math.pi, math.pi, env.act_dim)
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs if not done else env.reset(seed=seed + step)[0]

        F = info["fidelity"]
        rolling_F_sum  += F
        rolling_count  += 1

        if F > best_F:
            best_F       = F
            plateau_steps = 0
        else:
            plateau_steps += 1

        if step >= warmup_steps:
            agent.update()

        # SLM correction
        if step % slm_interval == 0 and step > warmup_steps:
            applied = slm_correction(agent, env)
            if applied:
                slm_count += 1
                print(
                    f"[SiliQun v6] SLM correction #{slm_count} at step={step:,} | "
                    f"best_F={best_F:.4f}",
                    flush=True,
                )

        # Logging
        if step % log_interval == 0:
            rolling_F = rolling_F_sum / rolling_count
            elapsed   = time.time() - start_time
            print(
                f"[SiliQun v6] {n_qubits}Q/{target_state}/s{seed} | "
                f"step={step:,} | best_F={best_F:.4f} | "
                f"rolling_F={rolling_F:.4f} | plateau={plateau_steps:,} | "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            rolling_F_sum  = 0.0
            rolling_count  = 0

        # Early stop on plateau
        if plateau_steps >= plateau_limit:
            print(
                f"[SiliQun v6] Early stop: plateau {plateau_steps:,}/{plateau_limit:,} "
                f"at step={step:,} | best_F={best_F:.4f}",
                flush=True,
            )
            break

        # Success stop
        if best_F >= 0.999:
            print(
                f"[SiliQun v6] SUCCESS: F={best_F:.4f} >= 0.999 at step={step:,}",
                flush=True,
            )
            break

    elapsed = time.time() - start_time
    result  = {
        "algorithm":            "SiliQun-v6-SAC",
        "env":                  "SiliQunEnv-v1.0",   # ← no QUASAR reference
        "n_qubits":             n_qubits,
        "target_state":         target_state,
        "seed":                 seed,
        "noise_stage":          noise_stage,
        "steps":                step,
        "best_fidelity":        round(best_F, 6),
        "threshold_success":    best_F >= 0.99,
        "slm_interventions":    slm_count,
        "runtime_seconds":      round(elapsed, 1),
        "hidden_dim":           hidden_dim,
        "lr_actor":             lr_actor,
        "lr_critic":            lr_critic,
        "alpha_final":          round(agent.alpha, 6),
        "tau":                  tau,
        "batch_size":           batch_size,
        "slm_interval":         slm_interval,
    }

    # Save result JSON
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{n_qubits}Q_{target_state}_s{seed}.json"
    with open(fname, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[SiliQun v6] Result saved → {fname}", flush=True)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SiliQun v6 — SAC training harness")
    p.add_argument("--n-qubits",      type=int,   default=3)
    p.add_argument("--target",        type=str,   default="ghz",
                   choices=["ghz", "w", "cluster_linear", "dicke_k1", "dicke_k2", "dicke_k3"])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--noise-stage",   type=int,   default=5)
    p.add_argument("--max-steps",     type=int,   default=1_000_000)
    p.add_argument("--hidden-dim",    type=int,   default=256)
    p.add_argument("--lr-actor",      type=float, default=3e-4)
    p.add_argument("--lr-critic",     type=float, default=3e-4)
    p.add_argument("--alpha",         type=float, default=0.2)
    p.add_argument("--tau",           type=float, default=0.005)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--slm-interval",  type=int,   default=50_000)
    p.add_argument("--result-dir",    type=str,   default="results/siliqun_v6")
    p.add_argument("--log-interval",  type=int,   default=5_000)
    # Multi-cell sweep mode
    p.add_argument("--sweep", action="store_true",
                   help="Run all (n_qubits, target, seed) combinations")
    p.add_argument("--qubit-range",   type=int,   nargs=2, default=[2, 6])
    p.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 777])
    p.add_argument("--targets",       type=str,   nargs="+",
                   default=["ghz", "w", "cluster_linear", "dicke_k3"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep:
        cells = [
            (n, t, s)
            for n in range(args.qubit_range[0], args.qubit_range[1] + 1)
            for t in args.targets
            for s in args.seeds
        ]
        print(f"[SiliQun v6] Sweep mode: {len(cells)} cells", flush=True)
        for n, t, s in cells:
            run_cell(
                n_qubits     = n,
                target_state = t,
                seed         = s,
                noise_stage  = args.noise_stage,
                max_steps    = args.max_steps,
                hidden_dim   = args.hidden_dim,
                lr_actor     = args.lr_actor,
                lr_critic    = args.lr_critic,
                alpha        = args.alpha,
                tau          = args.tau,
                batch_size   = args.batch_size,
                slm_interval = args.slm_interval,
                result_dir   = args.result_dir,
                log_interval = args.log_interval,
            )
    else:
        run_cell(
            n_qubits     = args.n_qubits,
            target_state = args.target,
            seed         = args.seed,
            noise_stage  = args.noise_stage,
            max_steps    = args.max_steps,
            hidden_dim   = args.hidden_dim,
            lr_actor     = args.lr_actor,
            lr_critic    = args.lr_critic,
            alpha        = args.alpha,
            tau          = args.tau,
            batch_size   = args.batch_size,
            slm_interval = args.slm_interval,
            result_dir   = args.result_dir,
            log_interval = args.log_interval,
        )


if __name__ == "__main__":
    main()
