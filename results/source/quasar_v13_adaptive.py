"""
quasar_v13_adaptive.py — QUASAR v13: Adaptive Open-Ended Scalability Experiment
=================================================================================
Architecture
------------
This script implements an **adaptive curriculum** that starts at 2 qubits and
advances to the next qubit count only when the current one achieves F >= F_THRESHOLD
(default 0.99) across ALL target states and ALL seeds.  If the system stalls below
F_THRESHOLD at qubit count N, it records N-1 as the empirical scalability ceiling
and terminates.  The ceiling itself is the primary publishable result.

Key components (inherited from v12, all preserved):
  - SAC with FiLM conditioning (goal-conditioned actor/critic)
  - DER++ continual learning buffer (cross-qubit transfer memory)
  - SLM spectral landscape mapping correction
  - DEHB inner hyperparameter optimisation (IMP-13: slm_interval is searched)
  - Adaptive transfer: best agent from N qubits is transferred to N+1 qubits

Advancement logic
-----------------
  For each qubit count N (starting from MIN_QUBITS):
    1. Run DEHB inner loop to find best SAC hyperparameters for (N, target, seed)
    2. Record best_F per (target, seed)
    3. Compute advancement score = mean best_F over all (target, seed) pairs
    4. If advancement_score >= F_THRESHOLD AND min(best_F_per_target) >= F_FLOOR:
         → advance to N+1, transfer best agent via DER++
       Else:
         → record scalability ceiling = N, write final report, exit

Parameters
----------
  --min-qubits     : Starting qubit count (default 2)
  --max-qubits     : Hard ceiling (default 12, H100 memory limit)
  --targets        : Target states to evaluate (default: GHZ W Cluster Dicke-k3)
  --seeds          : Random seeds (default: 42 123 456)
  --f-threshold    : Advancement fidelity threshold (default 0.99)
  --f-floor        : Minimum per-target fidelity to advance (default 0.90)
  --inner-brackets : DEHB bracket count per cell (default 4)
  --max-steps-base : Max training steps at 2Q (default 500_000)
  --steps-scale    : Multiplier per qubit count (default 1.5)
  --noise-stage    : Fixed noise stage (default 5, Stage-5 = full noise)
  --results-dir    : Output directory (default: ~/quasar_v13/results)

Output
------
  results/v13_adaptive/
    scalability_report.json          — final ceiling, per-qubit summary
    {N}Q_{target}_s{seed}.json       — per-cell result
    checkpoints/{N}Q_best_agent.pt   — best agent per qubit count
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigSpace import ConfigurationSpace, Float, Integer
from dehb import DEHB
from loguru import logger as log

# ── Project-local modules (same directory as this script on Aziz) ─────────────
from envs import QuasarEnv, NOISE_CURRICULUM
from der_plus_plus import DERPlusPlusBuffer

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE   = 300_000
WARMUP_STEPS  = 5_000
UPDATE_EVERY  = 1
LOG_EVERY     = 10_000

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, tgt_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),        dtype=np.float32)
        self.targets  = np.zeros((capacity, tgt_dim),  dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, target):
        idx = self.ptr % self.capacity
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = done
        self.targets[idx]  = target
        self.ptr  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        def t(x): return torch.FloatTensor(x[idx]).to(DEVICE)
        return t(self.obs), t(self.actions), t(self.rewards), \
               t(self.next_obs), t(self.dones), t(self.targets)

    def __len__(self): return self.size


# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning
# ─────────────────────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * x + self.beta(cond)


# ─────────────────────────────────────────────────────────────────────────────
# Goal-conditioned Actor
# ─────────────────────────────────────────────────────────────────────────────
class GoalConditionedActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        self.state_enc  = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.target_enc = nn.Sequential(nn.Linear(tgt_dim, cond_dim), nn.ReLU(),
                                        nn.Linear(cond_dim, cond_dim))
        self.film       = FiLM(cond_dim, hidden_dim)
        self.mean_head  = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, target: torch.Tensor):
        h    = self.state_enc(obs)
        cond = self.target_enc(target)
        h    = self.film(h, cond)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        # NaN guard: replace any NaN/Inf with safe defaults
        mean    = torch.nan_to_num(mean,    nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=self.LOG_STD_MIN)
        return mean, log_std

    def get_action_and_logit(self, obs: torch.Tensor, target: torch.Tensor):
        mean, log_std = self.forward(obs, target)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        action   = y_t * math.pi
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Goal-conditioned Critic (twin Q)
# ─────────────────────────────────────────────────────────────────────────────
class GoalConditionedCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        inp = obs_dim + act_dim + tgt_dim
        self.q1_enc = nn.Sequential(nn.Linear(inp, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.q2_enc = nn.Sequential(nn.Linear(inp, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, obs, action, target):
        x = torch.cat([obs, action, target], dim=-1)
        return self.q1_enc(x), self.q2_enc(x)


# ─────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 alpha: float = 0.05, lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4, tau: float = 0.005,
                 hidden_dim: int = 256, cond_dim: int = 64):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.target_dim = tgt_dim
        self.tau        = tau
        self.log_alpha  = torch.tensor(math.log(alpha), requires_grad=True, device=DEVICE)
        self.target_entropy = -act_dim

        self.actor   = GoalConditionedActor(obs_dim, act_dim, tgt_dim,
                                            hidden_dim, cond_dim).to(DEVICE)
        self.critic  = GoalConditionedCritic(obs_dim, act_dim, tgt_dim,
                                             hidden_dim, cond_dim).to(DEVICE)
        self.critic_target = GoalConditionedCritic(obs_dim, act_dim, tgt_dim,
                                                    hidden_dim, cond_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha],          lr=lr_actor)

    @property
    def alpha(self): return self.log_alpha.exp().item()

    def select_action(self, obs: np.ndarray, target: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            t = torch.FloatTensor(target).unsqueeze(0).to(DEVICE)
            action, _, mean_action = self.actor.get_action_and_logit(o, t)
        a = mean_action if deterministic else action
        return a.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer, batch_size: int,
               der_buffer: Optional[DERPlusPlusBuffer] = None,
               der_batch_size: int = 64):
        if len(buffer) < batch_size:
            return
        obs, acts, rews, next_obs, dones, tgts = buffer.sample(batch_size)

        # ── Critic update ────────────────────────────────────────────────────
        with torch.no_grad():
            next_a, next_lp, _ = self.actor.get_action_and_logit(next_obs, tgts)
            q1_t, q2_t = self.critic_target(next_obs, next_a, tgts)
            q_t  = torch.min(q1_t, q2_t) - self.alpha * next_lp
            y    = rews + (1.0 - dones) * 0.99 * q_t

        q1, q2 = self.critic(obs, acts, tgts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # ── DER++ auxiliary loss ─────────────────────────────────────────────
        if der_buffer is not None and der_buffer._n_stored >= der_batch_size:
            der_sample = der_buffer.sample(der_batch_size, DEVICE)
            if der_sample is not None:
                d_obs, d_acts, d_rews, d_next_obs, d_dones, d_tgts = der_sample
                # Pad obs/next_obs if qubit count changed (different obs_dim)
                if d_obs.shape[1] != obs.shape[1]:
                    pad = obs.shape[1] - d_obs.shape[1]
                    if pad > 0:
                        d_obs      = F.pad(d_obs,      (0, pad))
                        d_next_obs = F.pad(d_next_obs, (0, pad))
                    else:
                        d_obs      = d_obs[:, :obs.shape[1]]
                        d_next_obs = d_next_obs[:, :obs.shape[1]]
                # Pad target_vec if dimension changed
                if d_tgts.shape[1] != tgts.shape[1]:
                    pad = tgts.shape[1] - d_tgts.shape[1]
                    if pad > 0:
                        d_tgts = F.pad(d_tgts, (0, pad))
                    else:
                        d_tgts = d_tgts[:, :tgts.shape[1]]
                with torch.no_grad():
                    d_next_a, d_next_lp, _ = self.actor.get_action_and_logit(d_next_obs, d_tgts)
                    dq1_t, dq2_t = self.critic_target(d_next_obs, d_next_a, d_tgts)
                    dy = d_rews + (1.0 - d_dones) * 0.99 * (torch.min(dq1_t, dq2_t) - self.alpha * d_next_lp)
                dq1, dq2 = self.critic(d_obs, d_acts, d_tgts)
                critic_loss = critic_loss + 0.5 * (F.mse_loss(dq1, dy) + F.mse_loss(dq2, dy))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ─────────────────────────────────────────────────────
        a_new, log_pi, _ = self.actor.get_action_and_logit(obs, tgts)
        q1_new, q2_new   = self.critic(obs, a_new, tgts)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Alpha (entropy) update ───────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update ───────────────────────────────────────────────
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "log_alpha": self.log_alpha.item(),
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "target_dim": self.target_dim}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic"])


# ─────────────────────────────────────────────────────────────────────────────
# Transfer agent (weight transplant from N → N+1 qubits)
# ─────────────────────────────────────────────────────────────────────────────
def transfer_agent(source: SACAgent, target_obs_dim: int, target_act_dim: int,
                   target_tgt_dim: int, **sac_kwargs) -> SACAgent:
    new_agent = SACAgent(target_obs_dim, target_act_dim, target_tgt_dim, **sac_kwargs)

    def _copy(src_lin, tgt_lin):
        with torch.no_grad():
            r = min(src_lin.weight.shape[0], tgt_lin.weight.shape[0])
            c = min(src_lin.weight.shape[1], tgt_lin.weight.shape[1])
            tgt_lin.weight.data[:r, :c] = src_lin.weight.data[:r, :c]
            tgt_lin.bias.data[:r]       = src_lin.bias.data[:r]
            if tgt_lin.weight.shape[1] > c:
                tgt_lin.weight.data[:r, c:] = torch.randn_like(
                    tgt_lin.weight.data[:r, c:]) * 0.01
            if tgt_lin.weight.shape[0] > r:
                tgt_lin.weight.data[r:, :] = torch.randn_like(
                    tgt_lin.weight.data[r:, :]) * 0.01
                tgt_lin.bias.data[r:] = 0.0

    _copy(source.actor.state_enc[0],  new_agent.actor.state_enc[0])
    _copy(source.actor.state_enc[2],  new_agent.actor.state_enc[2])  # fixed: was [3] (ReLU), now [2] (Linear)
    _copy(source.actor.target_enc[0], new_agent.actor.target_enc[0])
    _copy(source.actor.target_enc[2], new_agent.actor.target_enc[2])
    _copy(source.actor.film.gamma,    new_agent.actor.film.gamma)
    _copy(source.actor.film.beta,     new_agent.actor.film.beta)
    _copy(source.actor.mean_head,     new_agent.actor.mean_head)
    _copy(source.actor.log_std_head,  new_agent.actor.log_std_head)
    for sq, tq in [(source.critic.q1_enc, new_agent.critic.q1_enc),
                   (source.critic.q2_enc, new_agent.critic.q2_enc)]:
        _copy(sq[0], tq[0]); _copy(sq[2], tq[2])  # fixed: was [3] (ReLU), now [2] (Linear)

    log.info(f"  Transfer: {source.obs_dim}→{target_obs_dim} obs, "
             f"{source.act_dim}→{target_act_dim} act")
    return new_agent


# ─────────────────────────────────────────────────────────────────────────────
# SLM — Spectral Landscape Mapping correction (IMP-7)
# ─────────────────────────────────────────────────────────────────────────────
def slm_correction(agent: SACAgent, env: QuasarEnv,
                   target_vec: np.ndarray, n_probe: int = 200) -> bool:
    obs = env.reset()
    best_F, best_action = -1.0, None
    for _ in range(n_probe):
        a = agent.select_action(obs, target_vec, deterministic=False)
        a_probe = np.clip(a + np.random.randn(*a.shape) * 0.3, -math.pi, math.pi)
        _, _, _, info = env.step(a_probe)
        F = info.get("F", info.get("fidelity", 0.0))
        if F > best_F:
            best_F, best_action = F, a_probe
    if best_action is not None and best_F > 0.5:
        o_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        t_t = torch.FloatTensor(target_vec).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(best_action).unsqueeze(0).to(DEVICE)
        mean, log_std = agent.actor(o_t, t_t)
        std  = log_std.exp()
        loss = -torch.distributions.Normal(mean, std).log_prob(a_t / math.pi).sum()
        agent.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_opt.step()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# run_cell — single training cell (one qubit count / target / seed)
# ─────────────────────────────────────────────────────────────────────────────
def run_cell(
    n_qubits:      int,
    target_state:  str,
    seed:          int,
    noise_stage:   int,
    max_ep_steps:  int,
    reward_w_F:    float,
    reward_w_S:    float,
    reward_w_str:  float,
    alpha:         float,
    lr_actor:      float,
    lr_critic:     float,
    tau:           float,
    batch_size:    int,
    hidden_dim:    int,
    cond_dim:      int,
    max_steps:     int,
    init_agent:    Optional[SACAgent] = None,
    plateau_patience: int = 150_000,
    slm_interval:  int = 50_000,
    der_buffer:    Optional[DERPlusPlusBuffer] = None,
    der_batch_size: int = 64,
) -> Tuple[float, int, List[dict], SACAgent]:
    """Run one training cell. Returns (best_F, total_steps, history, agent)."""
    _TARGET_NORM = {
        "GHZ": "ghz", "ghz": "ghz",
        "W": "w", "w": "w",
        "Cluster": "cluster_linear", "cluster": "cluster_linear", "cluster_linear": "cluster_linear",
        "Dicke-k3": "dicke_k3", "Dicke_k3": "dicke_k3", "dicke-k3": "dicke_k3", "dicke_k3": "dicke_k3",
    }
    target_state = _TARGET_NORM.get(target_state, target_state.lower())
    set_seed(seed)

    _w_log = max(0.0, 1.0 - reward_w_F - reward_w_S - reward_w_str)
    env = QuasarEnv(
        n_qubits=n_qubits,
        target_state=target_state,
        max_steps=max_ep_steps,
        noise_curriculum_stage=noise_stage,
        seed=seed,
        reward_weights=(reward_w_F, reward_w_S, reward_w_str, _w_log),
    )
    obs_dim    = env.obs_dim
    act_dim    = env.act_dim
    target_dim = 2 * env.dim

    sac_kwargs = dict(alpha=alpha, lr_actor=lr_actor, lr_critic=lr_critic,
                      tau=tau, hidden_dim=hidden_dim, cond_dim=cond_dim)

    agent = (transfer_agent(init_agent, obs_dim, act_dim, target_dim, **sac_kwargs)
             if init_agent is not None
             else SACAgent(obs_dim, act_dim, target_dim, **sac_kwargs))

    buffer     = ReplayBuffer(BUFFER_SIZE, obs_dim, act_dim, target_dim)
    target_vec = np.concatenate([env.target.real, env.target.imag]).astype(np.float32)

    obs       = env.reset()
    best_F    = 0.0
    plateau_ctr = 0
    history   = []
    step      = 0

    while step < max_steps:
        action = (env.action_space_sample() if step < WARMUP_STEPS
                  else agent.select_action(obs, target_vec))
        next_obs, reward, done, info = env.step(action)
        F = info.get("F", info.get("fidelity", 0.0))
        buffer.add(obs, action, reward, next_obs, float(done), target_vec)
        obs = next_obs if not done else env.reset()
        step += 1

        if step >= WARMUP_STEPS and step % UPDATE_EVERY == 0:
            agent.update(buffer, batch_size, der_buffer=der_buffer,
                         der_batch_size=der_batch_size)

        if F > best_F:
            best_F = F
            plateau_ctr = 0
        else:
            plateau_ctr += 1

        if step % slm_interval == 0 and step > WARMUP_STEPS:
            if slm_correction(agent, env, target_vec):
                log.info(f"    SLM correction applied at step={step}")

        if step % LOG_EVERY == 0:
            history.append({"step": step, "best_F": round(best_F, 5)})
            log.info(f"    {n_qubits}Q/{target_state}/s{seed} "
                     f"step={step:,} best_F={best_F:.4f} "
                     f"plateau={plateau_ctr:,} alpha={alpha:.4f}")

        if plateau_ctr >= plateau_patience:
            log.info(f"    PLATEAU → early stop at step={step}")
            break

    history.append({"step": step, "best_F": round(best_F, 5)})
    return best_F, step, history, agent


# ─────────────────────────────────────────────────────────────────────────────
# DEHB inner config space (IMP-13: slm_interval is searched)
# ─────────────────────────────────────────────────────────────────────────────
def build_inner_cs() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=42)
    cs.add([
        Float("alpha",          (0.005, 0.50),    log=True,  default=0.05),
        Float("lr_actor",       (1e-4,  5e-3),    log=True,  default=3e-4),
        Float("lr_critic",      (1e-4,  5e-3),    log=True,  default=3e-4),
        Float("tau",            (0.001, 0.05),     log=False, default=0.005),
        Integer("batch_size",   (64,   512),       log=True,  default=256),
        Integer("hidden_dim",   (128,  512),       log=True,  default=256),
        Integer("cond_dim",     (32,   128),       log=True,  default=64),
        Integer("slm_interval", (5_000, 100_000),  log=True,  default=50_000),
    ])
    return cs


# ─────────────────────────────────────────────────────────────────────────────
# InnerTargetFactory — wraps run_cell for DEHB
# ─────────────────────────────────────────────────────────────────────────────
class InnerTargetFactory:
    def __init__(self, n_qubits: int, target_state: str, seed: int,
                 noise_stage: int, max_ep_steps: int,
                 reward_w_F: float, reward_w_S: float, reward_w_str: float,
                 max_steps: int, init_agent: Optional[SACAgent],
                 der_buffer: Optional[DERPlusPlusBuffer]):
        self.n_qubits     = n_qubits
        self.target_state = target_state
        self.seed         = seed
        self.noise_stage  = noise_stage
        self.max_ep_steps = max_ep_steps
        self.reward_w_F   = reward_w_F
        self.reward_w_S   = reward_w_S
        self.reward_w_str = reward_w_str
        self.max_steps    = max_steps
        self.init_agent   = init_agent
        self.der_buffer   = der_buffer
        self.best_F       = 0.0
        self.best_agent: Optional[SACAgent] = None

    def __call__(self, config: dict, fidelity: float, **kwargs) -> dict:
        steps = int(fidelity)
        best_F, total_steps, history, agent = run_cell(
            n_qubits=self.n_qubits,
            target_state=self.target_state,
            seed=self.seed,
            noise_stage=self.noise_stage,
            max_ep_steps=self.max_ep_steps,
            reward_w_F=self.reward_w_F,
            reward_w_S=self.reward_w_S,
            reward_w_str=self.reward_w_str,
            alpha=float(config["alpha"]),
            lr_actor=float(config["lr_actor"]),
            lr_critic=float(config["lr_critic"]),
            tau=float(config["tau"]),
            batch_size=int(config["batch_size"]),
            hidden_dim=int(config["hidden_dim"]),
            cond_dim=int(config["cond_dim"]),
            max_steps=steps,
            init_agent=self.init_agent,
            plateau_patience=max(50_000, steps // 3),
            slm_interval=int(config["slm_interval"]),
            der_buffer=self.der_buffer,
        )
        if best_F > self.best_F:
            self.best_F    = best_F
            self.best_agent = agent
        return {"fitness": -best_F, "cost": total_steps}


# ─────────────────────────────────────────────────────────────────────────────
# run_qubit_level — run DEHB inner loop for all (target, seed) at one N
# ─────────────────────────────────────────────────────────────────────────────
def run_qubit_level(
    n_qubits:       int,
    targets:        List[str],
    seeds:          List[int],
    noise_stage:    int,
    max_steps:      int,
    inner_brackets: int,
    results_dir:    Path,
    ckpt_dir:       Path,
    init_agent:     Optional[SACAgent],
    der_buffer:     DERPlusPlusBuffer,
    # Fixed env defaults (DEHB will search SAC HPs; env params fixed at best known)
    max_ep_steps:   int = 500,
    reward_w_F:     float = 0.70,
    reward_w_S:     float = 0.15,
    reward_w_str:   float = 0.15,
) -> Tuple[Dict[str, Dict[int, float]], Optional[SACAgent]]:
    """
    Run DEHB inner loop for every (target, seed) at qubit count n_qubits.
    Returns:
      cell_results: {target: {seed: best_F}}
      best_agent:   best SACAgent across all cells (for transfer to N+1)
    """
    cell_results: Dict[str, Dict[int, float]] = {t: {} for t in targets}
    level_best_F  = 0.0
    level_best_agent: Optional[SACAgent] = None

    cs = build_inner_cs()
    # Fidelity schedule: min=max_steps//9, max=max_steps, 3 rungs
    min_fid = max(10_000, max_steps // 9)
    max_fid = max_steps

    for target in targets:
        for seed in seeds:
            log.info(f"  ── {n_qubits}Q / {target} / seed={seed} ──")
            # ── Random HP search (replaces DEHB which hangs in PBS multiprocessing) ──
            import random as _rnd
            _rng = _rnd.Random(seed + n_qubits * 1000)
            _n_trials = max(inner_brackets * 3, 9)
            _best_F_cell = 0.0
            _best_cfg = None
            _best_agent_cell = None
            _short_budget = max(10_000, max_steps // 9)
            for _trial in range(_n_trials):
                _cfg = {
                    "alpha":        10 ** _rng.uniform(-3, -1.5),
                    "lr_actor":     10 ** _rng.uniform(-4, -2),
                    "lr_critic":    10 ** _rng.uniform(-4, -2),
                    "tau":          _rng.uniform(0.005, 0.05),
                    "batch_size":   _rng.choice([128, 256, 512]),
                    "hidden_dim":   _rng.choice([256, 384, 512]),
                    "cond_dim":     _rng.choice([32, 64, 96]),
                    "slm_interval": _rng.randint(10000, 100000),
                }
                try:
                    _trial_F, _trial_steps, _trial_hist, _trial_agent = run_cell(
                        n_qubits=n_qubits, target_state=target, seed=seed,
                        noise_stage=noise_stage, max_ep_steps=max_ep_steps,
                        reward_w_F=reward_w_F, reward_w_S=reward_w_S,
                        reward_w_str=reward_w_str,
                        alpha=float(_cfg["alpha"]),
                        lr_actor=float(_cfg["lr_actor"]),
                        lr_critic=float(_cfg["lr_critic"]),
                        tau=float(_cfg["tau"]),
                        batch_size=int(_cfg["batch_size"]),
                        hidden_dim=int(_cfg["hidden_dim"]),
                        cond_dim=int(_cfg["cond_dim"]),
                        max_steps=_short_budget,
                        init_agent=init_agent,
                        plateau_patience=max(5_000, _short_budget // 3),
                        slm_interval=int(_cfg["slm_interval"]),
                        der_buffer=der_buffer,
                    )
                    log.info(f"    HP trial {_trial+1}/{_n_trials}: best_F={_trial_F:.4f}")
                    if _trial_F > _best_F_cell:
                        _best_F_cell = _trial_F
                        _best_cfg = _cfg
                        _best_agent_cell = _trial_agent
                except Exception as _e:
                    log.warning(f"    HP trial {_trial+1}/{_n_trials}: SKIPPED (NaN/error: {_e})")
                    _trial_F = 0.0
            # Full-budget run with best config
            log.info(f"  Best HP (F={_best_F_cell:.4f}), running full budget {max_steps:,} steps...")
            best_F, _full_steps, _full_hist, _full_agent = run_cell(
                n_qubits=n_qubits, target_state=target, seed=seed,
                noise_stage=noise_stage, max_ep_steps=max_ep_steps,
                reward_w_F=reward_w_F, reward_w_S=reward_w_S,
                reward_w_str=reward_w_str,
                alpha=float(_best_cfg["alpha"]),
                lr_actor=float(_best_cfg["lr_actor"]),
                lr_critic=float(_best_cfg["lr_critic"]),
                tau=float(_best_cfg["tau"]),
                batch_size=int(_best_cfg["batch_size"]),
                hidden_dim=int(_best_cfg["hidden_dim"]),
                cond_dim=int(_best_cfg["cond_dim"]),
                max_steps=max_steps,
                init_agent=init_agent if _best_agent_cell is None else _best_agent_cell,
                plateau_patience=max(50_000, max_steps // 3),
                slm_interval=int(_best_cfg["slm_interval"]),
                der_buffer=der_buffer,
            )
            if _full_agent is not None:
                _best_agent_cell = _full_agent
            best_F = max(best_F, _best_F_cell)
            cell_results[target][seed] = best_F
            log.info(f"  ✓ {n_qubits}Q/{target}/s{seed} → best_F={best_F:.4f}")
            # Save per-cell result JSON
            cell_json = results_dir / f"{n_qubits}Q_{target}_s{seed}.json"
            cell_json.write_text(json.dumps({
                "n_qubits": n_qubits, "target": target, "seed": seed,
                "best_F": best_F,
                "noise_stage": noise_stage,
                "max_steps": max_steps,
                "hp_search": "random",
                "best_config": {k: float(v) if isinstance(v, float) else int(v)
                                for k, v in (_best_cfg or {}).items()},
            }, indent=2))

            # Track best agent across all cells at this qubit level
            if _best_agent_cell is not None and best_F > level_best_F:
                level_best_F     = best_F
                level_best_agent = _best_agent_cell

            # Add this cell's experience to DER++ buffer
            if _best_agent_cell is not None:
                _add_to_der_buffer(der_buffer, _best_agent_cell, n_qubits,
                                   target, seed, noise_stage, max_ep_steps,
                                   reward_w_F, reward_w_S, reward_w_str)

    # Save best agent checkpoint for this qubit level
    if level_best_agent is not None:
        ckpt_path = ckpt_dir / f"{n_qubits}Q_best_agent.pt"
        level_best_agent.save(str(ckpt_path))
        log.info(f"  Checkpoint saved: {ckpt_path}")

    return cell_results, level_best_agent


def _add_to_der_buffer(der_buffer: DERPlusPlusBuffer, agent: SACAgent,
                       n_qubits: int, target: str, seed: int,
                       noise_stage: int, max_ep_steps: int,
                       reward_w_F: float, reward_w_S: float,
                       reward_w_str: float, n_snapshot: int = 500):
    """Collect n_snapshot transitions from a trained agent and add to DER++ buffer."""
    _TARGET_NORM = {
        "GHZ": "ghz", "W": "w", "Cluster": "cluster_linear", "Dicke-k3": "dicke_k3",
    }
    target_norm = _TARGET_NORM.get(target, target.lower())
    _w_log = max(0.0, 1.0 - reward_w_F - reward_w_S - reward_w_str)
    env = QuasarEnv(n_qubits=n_qubits, target_state=target_norm,
                    max_steps=max_ep_steps, noise_curriculum_stage=noise_stage,
                    seed=seed, reward_weights=(reward_w_F, reward_w_S, reward_w_str, _w_log))
    target_vec = np.concatenate([env.target.real, env.target.imag]).astype(np.float32)
    obs = env.reset()
    for _ in range(n_snapshot):
        action = agent.select_action(obs, target_vec, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        der_buffer.add(obs, action, reward, next_obs, float(done), target_vec)
        obs = next_obs if not done else env.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Advancement decision
# ─────────────────────────────────────────────────────────────────────────────
def should_advance(cell_results: Dict[str, Dict[int, float]],
                   f_threshold: float, f_floor: float) -> Tuple[bool, float, float]:
    """
    Returns (advance, mean_F, min_F).
    Advances if mean_F >= f_threshold AND min_F >= f_floor.
    """
    all_F = [f for seed_dict in cell_results.values() for f in seed_dict.values()]
    if not all_F:
        return False, 0.0, 0.0
    mean_F = float(np.mean(all_F))
    min_F  = float(np.min(all_F))
    advance = (mean_F >= f_threshold) and (min_F >= f_floor)
    return advance, mean_F, min_F


# ─────────────────────────────────────────────────────────────────────────────
# Main adaptive loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QUASAR v13 Adaptive Open-Ended Scalability")
    parser.add_argument("--min-qubits",     type=int,   default=2)
    parser.add_argument("--max-qubits",     type=int,   default=12,
                        help="Hard ceiling (H100 memory limit)")
    parser.add_argument("--targets",        nargs="+",  default=["GHZ", "W", "Cluster", "Dicke-k3"])
    parser.add_argument("--seeds",          nargs="+",  type=int, default=[42, 123, 456])
    parser.add_argument("--f-threshold",    type=float, default=0.99,
                        help="Mean fidelity required to advance to next qubit count")
    parser.add_argument("--f-floor",        type=float, default=0.90,
                        help="Minimum per-cell fidelity required to advance")
    parser.add_argument("--inner-brackets", type=int,   default=4,
                        help="DEHB bracket count per (target, seed) cell")
    parser.add_argument("--max-steps-base", type=int,   default=500_000,
                        help="Max training steps at min-qubits")
    parser.add_argument("--steps-scale",    type=float, default=1.5,
                        help="Multiplier applied per additional qubit count")
    parser.add_argument("--noise-stage",    type=int,   default=5)
    parser.add_argument("--results-dir",    type=str,
                        default="/home/ralshehri0468/quasar_v13/results/v13_adaptive")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("QUASAR v13 — Adaptive Open-Ended Scalability Experiment")
    log.info(f"  Device:      {DEVICE}")
    log.info(f"  Qubit range: {args.min_qubits} → {args.max_qubits} (adaptive)")
    log.info(f"  Targets:     {args.targets}")
    log.info(f"  Seeds:       {args.seeds}")
    log.info(f"  F threshold: {args.f_threshold} (advance) / {args.f_floor} (floor)")
    log.info(f"  Max steps:   {args.max_steps_base} @ {args.min_qubits}Q, "
             f"×{args.steps_scale} per qubit")
    log.info(f"  Noise stage: {args.noise_stage}")
    log.info("=" * 70)

    # ── Shared DER++ buffer (persists across all qubit levels) ───────────────
    der_buffer = None  # will be initialised inside the loop once env dims are known

    # ── Per-qubit tracking ───────────────────────────────────────────────────
    scalability_report = {
        "config": vars(args),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qubit_levels": {},
        "scalability_ceiling": None,
        "ceiling_reason": None,
    }

    prev_best_agent: Optional[SACAgent] = None
    ceiling_reached = False

    for n_qubits in range(args.min_qubits, args.max_qubits + 1):
        # Compute step budget for this qubit count (scales with complexity)
        extra = n_qubits - args.min_qubits
        max_steps = int(args.max_steps_base * (args.steps_scale ** extra))
        max_steps = min(max_steps, 5_000_000)  # hard cap at 5M steps

        log.info("")
        log.info("=" * 70)
        log.info(f"  QUBIT LEVEL: {n_qubits}Q  (budget={max_steps:,} steps)")
        log.info("=" * 70)

        t0 = time.time()
        # Lazy-init DER++ buffer — compute dims directly from n_qubits formula
        if der_buffer is None:
            _obs_dim = 4 * (2 ** args.min_qubits) + args.min_qubits + 2
            _act_dim = args.min_qubits * 3
            der_buffer = DERPlusPlusBuffer(
                obs_dim=_obs_dim, act_dim=_act_dim, capacity=10_000
            )
            log.info(f'DER++ buffer init: obs_dim={_obs_dim}, act_dim={_act_dim}')
        cell_results, best_agent = run_qubit_level(
            n_qubits=n_qubits,
            targets=args.targets,
            seeds=args.seeds,
            noise_stage=args.noise_stage,
            max_steps=max_steps,
            inner_brackets=args.inner_brackets,
            results_dir=results_dir,
            ckpt_dir=ckpt_dir,
            init_agent=prev_best_agent,
            der_buffer=der_buffer,
        )
        elapsed = time.time() - t0

        advance, mean_F, min_F = should_advance(cell_results, args.f_threshold, args.f_floor)

        # Per-target mean for reporting
        per_target_mean = {
            t: float(np.mean(list(seed_dict.values())))
            for t, seed_dict in cell_results.items()
        }

        level_summary = {
            "n_qubits":       n_qubits,
            "mean_F":         round(mean_F, 5),
            "min_F":          round(min_F, 5),
            "per_target_mean": {k: round(v, 5) for k, v in per_target_mean.items()},
            "cell_results":   {t: {str(s): round(f, 5) for s, f in sd.items()}
                               for t, sd in cell_results.items()},
            "max_steps_used": max_steps,
            "elapsed_s":      round(elapsed, 1),
            "advanced":       advance,
        }
        scalability_report["qubit_levels"][str(n_qubits)] = level_summary

        log.info(f"  {n_qubits}Q SUMMARY: mean_F={mean_F:.4f}  min_F={min_F:.4f}")
        for t, v in per_target_mean.items():
            log.info(f"    {t}: mean_F={v:.4f}")

        # Write incremental report after every qubit level
        report_path = results_dir / "scalability_report.json"
        report_path.write_text(json.dumps(scalability_report, indent=2))

        if advance:
            log.info(f"  ✓ {n_qubits}Q PASSED (mean_F={mean_F:.4f} >= {args.f_threshold}) "
                     f"→ advancing to {n_qubits + 1}Q")
            prev_best_agent = best_agent
        else:
            reason = (f"mean_F={mean_F:.4f} < threshold={args.f_threshold}"
                      if mean_F < args.f_threshold
                      else f"min_F={min_F:.4f} < floor={args.f_floor}")
            log.info(f"  ✗ {n_qubits}Q STALLED ({reason})")
            log.info(f"  ══ SCALABILITY CEILING REACHED: N* = {n_qubits - 1}Q ══")
            scalability_report["scalability_ceiling"] = n_qubits - 1
            scalability_report["ceiling_reason"]      = reason
            ceiling_reached = True
            break

    if not ceiling_reached:
        scalability_report["scalability_ceiling"] = args.max_qubits
        scalability_report["ceiling_reason"]      = f"Reached hard max-qubits={args.max_qubits}"
        log.info(f"  ══ EXPERIMENT COMPLETE: N* >= {args.max_qubits}Q (hard ceiling) ══")

    scalability_report["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report_path = results_dir / "scalability_report.json"
    report_path.write_text(json.dumps(scalability_report, indent=2))

    log.info("")
    log.info("=" * 70)
    log.info(f"  FINAL RESULT: Scalability ceiling N* = "
             f"{scalability_report['scalability_ceiling']}Q")
    log.info(f"  Report: {report_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
