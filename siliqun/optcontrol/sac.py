"""Self-contained SAC environment + agent for the E5 robustness sweep.

The v6 E4 harness (``siliqun.envs.SiliQunEnv``) belonged to an older
package revision and is not importable against the current ``siliqun``
physics.  This module rebuilds the same *protocol* on the current
physics, one segment per environment step:

* observation: [k/N, v_{k-1}(5), state populations(4), running fidelity]
* action: 5-dim, tanh output scaled to [-pi, pi] (as in E4), mapped to
  the normalized segment-control vector for step k
* reward: state fidelity after each segment (dense shaping)
* horizon: N = 200 segments x 1 ns = 200 ns gate

The agent mirrors the E4 SAC: 2x256 ReLU actor/critic, twin critics,
soft targets (tau=5e-3), automatic temperature (alpha0=0.2),
gamma=0.99, batch 256, Adam 3e-4, 10k uniform warmup steps.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from scipy.linalg import expm

from .core import (
    H_segment,
    input_state_plus_plus,
    state_fidelity,
    target_bell_plus,
    OMEGA_MAX_HZ,
    J_MAX_HZ,
)

LOG_STD_MAX, LOG_STD_MIN = 2.0, -20.0

N_SEGMENTS = 200
DT = 1e-9
OBS_DIM = 11     # 1 + 5 + 4 + 1
ACT_DIM = 5


class SACEnv:
    """One-segment-step gym-style environment on SiliQun physics."""

    def __init__(self, dt: float = DT, n_segments: int = N_SEGMENTS,
                 noise_draw=None, t2_star: float | None = None,
                 t1: float | None = None, seed: int = 0):
        self.dt = dt
        self.n = n_segments
        self.noise_draw = noise_draw
        self.t2_star = t2_star
        self.t1 = t1
        self.target = target_bell_plus()
        self.eta_pop = math.exp(-dt / t2_star) if t2_star else 1.0
        self.p_relax = 1.0 - math.exp(-dt / t1) if t1 else 0.0
        self._reset_state(seed)

    def _reset_state(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.psi = input_state_plus_plus()
        self.rho = None
        self.k = 0
        self.v_prev = np.zeros(ACT_DIM)
        self.v_history = []

    def reset(self, seed=None):
        self._reset_state(seed)
        return self._obs()

    def _obs(self):
        rho = self.rho if self.rho is not None else np.outer(self.psi, self.psi.conj())
        pop = np.real(np.diag(rho))
        fid = float(state_fidelity(rho, self.target))
        return np.concatenate(
            [[self.k / self.n], self.v_prev, pop, [fid]]
        ).astype(np.float32)

    def step(self, action: np.ndarray):
        """action: raw tanh output in [-pi, pi]; mapped to normalized v."""
        a = np.clip(action, -np.pi, np.pi)
        v = (np.tanh(a) + 1.0) / 2.0
        self.v_prev = v
        self.v_history.append(v)
        omega1 = v[0] * OMEGA_MAX_HZ
        phi1 = (2 * v[1] - 1) * np.pi
        omega2 = v[2] * OMEGA_MAX_HZ
        phi2 = (2 * v[3] - 1) * np.pi
        jseg = v[4] * J_MAX_HZ
        H = H_segment(omega1, phi1, omega2, phi2, jseg, self.noise_draw)
        self.psi = expm(-1j * H * self.dt) @ self.psi
        if self.t2_star or self.t1:
            if self.rho is None:
                self.rho = np.outer(self.psi, self.psi.conj())
            else:
                self.rho = expm(-1j * H * self.dt) @ self.rho @ expm(1j * H * self.dt)
            # per-step phase damping (coherences between different-pop
            # levels decay by eta_pop^(d_pop/2))
            pop = np.array([0, 1, 1, 2])
            f = np.exp(0.5 * (pop[:, None] + pop[None, :]) * np.log(self.eta_pop))
            self.rho = self.rho * f
            # per-step single-site amplitude damping, kron over both sites
            q, q0 = math.sqrt(self.p_relax), math.sqrt(1.0 - self.p_relax)
            K0 = np.array([[1.0, 0.0], [0.0, q0]])
            K1 = np.array([[q, 0.0], [0.0, 0.0]])
            out = np.zeros((4, 4), dtype=complex)
            for A in (K0, K1):
                for B in (K0, K1):
                    K = np.kron(A, B)
                    out += K @ self.rho @ K.conj().T
            self.rho = out
        rho = self.rho if self.rho is not None else np.outer(self.psi, self.psi.conj())
        fid = float(state_fidelity(rho, self.target))
        self.k += 1
        done = self.k >= self.n
        return self._obs(), float(fid), bool(done), bool(done), {"fidelity": fid}

    def episode_controls(self) -> np.ndarray:
        return np.concatenate(self.v_history) if self.v_history else np.zeros(5 * self.n)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        return self.mu_head(h), self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        y = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y * math.pi, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity=1_000_000, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.nobs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.ptr, self.size = 0, 0

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.nobs[self.ptr] = no
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        out = {
            "obs": self.obs[idx],
            "next_obs": self.nobs[idx],
            "actions": self.act[idx],
            "rewards": self.rew[idx],
            "dones": self.done[idx],
        }
        return {k: torch.FloatTensor(v) for k, v in out.items()}


class SACAgent:
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256,
                 lr_actor=3e-4, lr_critic=3e-4, tau=0.005,
                 gamma=0.99, batch_size=256, device="cpu", seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_t = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_actor)
        self.target_entropy = -act_dim
        self.buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim)

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if deterministic:
                mu, _ = self.actor(o)
                a = torch.tanh(mu) * math.pi
            else:
                a, _ = self.actor.sample(o)
        return a.cpu().numpy().flatten()

    def update(self):
        if self.buffer.size < self.batch_size:
            return None
        d = self.buffer.sample(self.batch_size)
        batch = {k: t.to(self.device) for k, t in d.items()}
        na, nlp = self.actor.sample(batch["next_obs"])
        q1t, q2t = self.critic_t(batch["next_obs"], na)
        qn = torch.min(q1t, q2t) - self.log_alpha.exp() * nlp
        qt = batch["rewards"] + self.gamma * (1 - batch["dones"]) * qn
        q1, q2 = self.critic(batch["obs"], batch["actions"])
        self.critic_opt.zero_grad()
        cl = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
        cl.backward()
        self.critic_opt.step()
        a, lp = self.actor.sample(batch["obs"])
        qa1, qa2 = self.critic(batch["obs"], a)
        qa = torch.min(qa1, qa2)
        al = (self.log_alpha.exp() * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        al.backward()
        self.alpha_opt.step()
        self.actor_opt.zero_grad()
        actor_loss = (lp - qa).mean()
        actor_loss.backward()
        self.actor_opt.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.add_((p - pt) * self.tau)
        return float(cl.detach())


def train_sac(n_segments: int = N_SEGMENTS, dt: float = DT, seed: int = 0,
              max_env_steps: int = 300_000, warmup_steps: int = 10_000,
              log_interval: int = 25_000, device: str = "cpu",
              verbose: bool = True) -> dict:
    """Train SAC to synthesize the 2-qubit CZ (|++> -> |Phi+>) on SiliQun.

    Returns {"vec": (5N,) normalized controls, "fidelity": best
    stochastic-episode fidelity, "trace": [(step, best_F), ...]}.
    """
    env = SACEnv(dt=dt, n_segments=n_segments, seed=seed)
    agent = SACAgent(device=device, seed=seed)
    obs = env.reset(seed=seed)
    trace = []
    best_F = 0.0
    best_v = None
    last_info = {"fidelity": 0.0}
    for step in range(1, max_env_steps + 1):
        if step < warmup_steps:
            a = env._rng.uniform(-np.pi, np.pi, ACT_DIM).astype(np.float32)
        else:
            a = agent.select_action(obs)
        nobs, r, done, _, info = env.step(a)
        last_info = info
        agent.buffer.add(obs, a, r, nobs, float(done))
        if done:
            F = info["fidelity"]
            if F > best_F:
                best_F = F
                best_v = env.episode_controls()
            obs = env.reset(seed=seed + step)
        else:
            obs = nobs
        if step >= warmup_steps:
            agent.update()
        if verbose and step % log_interval == 0:
            trace.append((step, best_F))
            print(f"    sac s{seed} step={step:,} best_F={best_F:.4f}", flush=True)
    if best_v is None:
        obs = env.reset(seed=seed)
        for _ in range(n_segments):
            a = agent.select_action(obs, deterministic=True)
            obs, r, done, _, info = env.step(a)
            last_info = info
        best_v = env.episode_controls()
        best_F = last_info["fidelity"]
    return {"vec": best_v, "fidelity": best_F, "trace": trace}
