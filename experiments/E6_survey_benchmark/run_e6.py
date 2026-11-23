#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E6 runner: survey benchmark (bench_env_v5) - 7 algorithms x 5 seeds.

Usage:
    python run_e6.py --chunk 0/4     # run 1/4 of the matrix (parallel workers)
    python run_e6.py --algo dqn --seed 0   # single run
    python run_e6.py --random-baseline     # random-policy baseline (200 episodes)

Each run trains an SB3 policy for 500,000 environment steps on the
2-qubit GHZ task with held-out evaluation every 2,500 steps, and writes
a per-run JSON receipt to results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import platform

os.environ.setdefault("OMP_NUM_THREADS", "4")
_nt = int(os.environ.get("TORCH_THREADS", "4"))
import numpy as np
import torch
torch.set_num_threads(_nt)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root (siliqun pkg)

from bench_env_v5 import (  # noqa: E402
    make_bench_env, BenchHistoryEnv, BenchRelaxationEnv, N_TOKENS,
)

ALGOS = ["dqn", "dqn_nas", "ppo", "ppo_ent", "ppo_hier", "sac", "ddpg"]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 500_000
EVAL_EVERY = 2_500
EVAL_NOISE_SEED = 12345
N_EVAL_EPISODES = 10
MAX_EVAL_GATES = 20
FIDELITY_THRESHOLD = 0.99
FAILURE_THRESHOLD = 0.75

BASE = dict(learning_rate=3e-4, batch_size=128, gamma=0.99)
RESULTS = os.path.join(HERE, "results")
# E6_DEVICE override: CPU is the default on this host — measured 850
# steps/s CPU vs 508 steps/s CUDA for this 2-layer MLP (per-step H2D/D2H
# latency dominates). Set E6_DEVICE=cuda to force the GPU.
DEVICE = os.environ.get("E6_DEVICE", "cpu")


def env_instance_for(algo: str, seed: int):
    """Fresh env instance with run-seeded noise, wrapper per algorithm."""
    env = make_bench_env(seed=seed)
    if algo == "ppo_hier":
        env = BenchHistoryEnv(env)
    elif algo in ("ddpg", "sac"):
        # SB3 (this build) SAC only supports Box action spaces; both
        # off-policy rows therefore run on the same Box(7,) continuous
        # relaxation (argmax selects the gate). SAC adds the stochastic
        # squashed-Gaussian policy + max-entropy objective.
        env = BenchRelaxationEnv(env)
    return env


def build_model(algo: str, seed: int):
    from stable_baselines3 import DQN, PPO, SAC, DDPG
    env = env_instance_for(algo, seed)
    if algo == "dqn":
        m = DQN("MlpPolicy", env, **BASE, target_update_interval=1000,
                policy_kwargs=dict(net_arch=[256, 256]), device=DEVICE)
    elif algo == "dqn_nas":
        m = DQN("MlpPolicy", env, **BASE, target_update_interval=1000,
                policy_kwargs=dict(net_arch=[512, 512, 512]), device=DEVICE)
    elif algo == "ppo":
        m = PPO("MlpPolicy", env, **BASE, n_steps=2048, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[256, 256]), device=DEVICE)
    elif algo == "ppo_ent":
        m = PPO("MlpPolicy", env, **BASE, n_steps=2048, ent_coef=0.05,
                policy_kwargs=dict(net_arch=[256, 256]), device=DEVICE)
    elif algo == "ppo_hier":
        m = PPO("MlpPolicy", env, **BASE, n_steps=2048, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[256, 256]), device=DEVICE)
    elif algo == "sac":
        # SB3 SAC always learns the temperature (alpha) — no flag needed
        m = SAC("MlpPolicy", env, **BASE,
                policy_kwargs=dict(net_arch=[256, 256]), device=DEVICE)
    elif algo == "ddpg":
        m = DDPG("MlpPolicy", env, **BASE,
                 policy_kwargs=dict(net_arch=[256, 256]), device=DEVICE)
    else:
        raise ValueError(algo)
    m.set_random_seed(seed)
    return m


def held_out_eval(algo: str) -> float:
    """Mean best fidelity over 10 held-out episodes (20-gate cap,
    rotating noise seeds). See config_e6.json for the protocol rationale."""
    env = _eval_env_for(algo)
    vals = []
    for i in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=EVAL_NOISE_SEED + i)
        best = 0.0
        for _ in range(MAX_EVAL_GATES):
            action, _ = policy_holder["model"].predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            best = max(best, float(info.get("fidelity", 0.0)))
            if terminated or truncated:
                break
        vals.append(best)
    env.close()
    return float(np.mean(vals))


def _eval_env_for(algo: str):
    env = make_bench_env(seed=EVAL_NOISE_SEED)
    if algo == "ppo_hier":
        env = BenchHistoryEnv(env)
    elif algo in ("ddpg", "sac"):
        env = BenchRelaxationEnv(env)
    return env


policy_holder: dict = {}


class EvalCB:
    """Lightweight eval loop driven from train() to avoid callback kwargs drift."""
    pass


def run_single(algo: str, seed: int) -> dict:
    from stable_baselines3 import DQN, PPO, SAC, DDPG
    t0 = time.time()
    model = build_model(algo, seed)
    policy_holder["model"] = model

    # Training loop with periodic held-out eval (finer than SB3 callbacks)
    eval_curve = []
    done = 0
    chunk = 50_000
    next_eval = EVAL_EVERY
    while done < TOTAL_STEPS:
        n = min(chunk, TOTAL_STEPS - done)
        model.learn(n, reset_num_timesteps=False)
        done = model.num_timesteps
        while next_eval <= done:
            f = held_out_eval(algo)
            eval_curve.append({"step": next_eval, "F": f})
            next_eval += EVAL_EVERY
        print(f"[{algo} s{seed}] {done}/{TOTAL_STEPS} "
              f"(last eval F={eval_curve[-1]['F'] if eval_curve else float('nan'):.4f}) "
              f"wall={time.time()-t0:.0f}s", flush=True)

    final_F = held_out_eval(algo)
    wall = time.time() - t0

    solved_steps = [e["step"] for e in eval_curve if e["F"] >= FIDELITY_THRESHOLD]
    best_F = max([e["F"] for e in eval_curve] + [final_F])
    receipt = {
        "experiment": "E6_survey_benchmark",
        "algo": algo,
        "seed": seed,
        "total_timesteps": TOTAL_STEPS,
        "eval_every": EVAL_EVERY,
        "first_solve_step": solved_steps[0] if solved_steps else None,
        "solved": bool(solved_steps),
        "final_F": final_F,
        "best_F": best_F,
        "failed": best_F < FAILURE_THRESHOLD,
        "eval_curve": eval_curve,
        "wall_seconds": wall,
        "host": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cpu": platform.processor() or "unknown",
            "device": DEVICE,
            "cuda": torch.version.cuda if DEVICE == "cuda" else None,
            "gpu": (torch.cuda.get_device_name(0)
                    if DEVICE == "cuda" else None),
        },
        "config_ref": "config_e6.json",
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"e6_{algo}_s{seed}.json")
    with open(out, "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(f"[{algo} s{seed}] DONE solved={receipt['solved']} "
          f"first_solve={receipt['first_solve_step']} final_F={final_F:.4f} "
          f"failed={receipt['failed']} -> {out}", flush=True)
    return receipt


def random_baseline(n_episodes: int = 200) -> dict:
    """Random-policy baseline under the eval protocol (20-gate cap)."""
    env = make_bench_env(seed=7)
    fids = []
    for ep in range(n_episodes):
        obs = env.reset(seed=1000 + ep)[0]
        best = 0.0
        for _ in range(MAX_EVAL_GATES):
            a = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(a)
            best = max(best, float(info.get("fidelity", 0.0)))
            if terminated or truncated:
                break
        fids.append(best)
    out = {
        "experiment": "E6_survey_benchmark",
        "protocol": (f"random tokens, best fidelity over <= {MAX_EVAL_GATES} "
                     f"gates, {n_episodes} episodes (eval-protocol cap)"),
        "n_episodes": n_episodes,
        "mean_best_F": float(np.mean(fids)),
        "std_best_F": float(np.std(fids)),
        "fidelity_max": float(np.max(fids)),
        "frac_ge_threshold": float(np.mean(np.array(fids) >= FIDELITY_THRESHOLD)),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "e6_random_baseline.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", help="i/n : run the i-th of n round-robin shards")
    ap.add_argument("--algo", choices=ALGOS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--random-baseline", action="store_true")
    args = ap.parse_args()

    if args.random_baseline:
        random_baseline()
        return

    if args.algo and args.seed is not None:
        jobs = [(args.algo, args.seed)]
    elif args.chunk:
        i, n = map(int, args.chunk.split("/"))
        full = [(a, s) for a in ALGOS for s in SEEDS]
        jobs = full[i::n]
    else:
        jobs = [(a, s) for a in ALGOS for s in SEEDS]

    for algo, seed in jobs:
        run_single(algo, seed)


if __name__ == "__main__":
    main()
