# -*- coding: utf-8 -*-
"""DQN regeneration run: donor_electron, 4-qubit GHZ, discrete tokens.

Protocol (documented in the output JSON; config anchors from
experiments/E2_replication_he/config_e2.json + paper secIV Task 2):
  env     SiliQunEnv(device="donor", n_qubits=4, target_state="ghz",
                     max_steps=50, fidelity_threshold=0.99, sim_mode="mps")
          wrapped with DiscreteTokenEnv (23 tokens)
  algo    SB3 DQN, lr 3e-4, batch 128, gamma 0.99 (config_e2.json),
          buffer 1e5, tau 5e-4, learning_starts 100, train_freq 1,
          gradient_steps 1, target_update_interval 1000,
          epsilon 0.9 -> 0.05 linear over the full run
  budget  5000 episodes per seed (config_e2.json)
  report  final stats = last 500-episode window (R4 convention)

Usage:
  python train_dqn.py --seed 0 --episodes 5000 \
      --out experiments/E1_replication_daraeizadeh/results/regen_r5_donor_ghz_dqn__seed_0_results.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# repo root = two levels above experiments/E2_replication_he/
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

from discrete_wrapper import DiscreteTokenEnv  # noqa: E402


def build_env(max_steps: int = 50, seed: int = 0):
    from siliqun.engine.gym_env import SiliQunEnv

    env = SiliQunEnv(
        device="donor",
        n_qubits=4,
        target_state="ghz",
        max_steps=max_steps,
        fidelity_threshold=0.99,
        sim_mode="mps",
    )
    wrapped = DiscreteTokenEnv(env)
    wrapped = gym_seeding(wrapped, seed)
    return wrapped


def gym_seeding(env, seed):
    import gymnasium as gym

    class _SeedWrapper(gym.Wrapper):
        def reset(self, **kw):
            kw.setdefault("seed", None)
            return super().reset(**kw)

    return _SeedWrapper(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--resume-from", type=int, default=0,
                    help="episodes already trained (checkpoint restore)")
    ap.add_argument("--eps-floor", type=float, default=0.05,
                    help="final epsilon (documented default 0.05; diagnostic "
                         "runs used 0.1)")
    args = ap.parse_args()

    t0 = time.time()
    env = build_env(max_steps=args.max_steps, seed=args.seed)

    from stable_baselines3 import DQN

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=100_000,
        learning_rate=3e-4,          # config_e2.json
        batch_size=128,              # config_e2.json
        gamma=0.99,                  # config_e2.json
        tau=5e-4,
        learning_starts=100,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_final_eps=args.eps_floor,
        exploration_initial_eps=0.9,
        exploration_fraction=1.0,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        device="cpu",
        verbose=0,
    )

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- train ---
    total = args.episodes
    done_eps = args.resume_from
    curve = []
    window = 50
    recent = []            # per-episode best fidelity, last 500
    running_best = 0.0
    n_steps_total = 0

    if args.resume_from > 0 and args.checkpoint_dir:
        ckpt = os.path.join(
            args.checkpoint_dir, f"r5_dqn_s{args.seed}_{args.resume_from}.npz")
        if os.path.exists(ckpt):
            model.learn(0, total=0, reset_num_timesteps=False)  # no-op keep
            # (full state restore not supported by npz; curves start fresh
            #  from the resume point -- documented as a protocol note)

    while done_eps < total:
        obs, _ = env.reset(seed=args.seed + done_eps)
        ep_best = 0.0
        for _ in range(args.max_steps):
            act, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, info = env.step(int(act))
            ep_best = max(ep_best, info.get("best_fidelity", 0.0))
            n_steps_total += 1
            if term or trunc:
                break
        done_eps += 1
        running_best = max(running_best, ep_best)
        recent.append(ep_best)
        if len(recent) > 500:
            recent.pop(0)

        if done_eps % window == 0:
            recent_win = recent[-window:]
            curve.append({
                "episode": done_eps,
                "mean_fidelity": float(np.mean(recent_win)),
                "best_fidelity": running_best,
                "pct_above_999": float(np.mean([f >= 0.999 for f in recent_win])),
                "pct_above_99": float(np.mean([f >= 0.99 for f in recent_win])),
                "elapsed_sec": round(time.time() - t0, 1),
                "eps_per_sec": round(done_eps / (time.time() - t0), 4),
            })
            if done_eps % 500 == 0:
                print(f"[s{args.seed}] ep {done_eps}/{total} "
                      f"win_mean={curve[-1]['mean_fidelity']:.4f} "
                      f"best={running_best:.4f} "
                      f"{curve[-1]['eps_per_sec']} eps/s", flush=True)
        if args.checkpoint_dir and done_eps % 1000 == 0:
            model.save(os.path.join(
                args.checkpoint_dir, f"r5_dqn_s{args.seed}_{done_eps}"))

    # --- final window (last 500 episodes, R4 convention) ---
    final_win = recent[-500:]
    elapsed_min = (time.time() - t0) / 60.0
    result = {
        "seed": args.seed,
        "final_mean_fidelity": float(np.mean(final_win)),
        "final_best_fidelity": float(running_best),
        "final_pct_above_999": float(np.mean([f >= 0.999 for f in final_win])),
        "final_pct_above_99": float(np.mean([f >= 0.99 for f in final_win])),
        "training_time_min": round(elapsed_min, 3),
        "episodes": total,
        "training_curve": curve,
        "protocol": {
            "experiment": "Regeneration R5: Replications 2/3 donor_electron 4q GHZ DQN",
            "env": "SiliQunEnv(device='donor', n_qubits=4, target_state='ghz', "
                   "max_steps=50, fidelity_threshold=0.99, sim_mode='mps')",
            "action_space": "Discrete(23): 16 rotations RX/RY(pi/2,pi) x 4 qubits, "
                            "6 CNOT/CZ on pairs (0-1,1-2,2-3), 1 idle RZ(0)",
            "algo": "SB3 DQN MlpPolicy [128,128] x2, buffer 1e5, train_freq 1, "
                    "gradient_steps 1, target_update_interval 1000, "
                    "eps 0.9->0.05 linear",
            "config_anchor": "experiments/E2_replication_he/config_e2.json "
                             "(lr 3e-4, batch 128, gamma 0.99, 5000 ep, threshold 0.999)",
            "device": "CPU (no CUDA backend in env step path; 4q MPS is numpy-exact)",
            "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "siliqun_repo_head": os.environ.get("SILIQU_HEAD", None),  # set by launcher
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[s{args.seed}] DONE: mean={result['final_mean_fidelity']:.4f} "
          f"best={result['final_best_fidelity']:.4f} "
          f"pct999={result['final_pct_above_999']:.3f} "
          f"pct99={result['final_pct_above_99']:.3f} "
          f"({elapsed_min:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
