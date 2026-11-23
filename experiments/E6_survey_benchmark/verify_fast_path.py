#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the benchmark fast path reproduces native SiliQunEnv values."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np
from siliqun.engine.gym_env import make_siliqun_env
from bench_env_v5 import _install_fast_path, make_bench_env, GATE_NAMES


def make_raw(seed):
    return make_siliqun_env(n_qubits=2, device="simos", target="ghz",
                            sim_mode="sv", noise=True, max_steps=200,
                            fidelity_threshold=0.99, reward_type="shaped",
                            use_gpu=False, seed=seed)


def box_for(token):
    from bench_env_v5 import _TOKEN_MAP
    gate_idx, q_logits, theta = _TOKEN_MAP[int(token)]
    a = np.zeros(9, dtype=np.float32)
    a[gate_idx] = 1.0
    a[5] = q_logits[0]
    a[6] = q_logits[1]
    a[7] = theta
    return a


def main():
    raw = make_raw(1)
    fast = make_bench_env(seed=1)
    fast.env._prev_fidelity = 0.25  # match wrapper convention for both
    raw._prev_fidelity = 0.25

    rng = np.random.RandomState(42)
    max_obs_diff = 0.0
    max_fid_diff = 0.0
    n = 400
    for i in range(n):
        raw.reset(seed=i)
        fast.env.reset(seed=i)
        obs_r = raw._get_obs()
        obs_f = fast.env._get_obs()
        max_obs_diff = max(max_obs_diff, float(np.max(np.abs(obs_r - obs_f))))
        for _ in range(5):
            tok = int(rng.randint(7))
            obs_r, r_r, _, _, info_r = raw.step(box_for(tok))
            obs_f, r_f, _, _, info_f = fast.env.step(box_for(tok))
            max_obs_diff = max(max_obs_diff, float(np.max(np.abs(obs_r - obs_f))))
            max_fid_diff = max(max_fid_diff, abs(info_r["fidelity"] - info_f["fidelity"]))
            assert abs(r_r - r_f) < 1e-12, (r_r, r_f)
    print(f"max obs diff over {n} episodes: {max_obs_diff:.3e}")
    print(f"max fidelity diff:              {max_fid_diff:.3e}")

    # solvability through the fast path
    fast2 = make_bench_env(seed=7)
    fast2.reset(seed=7)
    for t in (0, 6, 5):
        obs, r, term, trunc, info = fast2.step(t)
    print(f"3-gate solution F = {info['fidelity']:.6f} (term={term})")

    # speed
    import time
    fast3 = make_bench_env(seed=7)
    fast3.reset()
    for _ in range(200):
        fast3.step(0)
    t0 = time.time()
    for i in range(5000):
        fast3.step(i % 7)
    dt = time.time() - t0
    print(f"env step: {1e3*dt/5000:.2f} ms/step")


if __name__ == "__main__":
    main()
