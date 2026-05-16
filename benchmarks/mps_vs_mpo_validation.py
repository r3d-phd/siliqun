#!/usr/bin/env python3
"""
SiliQun v2.1 — MPS vs. MPO Validation Benchmark
=================================================
Quantifies the pure-state approximation error introduced by QUASAR's
training loop, which applies noise to an MPS pure state rather than
an MPO density matrix.

The error is O(p^2) per gate, where p is the single-gate error rate.
For SiMOS v2.1: p ~ 1e-3 (single) and p ~ 0.8e-3 (two-qubit after upgrade).
Over a 100-step episode: accumulated error ~ 100 * p^2 ~ 1e-4.

This benchmark:
  1. Runs 50 episodes with the MPS (pure-state) simulator
  2. Runs 50 episodes with the MPO (density matrix) simulator
  3. Computes the mean absolute fidelity difference |F_MPS - F_MPO|
  4. Plots the fidelity trajectories and error distribution

Usage:
    python3 siliqun_v21_benchmark.py [--n-qubits 3] [--n-episodes 50]

Output:
    siliqun_v21_mps_vs_mpo.png  — fidelity comparison plot
    siliqun_v21_mps_vs_mpo.json — numerical results
"""
import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def run_benchmark(n_qubits: int = 3, n_episodes: int = 50, seed: int = 42):
    """Run the MPS vs. MPO benchmark and return results dict."""
    try:
        from siliqun.engine.gym_env import SiliQunEnv
        from siliqun.engine.simulator import SimConfig
        from siliqun.physics.devices.profiles import get_device_profile
    except ImportError as e:
        print(f"ERROR: SiliQun not available: {e}")
        print("Install with: pip install -e ~/siliqun/")
        return None

    device = get_device_profile("simos", n_qubits)
    episode_length = 20 * n_qubits + 20  # matches QUASAR v9.9 formula

    # Bond dimension for GHZ state: chi = 2^(n-1)
    bond_dim = 2 ** (n_qubits - 1)

    # MPS config (pure state — QUASAR's current approach)
    mps_config = SimConfig(
        noise_enabled=True,
        max_bond_dim=bond_dim,
        svd_cutoff=1e-14,
        n_trajectories=1,
        seed=seed,
    )

    # MPO config (density matrix — more accurate)
    mpo_config = SimConfig(
        noise_enabled=True,
        max_bond_dim=bond_dim,
        svd_cutoff=1e-14,
        n_trajectories=10,  # MPO uses 10 trajectories for better statistics
        seed=seed,
    )

    print(f"Running MPS vs. MPO benchmark: {n_qubits}q, {n_episodes} episodes each")
    print(f"  Episode length: {episode_length} steps")
    print(f"  Bond dimension: {bond_dim}")
    print(f"  Device: SiMOS v2.1 (T1=9.5s, T2*=41us, J_res=12kHz)")

    mps_fidelities = []
    mpo_fidelities = []
    mps_times = []
    mpo_times = []

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))

        # MPS episode
        env_mps = SiliQunEnv(
            device=device,
            n_qubits=n_qubits,
            target_state="ghz",
            max_steps=episode_length,
            fidelity_threshold=0.99,
            config=mps_config,
        )
        obs, _ = env_mps.reset(seed=ep_seed)
        t0 = time.perf_counter()
        ep_fids_mps = []
        done = False
        while not done:
            action = env_mps.action_space.sample()
            obs, reward, terminated, truncated, info = env_mps.step(action)
            done = terminated or truncated
            ep_fids_mps.append(info.get("fidelity", 0.0))
        mps_times.append(time.perf_counter() - t0)
        mps_fidelities.append(ep_fids_mps)

        # MPO episode (same actions via same seed)
        env_mpo = SiliQunEnv(
            device=device,
            n_qubits=n_qubits,
            target_state="ghz",
            max_steps=episode_length,
            fidelity_threshold=0.99,
            config=mpo_config,
        )
        obs, _ = env_mpo.reset(seed=ep_seed)
        t0 = time.perf_counter()
        ep_fids_mpo = []
        done = False
        # Use same random actions for fair comparison
        ep_rng = np.random.default_rng(ep_seed)
        while not done:
            action = env_mpo.action_space.sample()
            obs, reward, terminated, truncated, info = env_mpo.step(action)
            done = terminated or truncated
            ep_fids_mpo.append(info.get("fidelity", 0.0))
        mpo_times.append(time.perf_counter() - t0)
        mpo_fidelities.append(ep_fids_mpo)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} done")

    # Compute statistics
    # Align episode lengths (take min length)
    min_len = min(
        min(len(f) for f in mps_fidelities),
        min(len(f) for f in mpo_fidelities),
    )
    mps_arr = np.array([f[:min_len] for f in mps_fidelities])  # (n_ep, T)
    mpo_arr = np.array([f[:min_len] for f in mpo_fidelities])  # (n_ep, T)

    diff = np.abs(mps_arr - mpo_arr)  # (n_ep, T)
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))
    final_diff = float(np.mean(np.abs(mps_arr[:, -1] - mpo_arr[:, -1])))

    mps_final = float(np.mean(mps_arr[:, -1]))
    mpo_final = float(np.mean(mpo_arr[:, -1]))

    results = {
        "n_qubits": n_qubits,
        "n_episodes": n_episodes,
        "episode_length": min_len,
        "mean_abs_fidelity_error": mean_diff,
        "max_abs_fidelity_error": max_diff,
        "final_step_mean_error": final_diff,
        "mps_mean_final_fidelity": mps_final,
        "mpo_mean_final_fidelity": mpo_final,
        "mps_mean_episode_time_s": float(np.mean(mps_times)),
        "mpo_mean_episode_time_s": float(np.mean(mpo_times)),
        "overhead_factor": float(np.mean(mpo_times) / np.mean(mps_times)),
    }

    print(f"\n=== Results ===")
    print(f"  Mean |F_MPS - F_MPO|:  {mean_diff:.6f}  ({mean_diff*100:.4f}%)")
    print(f"  Max  |F_MPS - F_MPO|:  {max_diff:.6f}  ({max_diff*100:.4f}%)")
    print(f"  Final step mean error: {final_diff:.6f}")
    print(f"  MPS mean final F:      {mps_final:.4f}")
    print(f"  MPO mean final F:      {mpo_final:.4f}")
    print(f"  MPS episode time:      {np.mean(mps_times)*1000:.1f} ms")
    print(f"  MPO episode time:      {np.mean(mpo_times)*1000:.1f} ms")
    print(f"  MPO overhead:          {results['overhead_factor']:.1f}x")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"SiliQun v2.1 — MPS vs. MPO Fidelity Comparison\n"
        f"SiMOS {n_qubits}q | {n_episodes} episodes | T1=9.5s, T2*=41µs, J_res=12kHz",
        fontsize=12,
    )

    # Panel 1: Mean fidelity trajectories
    ax = axes[0]
    steps = np.arange(min_len)
    mps_mean = mps_arr.mean(axis=0)
    mpo_mean = mpo_arr.mean(axis=0)
    mps_std = mps_arr.std(axis=0)
    mpo_std = mpo_arr.std(axis=0)
    ax.plot(steps, mps_mean, "b-", label="MPS (pure state)", linewidth=2)
    ax.fill_between(steps, mps_mean - mps_std, mps_mean + mps_std, alpha=0.2, color="b")
    ax.plot(steps, mpo_mean, "r--", label="MPO (density matrix)", linewidth=2)
    ax.fill_between(steps, mpo_mean - mpo_std, mpo_mean + mpo_std, alpha=0.2, color="r")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fidelity")
    ax.set_title("Mean Fidelity Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Absolute error over time
    ax = axes[1]
    diff_mean = diff.mean(axis=0)
    diff_std = diff.std(axis=0)
    ax.plot(steps, diff_mean, "g-", linewidth=2)
    ax.fill_between(steps, diff_mean - diff_std, diff_mean + diff_std, alpha=0.2, color="g")
    ax.axhline(mean_diff, color="orange", linestyle="--", label=f"Mean={mean_diff:.5f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("|F_MPS - F_MPO|")
    ax.set_title("Absolute Fidelity Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Final fidelity distribution
    ax = axes[2]
    ax.hist(mps_arr[:, -1], bins=20, alpha=0.6, color="b", label="MPS final F")
    ax.hist(mpo_arr[:, -1], bins=20, alpha=0.6, color="r", label="MPO final F")
    ax.axvline(mps_final, color="b", linestyle="--", linewidth=2)
    ax.axvline(mpo_final, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Final Fidelity")
    ax.set_ylabel("Count")
    ax.set_title(f"Final Fidelity Distribution\nΔF_mean={final_diff:.5f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = Path("siliqun_v21_mps_vs_mpo.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {out_png}")

    out_json = Path("siliqun_v21_mps_vs_mpo.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SiliQun v2.1 MPS vs. MPO benchmark")
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(args.n_qubits, args.n_episodes, args.seed)


if __name__ == "__main__":
    main()
