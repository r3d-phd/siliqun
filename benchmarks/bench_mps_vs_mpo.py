"""
SiliQun Benchmark: MPS (Pure State) vs MPO (Density Matrix) Simulator.

Compares:
    1. Accuracy: Observable agreement in noiseless limit
    2. Noise fidelity: Exact (MPO) vs stochastic (MPS) noise
    3. Performance: Wall-clock time for gate application
    4. Scalability: Bond dimension growth with qubits and circuit depth
    5. Gymnasium episode throughput

Results are saved as JSON and plotted as publication-quality figures.
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siliqun.engine.simulator import SiliQunSimulator, SimConfig
from siliqun.engine.mpo_simulator import MPODensityMatrixSimulator, MPOSimConfig
from siliqun.engine.gym_env import SiliQunEnv, make_siliqun_env
from siliqun.physics.devices.profiles import get_device_profile
from siliqun.physics.gates import rx, ry, rz, hadamard, cnot
from siliqun.tensor.mps import MPS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def benchmark_accuracy(n_qubits_list=[2, 3, 4], n_circuits=5, depth=10):
    """Compare MPS and MPO observable accuracy in noiseless limit."""
    print("=" * 60)
    print("BENCHMARK 1: Accuracy (Noiseless MPS vs MPO)")
    print("=" * 60)

    results = []
    for n_q in n_qubits_list:
        z_errors = []
        zz_errors = []
        fid_errors = []

        for seed in range(n_circuits):
            dev = get_device_profile("donor", n_q)
            mps_sim = SiliQunSimulator(dev, SimConfig(
                noise_enabled=False, max_bond_dim=64))
            mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
                noise_enabled=False, max_bond_dim=64))

            np.random.seed(seed)
            for _ in range(depth):
                gate_type = np.random.choice(["rx", "ry", "rz", "cnot"])
                if gate_type in ["rx", "ry", "rz"]:
                    theta = np.random.uniform(-np.pi, np.pi)
                    qubit = np.random.randint(n_q)
                    gate = {"rx": rx, "ry": ry, "rz": rz}[gate_type](theta)
                    mps_sim.apply_single_gate(gate, qubit)
                    mpo_sim._apply_single_gate_to_mpo(gate, qubit)
                else:
                    q = np.random.randint(n_q - 1)
                    mps_sim.apply_two_qubit_gate(cnot(), q, q + 1)
                    mpo_sim._apply_two_qubit_gate_to_mpo(cnot(), q, q + 1)

            # Compare Z expectations
            try:
                for q in range(n_q):
                    z_mps = mps_sim.expectation_z(q)
                    z_mpo = mpo_sim.expectation_z(q)
                    z_errors.append(abs(z_mps - z_mpo))

                # Compare ZZ correlators
                for q in range(n_q - 1):
                    zz_mps = mps_sim.expectation_zz(q, q + 1)
                    zz_mpo = mpo_sim.expectation_zz(q, q + 1)
                    zz_errors.append(abs(zz_mps - zz_mpo))

                # Compare fidelity
                target = MPS.computational_basis(n_q, state=0)
                fid_mps = mps_sim.compute_fidelity(target)
                fid_mpo = mpo_sim.compute_fidelity(target)
                fid_errors.append(abs(fid_mps - fid_mpo))
            except Exception as e:
                print(f"    Warning: circuit {seed} failed for {n_q}Q: {e}")
                continue

        result = {
            "n_qubits": n_q,
            "z_error_mean": float(np.mean(z_errors)),
            "z_error_max": float(np.max(z_errors)),
            "zz_error_mean": float(np.mean(zz_errors)),
            "zz_error_max": float(np.max(zz_errors)),
            "fid_error_mean": float(np.mean(fid_errors)),
            "fid_error_max": float(np.max(fid_errors)),
        }
        results.append(result)
        print(f"  {n_q}Q: <Z> err={result['z_error_mean']:.2e}, "
              f"<ZZ> err={result['zz_error_mean']:.2e}, "
              f"Fid err={result['fid_error_mean']:.2e}")

    return results


def benchmark_noise_comparison(n_qubits=2, n_trials=20, depth=15):
    """Compare exact MPO noise vs stochastic MPS noise."""
    print()
    print("=" * 60)
    print("BENCHMARK 2: Noise Handling (Exact MPO vs Stochastic MPS)")
    print("=" * 60)

    dev = get_device_profile("donor", n_qubits)

    # MPO: exact noise (single run)
    mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
        noise_enabled=True, max_bond_dim=32))

    np.random.seed(42)
    circuit = []
    for _ in range(depth):
        gate_type = np.random.choice(["ry", "cnot"])
        if gate_type == "ry":
            theta = np.random.uniform(-np.pi, np.pi)
            qubit = np.random.randint(n_qubits)
            circuit.append(("ry", theta, qubit))
        else:
            q = np.random.randint(n_qubits - 1)
            circuit.append(("cnot", None, (q, q + 1)))

    # Run MPO (exact)
    for gate_type, param, qubits in circuit:
        if gate_type == "ry":
            mpo_sim.apply_ry(param, qubits)
        else:
            mpo_sim.apply_cnot(qubits[0], qubits[1])

    z_mpo = [mpo_sim.expectation_z(q) for q in range(n_qubits)]
    purity_mpo = mpo_sim.compute_purity()
    trace_mpo = mpo_sim.trace()

    # Run MPS (stochastic, multiple trials)
    z_mps_trials = []
    for trial in range(n_trials):
        mps_sim = SiliQunSimulator(dev, SimConfig(
            noise_enabled=True, max_bond_dim=32, seed=trial))
        for gate_type, param, qubits in circuit:
            if gate_type == "ry":
                mps_sim.apply_ry(param, qubits)
            else:
                mps_sim.apply_cnot(qubits[0], qubits[1])
        z_mps_trials.append([mps_sim.expectation_z(q) for q in range(n_qubits)])

    z_mps_mean = np.mean(z_mps_trials, axis=0)
    z_mps_std = np.std(z_mps_trials, axis=0)

    results = {
        "n_qubits": n_qubits,
        "depth": depth,
        "n_trials": n_trials,
        "mpo_z": [float(z) for z in z_mpo],
        "mpo_purity": float(purity_mpo),
        "mpo_trace": float(np.real(trace_mpo)),
        "mps_z_mean": [float(z) for z in z_mps_mean],
        "mps_z_std": [float(z) for z in z_mps_std],
        "z_agreement": [float(abs(z_mpo[q] - z_mps_mean[q])) for q in range(n_qubits)],
    }

    print(f"  MPO (exact):     <Z> = {z_mpo}")
    print(f"  MPS (mean±std):  <Z> = {z_mps_mean} ± {z_mps_std}")
    print(f"  Agreement:       {results['z_agreement']}")
    print(f"  MPO purity:      {purity_mpo:.6f}")
    print(f"  MPO trace:       {np.real(trace_mpo):.6f}")

    return results


def benchmark_performance(n_qubits_list=[2, 3, 4, 5, 6], depth=20):
    """Compare wall-clock time for MPS vs MPO gate application."""
    print()
    print("=" * 60)
    print("BENCHMARK 3: Performance (Wall-Clock Time)")
    print("=" * 60)

    results = []
    for n_q in n_qubits_list:
        dev = get_device_profile("donor", n_q)

        # MPS timing
        mps_sim = SiliQunSimulator(dev, SimConfig(
            noise_enabled=True, max_bond_dim=32))
        t0 = time.perf_counter()
        np.random.seed(42)
        for _ in range(depth):
            gate_type = np.random.choice(["ry", "cnot"])
            if gate_type == "ry":
                theta = np.random.uniform(-np.pi, np.pi)
                qubit = np.random.randint(n_q)
                mps_sim.apply_ry(theta, qubit)
            else:
                q = np.random.randint(n_q - 1)
                mps_sim.apply_cnot(q, q + 1)
        mps_time = time.perf_counter() - t0

        # MPO timing
        mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
            noise_enabled=True, max_bond_dim=32))
        t0 = time.perf_counter()
        np.random.seed(42)
        for _ in range(depth):
            gate_type = np.random.choice(["ry", "cnot"])
            if gate_type == "ry":
                theta = np.random.uniform(-np.pi, np.pi)
                qubit = np.random.randint(n_q)
                mpo_sim.apply_ry(theta, qubit)
            else:
                q = np.random.randint(n_q - 1)
                mpo_sim.apply_cnot(q, q + 1)
        mpo_time = time.perf_counter() - t0

        result = {
            "n_qubits": n_q,
            "depth": depth,
            "mps_time_s": float(mps_time),
            "mpo_time_s": float(mpo_time),
            "slowdown_factor": float(mpo_time / mps_time) if mps_time > 0 else float("inf"),
        }
        results.append(result)
        print(f"  {n_q}Q: MPS={mps_time:.3f}s, MPO={mpo_time:.3f}s, "
              f"slowdown={result['slowdown_factor']:.1f}x")

    return results


def benchmark_bond_dim_growth(n_qubits=4, max_depth=30):
    """Track bond dimension growth during circuit execution."""
    print()
    print("=" * 60)
    print("BENCHMARK 4: Bond Dimension Growth")
    print("=" * 60)

    dev = get_device_profile("donor", n_qubits)
    mps_sim = SiliQunSimulator(dev, SimConfig(
        noise_enabled=True, max_bond_dim=64))
    mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
        noise_enabled=True, max_bond_dim=64, compress_every=0))

    mps_bonds = []
    mpo_bonds = []

    np.random.seed(42)
    for step in range(max_depth):
        gate_type = np.random.choice(["ry", "cnot"])
        if gate_type == "ry":
            theta = np.random.uniform(-np.pi, np.pi)
            qubit = np.random.randint(n_qubits)
            mps_sim.apply_ry(theta, qubit)
            mpo_sim.apply_ry(theta, qubit)
        else:
            q = np.random.randint(n_qubits - 1)
            mps_sim.apply_cnot(q, q + 1)
            mpo_sim.apply_cnot(q, q + 1)

        mps_max_bd = max(mps_sim.state.bond_dims) if mps_sim.state.bond_dims else 1
        mpo_max_bd = mpo_sim.state.max_bond_dim
        mps_bonds.append(int(mps_max_bd))
        mpo_bonds.append(int(mpo_max_bd))

    results = {
        "n_qubits": n_qubits,
        "mps_bond_dims": mps_bonds,
        "mpo_bond_dims": mpo_bonds,
        "mps_final_max": int(mps_bonds[-1]),
        "mpo_final_max": int(mpo_bonds[-1]),
    }

    print(f"  MPS final max bond dim: {results['mps_final_max']}")
    print(f"  MPO final max bond dim: {results['mpo_final_max']}")

    return results


def benchmark_gym_throughput(n_qubits=2, n_episodes=3, max_steps=50):
    """Compare Gymnasium episode throughput for MPS vs MPO."""
    print()
    print("=" * 60)
    print("BENCHMARK 5: Gymnasium Episode Throughput")
    print("=" * 60)

    results = {}
    for mode in ["mps", "mpo"]:
        env = make_siliqun_env(
            n_qubits=n_qubits, device="donor", target="bell",
            sim_mode=mode, noise=True, max_bond_dim=16,
            max_steps=max_steps, seed=42,
        )

        t0 = time.perf_counter()
        total_steps = 0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            for _ in range(max_steps):
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                total_steps += 1
                if term or trunc:
                    break
        elapsed = time.perf_counter() - t0

        results[mode] = {
            "total_steps": total_steps,
            "total_time_s": float(elapsed),
            "steps_per_sec": float(total_steps / elapsed) if elapsed > 0 else 0,
        }
        print(f"  {mode.upper()}: {total_steps} steps in {elapsed:.2f}s "
              f"({results[mode]['steps_per_sec']:.1f} steps/s)")

    results["slowdown_factor"] = (
        results["mps"]["steps_per_sec"] / results["mpo"]["steps_per_sec"]
        if results["mpo"]["steps_per_sec"] > 0 else float("inf")
    )
    print(f"  Slowdown: {results['slowdown_factor']:.1f}x")

    return results


def benchmark_purity_decay(n_qubits=2, depth=30):
    """Track purity decay under noise (MPO only)."""
    print()
    print("=" * 60)
    print("BENCHMARK 6: Purity Decay Under Noise (MPO)")
    print("=" * 60)

    dev = get_device_profile("donor", n_qubits)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
        noise_enabled=True, max_bond_dim=32))

    purities = [float(sim.compute_purity())]
    traces = [float(np.real(sim.trace()))]

    np.random.seed(42)
    for step in range(depth):
        gate_type = np.random.choice(["ry", "cnot"])
        if gate_type == "ry":
            theta = np.random.uniform(-np.pi, np.pi)
            qubit = np.random.randint(n_qubits)
            sim.apply_ry(theta, qubit)
        else:
            q = np.random.randint(n_qubits - 1)
            sim.apply_cnot(q, q + 1)

        purities.append(float(sim.compute_purity()))
        traces.append(float(np.real(sim.trace())))

    results = {
        "n_qubits": n_qubits,
        "purities": purities,
        "traces": traces,
        "initial_purity": purities[0],
        "final_purity": purities[-1],
        "min_purity": min(purities),
        "trace_deviation_max": max(abs(t - 1.0) for t in traces),
    }

    print(f"  Initial purity: {results['initial_purity']:.6f}")
    print(f"  Final purity:   {results['final_purity']:.6f}")
    print(f"  Min purity:     {results['min_purity']:.6f}")
    print(f"  Max trace dev:  {results['trace_deviation_max']:.2e}")

    return results


# ═══════════════════════════════════════════════════════════════
# RUN ALL BENCHMARKS
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_results = {}

    all_results["accuracy"] = benchmark_accuracy()
    all_results["noise_comparison"] = benchmark_noise_comparison()
    all_results["performance"] = benchmark_performance()
    all_results["bond_dim_growth"] = benchmark_bond_dim_growth()
    all_results["gym_throughput"] = benchmark_gym_throughput()
    all_results["purity_decay"] = benchmark_purity_decay()

    # Save results
    output_path = os.path.join(RESULTS_DIR, "mps_vs_mpo_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)
