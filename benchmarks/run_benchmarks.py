"""
SiliQun Benchmark Suite for SoftwareX Paper.

Benchmarks:
1. Simulation speed (steps/second) vs. qubit count
2. Gate fidelity accuracy vs. analytical solutions
3. Scaling: wall-clock time vs. qubit count
4. Noise model validation: T1 decay, T2 dephasing
5. DRL environment throughput (steps/second)
6. Bond dimension impact on accuracy and speed
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siliqun.backend import set_backend, active_backend
from siliqun.tensor.mps import MPS
from siliqun.tensor.mpo import MPO
from siliqun.physics.gates import rx, ry, rz, cnot, cz, hadamard, pauli_x, pauli_y, pauli_z
from siliqun.physics.hamiltonian import DeviceParams, build_hamiltonian_mpo
from siliqun.physics.noise.channels import NoiseParams, default_noise_params, ChargeNoiseGenerator
from siliqun.physics.devices.profiles import get_device_profile
from siliqun.engine.simulator import SiliQunSimulator, SimConfig
from siliqun.engine.gym_env import SiliQunEnv

set_backend("numpy")

RESULTS = {}


def benchmark_timer(func, n_repeats=5):
    """Run a function multiple times and return mean and std of execution time."""
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times), np.std(times), result


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 1: Gate Fidelity Accuracy
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("BENCHMARK 1: Gate Fidelity Accuracy vs Analytical Solutions")
print("=" * 60)

gate_accuracy = {}

# Test 1a: Rx(pi) on |0> should give |1>
dev = get_device_profile("donor", 2)
sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False, max_bond_dim=32))
sim.reset()
sim.apply_rx(np.pi, 0)
z0 = sim.expectation_z(0)
expected_z0 = -1.0  # |1> has <Z> = -1
error_rx = abs(z0 - expected_z0)
gate_accuracy["Rx(pi)_Z_expectation"] = {"measured": z0, "expected": expected_z0, "error": error_rx}
print(f"  Rx(pi)|0>: <Z> = {z0:.8f}, expected = {expected_z0:.8f}, error = {error_rx:.2e}")

# Test 1b: Ry(pi/2) on |0> should give (|0> + |1>)/sqrt(2), <Z> = 0
sim.reset()
sim.apply_ry(np.pi / 2, 0)
z0 = sim.expectation_z(0)
expected_z0 = 0.0
error_ry = abs(z0 - expected_z0)
gate_accuracy["Ry(pi/2)_Z_expectation"] = {"measured": z0, "expected": expected_z0, "error": error_ry}
print(f"  Ry(pi/2)|0>: <Z> = {z0:.8f}, expected = {expected_z0:.8f}, error = {error_ry:.2e}")

# Test 1c: Rz(pi) on |+> should give |->
sim.reset()
sim.apply_ry(np.pi / 2, 0)  # Create |+>
sim.apply_rz(np.pi, 0)       # Should give |->
# <X> of |-> is -1, but we measure <Z> after another Ry(-pi/2)
sim.apply_ry(-np.pi / 2, 0)  # Rotate back
z0 = sim.expectation_z(0)
expected_z0 = -1.0  # |1> state
error_rz = abs(z0 - expected_z0)
gate_accuracy["Rz(pi)_roundtrip"] = {"measured": z0, "expected": expected_z0, "error": error_rz}
print(f"  Rz(pi) roundtrip: <Z> = {z0:.8f}, expected = {expected_z0:.8f}, error = {error_rz:.2e}")

# Test 1d: Bell state creation: H|0> then CNOT
sim.reset()
sim.apply_ry(np.pi / 2, 0)   # Approximate Hadamard
sim.apply_rz(np.pi, 0)        # Complete Hadamard: Rz(pi)Ry(pi/2)
sim.apply_cnot(0, 1)
zz = sim.expectation_zz(0, 1)
expected_zz = 1.0  # Bell state: ZZ correlator = 1
error_bell = abs(zz - expected_zz)
gate_accuracy["Bell_ZZ_correlator"] = {"measured": zz, "expected": expected_zz, "error": error_bell}
print(f"  Bell state <ZZ> = {zz:.8f}, expected = {expected_zz:.8f}, error = {error_bell:.2e}")

# Test 1e: CNOT preserves |00>
sim.reset()
sim.apply_cnot(0, 1)
target_00 = MPS.computational_basis(2, state=0)
fid = sim.compute_fidelity(target_00)
error_cnot = abs(fid - 1.0)
gate_accuracy["CNOT_identity_fidelity"] = {"measured": fid, "expected": 1.0, "error": error_cnot}
print(f"  CNOT|00> fidelity = {fid:.8f}, expected = 1.0, error = {error_cnot:.2e}")

# Test 1f: Entanglement entropy of Bell state
sim.reset()
sim.apply_ry(np.pi / 2, 0)
sim.apply_rz(np.pi, 0)
sim.apply_cnot(0, 1)
S = sim.compute_entanglement_entropy(1)
expected_S = 1.0  # Bell state has 1 ebit
error_entropy = abs(S - expected_S)
gate_accuracy["Bell_entropy"] = {"measured": S, "expected": expected_S, "error": error_entropy}
print(f"  Bell entropy = {S:.8f}, expected = {expected_S:.8f}, error = {error_entropy:.2e}")

RESULTS["gate_accuracy"] = gate_accuracy


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 2: Simulation Speed vs Qubit Count
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 2: Simulation Speed (steps/sec) vs Qubit Count")
print("=" * 60)

speed_results = {}
qubit_counts = [2, 4, 6, 8, 10, 12]
n_steps = 50

for nq in qubit_counts:
    dev = get_device_profile("donor", nq)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False, max_bond_dim=32))

    def run_simulation():
        sim.reset()
        for step in range(n_steps):
            q = step % nq
            sim.apply_ry(np.pi / 4, q)
            if nq > 1 and q < nq - 1:
                sim.apply_cnot(q, q + 1)
        return sim.state.bond_dims

    mean_t, std_t, bond_dims = benchmark_timer(run_simulation, n_repeats=3)
    steps_per_sec = n_steps / mean_t
    speed_results[nq] = {
        "mean_time_s": mean_t,
        "std_time_s": std_t,
        "steps_per_sec": steps_per_sec,
        "max_bond_dim": max(bond_dims) if bond_dims else 1,
    }
    print(f"  {nq:2d} qubits: {steps_per_sec:8.1f} steps/s (mean={mean_t:.4f}s ± {std_t:.4f}s, max_bond={max(bond_dims) if bond_dims else 1})")

RESULTS["simulation_speed"] = speed_results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 3: Noisy vs Noiseless Comparison
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 3: Noisy vs Noiseless Simulation Speed")
print("=" * 60)

noise_comparison = {}
for nq in [2, 4, 6, 8]:
    for noise_on in [False, True]:
        dev = get_device_profile("donor", nq)
        sim = SiliQunSimulator(dev, SimConfig(noise_enabled=noise_on, max_bond_dim=32))

        def run_noisy():
            sim.reset()
            for step in range(30):
                q = step % nq
                sim.apply_ry(np.pi / 4, q)
            return None

        mean_t, std_t, _ = benchmark_timer(run_noisy, n_repeats=3)
        label = f"{nq}q_{'noisy' if noise_on else 'noiseless'}"
        noise_comparison[label] = {
            "n_qubits": nq,
            "noise_enabled": noise_on,
            "mean_time_s": mean_t,
            "steps_per_sec": 30 / mean_t,
        }
        print(f"  {label}: {30/mean_t:.1f} steps/s ({mean_t:.4f}s)")

RESULTS["noise_comparison"] = noise_comparison


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 4: T1 Decay Validation
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 4: T1 Relaxation Decay Validation")
print("=" * 60)

t1_results = {}
# Prepare |1> state and let it decay
dev = get_device_profile("donor", 1)
t1_time = dev.noise_params.t1_times[0]  # 30 s for donor
n_samples = 30
time_points = np.linspace(0, 0.0005, 10)  # 0 to 0.5 ms

for t_wait in time_points:
    z_values = []
    for _ in range(n_samples):
        sim = SiliQunSimulator(dev, SimConfig(noise_enabled=True, max_bond_dim=4, dt=1e-7))
        sim.reset()
        sim.apply_ry(np.pi, 0)  # Prepare |1>
        if t_wait > 0:
            sim.apply_idle_noise(t_wait)
        z = sim.expectation_z(0)
        z_values.append(z)
    mean_z = np.mean(z_values)
    # Analytical: <Z>(t) = -exp(-t/T1) for |1> initial state
    expected_z = -np.exp(-t_wait / t1_time)
    t1_results[f"t={t_wait:.6f}"] = {
        "time_s": t_wait,
        "measured_z": mean_z,
        "expected_z": expected_z,
        "error": abs(mean_z - expected_z),
    }

# Print summary
print(f"  T1 = {t1_time} s, testing decay over 0-1 ms ({n_samples} samples/point)")
for key, val in list(t1_results.items())[:5]:
    print(f"    t={val['time_s']:.6f}s: <Z>={val['measured_z']:.4f}, expected={val['expected_z']:.4f}, err={val['error']:.4f}")
print(f"    ... ({len(t1_results)} total points)")

RESULTS["t1_decay"] = t1_results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 5: DRL Environment Throughput
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 5: DRL Environment Throughput (steps/sec)")
print("=" * 60)

env_throughput = {}
for nq in [2, 4, 6, 8]:
    for noise_on in [False, True]:
        env = SiliQunEnv(
            device="donor", n_qubits=nq,
            target_state="ghz", max_steps=200,
            config=SimConfig(noise_enabled=noise_on, max_bond_dim=32),
            reward_type="dense",
        )

        def run_env():
            obs, _ = env.reset(seed=42)
            for _ in range(100):
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
            return info

        mean_t, std_t, last_info = benchmark_timer(run_env, n_repeats=3)
        steps_per_sec = 100 / mean_t
        label = f"{nq}q_{'noisy' if noise_on else 'clean'}"
        env_throughput[label] = {
            "n_qubits": nq,
            "noise_enabled": noise_on,
            "steps_per_sec": steps_per_sec,
            "mean_time_s": mean_t,
        }
        print(f"  {label}: {steps_per_sec:8.1f} env steps/s ({mean_t:.4f}s for 100 steps)")

RESULTS["env_throughput"] = env_throughput


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 6: Bond Dimension Impact
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 6: Bond Dimension Impact on Speed and Accuracy")
print("=" * 60)

bond_dim_results = {}
nq = 6
bond_dims_to_test = [4, 8, 16, 32, 64]

for bd in bond_dims_to_test:
    dev = get_device_profile("donor", nq)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False, max_bond_dim=bd))

    def run_bd():
        sim.reset()
        # Create entangled state
        for i in range(nq - 1):
            sim.apply_ry(np.pi / 2, i)
            sim.apply_cnot(i, i + 1)
        # Compute GHZ fidelity
        target = MPS.ghz_state(nq)
        fid = sim.compute_fidelity(target)
        return fid, sim.state.bond_dims

    mean_t, std_t, (fid, actual_bonds) = benchmark_timer(run_bd, n_repeats=3)
    bond_dim_results[bd] = {
        "max_bond_dim_config": bd,
        "actual_max_bond": max(actual_bonds) if actual_bonds else 1,
        "fidelity": fid,
        "time_s": mean_t,
        "steps_per_sec": (nq - 1) * 2 / mean_t,
    }
    print(f"  max_bond={bd:3d}: fidelity={fid:.6f}, time={mean_t:.4f}s, actual_max_bond={max(actual_bonds) if actual_bonds else 1}")

RESULTS["bond_dimension"] = bond_dim_results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 7: Device Profile Comparison
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 7: Device Profile Comparison (4 qubits)")
print("=" * 60)

device_comparison = {}
for dev_type in ["donor", "simos", "gaa"]:
    dev = get_device_profile(dev_type, 4)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=True, max_bond_dim=32))

    def run_device():
        sim.reset()
        for i in range(3):
            sim.apply_ry(np.pi / 2, i)
            sim.apply_cnot(i, i + 1)
        target = MPS.ghz_state(4)
        fid = sim.compute_fidelity(target)
        return fid

    mean_t, std_t, fid = benchmark_timer(run_device, n_repeats=3)
    device_comparison[dev_type] = {
        "device": dev_type,
        "n_qubits": 4,
        "fidelity_noisy": fid,
        "time_s": mean_t,
        "t1": dev.noise_params.t1_times[0],
        "t2_star": dev.noise_params.t2_star_times[0],
        "gate_time_single": dev.gate_times.get("single", 0),
        "gate_time_two": dev.gate_times.get("two", 0),
    }
    print(f"  {dev_type:6s}: fidelity={fid:.6f}, T1={dev.noise_params.t1_times[0]:.1f}s, T2*={dev.noise_params.t2_star_times[0]:.2e}s, time={mean_t:.4f}s")

RESULTS["device_comparison"] = device_comparison


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 8: Reward Function Comparison
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BENCHMARK 8: Reward Function Comparison (2Q Bell)")
print("=" * 60)

reward_comparison = {}
for reward_type in ["dense", "sparse", "shaped"]:
    env = SiliQunEnv(
        device="donor", n_qubits=2,
        target_state="bell", max_steps=100,
        config=SimConfig(noise_enabled=False, max_bond_dim=16),
        reward_type=reward_type,
    )

    total_rewards = []
    best_fids = []
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        ep_reward = 0
        best_fid = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            best_fid = max(best_fid, info["fidelity"])
            if term or trunc:
                break
        total_rewards.append(ep_reward)
        best_fids.append(best_fid)

    reward_comparison[reward_type] = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_best_fidelity": np.mean(best_fids),
        "std_best_fidelity": np.std(best_fids),
    }
    print(f"  {reward_type:7s}: reward={np.mean(total_rewards):.3f}±{np.std(total_rewards):.3f}, best_fid={np.mean(best_fids):.4f}±{np.std(best_fids):.4f}")

RESULTS["reward_comparison"] = reward_comparison


# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

results_clean = convert_numpy(RESULTS)
output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
with open(output_path, "w") as f:
    json.dump(results_clean, f, indent=2)

print(f"Results saved to {output_path}")
print("BENCHMARKS COMPLETE")
