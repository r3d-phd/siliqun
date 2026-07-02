#!/usr/bin/env python3
"""
Performance benchmarks for the StateVectorSimulator.

Compares MPS vs. SV backends across different system sizes and
measures gate application time, memory usage, and fidelity accuracy.

Usage:
    python benchmarks/benchmark_sv.py [--max-qubits 16] [--output results.json]
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from siliqun.engine.statevector_simulator import StateVectorSimulator, SVSimConfig
from siliqun.engine.simulator import SiliQunSimulator, SimConfig
from siliqun.physics.devices.profiles import (
    donor_device, sledge_device, sledge_2x2, sledge_3x3,
    sledge_4x4, sledge_5x5,
)


def benchmark_gate_time(sim, n_gates=100, n_qubits=None):
    """Benchmark single and two-qubit gate application time."""
    if n_qubits is None:
        n_qubits = sim.n_qubits

    # Single-qubit gates
    t0 = time.perf_counter()
    for _ in range(n_gates):
        q = np.random.randint(0, n_qubits)
        theta = np.random.uniform(0, 2 * np.pi)
        sim.apply_rx(theta, q)
    t_single = (time.perf_counter() - t0) / n_gates

    # Two-qubit gates (on connected pairs)
    t0 = time.perf_counter()
    for _ in range(n_gates):
        qi = np.random.randint(0, n_qubits - 1)
        qj = qi + 1
        sim.apply_cnot(qi, qj)
    t_two = (time.perf_counter() - t0) / n_gates

    return t_single, t_two


def benchmark_observables(sim, n_qubits=None):
    """Benchmark observable computation time."""
    if n_qubits is None:
        n_qubits = sim.n_qubits

    # Z expectations
    t0 = time.perf_counter()
    for q in range(n_qubits):
        sim.expectation_z(q)
    t_z = (time.perf_counter() - t0) / n_qubits

    # Entanglement entropy
    t0 = time.perf_counter()
    sim.compute_entanglement_entropy(n_qubits // 2)
    t_ent = time.perf_counter() - t0

    return t_z, t_ent


def benchmark_fidelity(sim, n_qubits=None):
    """Benchmark fidelity computation."""
    if n_qubits is None:
        n_qubits = sim.n_qubits

    target = np.zeros(2 ** n_qubits, dtype=np.complex128)
    target[0] = 1.0 / np.sqrt(2)
    target[-1] = 1.0 / np.sqrt(2)

    t0 = time.perf_counter()
    fid = sim.compute_fidelity(target)
    t_fid = time.perf_counter() - t0
    return fid, t_fid


def run_benchmark_suite(max_qubits=16, output_file=None):
    """Run the full benchmark suite."""
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_qubits": max_qubits,
        },
        "benchmarks": [],
    }

    # ---- SV backend benchmarks ----
    sv_configs = [
        ("donor_2q", donor_device(n_qubits=2), 2),
        ("donor_4q", donor_device(n_qubits=4), 4),
        ("donor_8q", donor_device(n_qubits=8), 8),
        ("sledge_2x2", sledge_2x2(), 4),
        ("sledge_3x3", sledge_3x3(), 9),
    ]

    if max_qubits >= 16:
        sv_configs.append(("sledge_4x4", sledge_4x4(), 16))

    if max_qubits >= 25:
        sv_configs.append(("sledge_5x5", sledge_5x5(), 25))

    # Detect GPU availability
    try:
        import cupy as cp
        gpu_available = True
        print(f"CuPy {cp.__version__} detected - GPU mode enabled")
    except ImportError:
        gpu_available = False
        print("CuPy not available - CPU mode only")

    config_sv = SVSimConfig(noise_enabled=False, use_gpu=gpu_available, seed=42)

    for name, device, n_q in sv_configs:
        print(f"\n--- SV Backend: {name} ({n_q} qubits, dim={2**n_q}) ---")

        tracemalloc.start()
        t_init = time.perf_counter()
        sim = StateVectorSimulator(device, config_sv)
        t_init = time.perf_counter() - t_init

        # Apply some gates to create entanglement
        n_gates = min(100, 10 * n_q)
        for i in range(min(n_q - 1, 5)):
            sim.apply_ry(np.pi / 4, i)
            sim.apply_cnot(i, i + 1)

        t_single, t_two = benchmark_gate_time(sim, n_gates=n_gates)
        t_z, t_ent = benchmark_observables(sim)
        fid, t_fid = benchmark_fidelity(sim)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        entry = {
            "name": name,
            "backend": "sv",
            "n_qubits": n_q,
            "dim": 2 ** n_q,
            "init_time_s": t_init,
            "single_gate_time_s": t_single,
            "two_gate_time_s": t_two,
            "z_expectation_time_s": t_z,
            "entropy_time_s": t_ent,
            "fidelity_time_s": t_fid,
            "peak_memory_mb": peak / 1024**2,
            "state_vector_memory_mb": (2**n_q * 16) / 1024**2,
            "is_dfs": sim.is_dfs,
        }
        results["benchmarks"].append(entry)

        print(f"  Init:        {t_init*1000:.2f} ms")
        print(f"  1Q gate:     {t_single*1e6:.1f} us")
        print(f"  2Q gate:     {t_two*1e6:.1f} us")
        print(f"  <Z>:         {t_z*1e6:.1f} us")
        print(f"  S(A:B):      {t_ent*1000:.2f} ms")
        print(f"  Fidelity:    {t_fid*1000:.2f} ms")
        print(f"  Peak memory: {peak/1024**2:.1f} MB")
        print(f"  SV memory:   {(2**n_q * 16)/1024**2:.1f} MB")
        print(f"  Backend:     {'GPU (CuPy)' if sim._use_gpu else 'CPU (NumPy)'}")
        entry["gpu"] = sim._use_gpu

    # If GPU was used for SV benchmarks, also run the main configs on CPU for comparison
    if gpu_available:
        print("\n--- SV Backend (CPU reference for comparison) ---")
        config_sv_cpu = SVSimConfig(noise_enabled=False, use_gpu=False, seed=42)
        for name, device, n_q in sv_configs:
            if n_q > 16:  # Skip 25q CPU in main loop, handled in dedicated section
                continue
            sim_cpu = StateVectorSimulator(device, config_sv_cpu)
            for i in range(min(n_q - 1, 5)):
                sim_cpu.apply_ry(np.pi / 4, i)
                sim_cpu.apply_cnot(i, i + 1)
            t_s, t_t = benchmark_gate_time(sim_cpu, n_gates=min(100, 10*n_q))
            print(f"  {name} CPU: 1Q={t_s*1e6:.1f}us, 2Q={t_t*1e6:.1f}us")

    # ---- MPS backend comparison (small systems only) ----
    mps_configs = [
        ("donor_2q", donor_device(n_qubits=2), 2),
        ("donor_4q", donor_device(n_qubits=4), 4),
        ("donor_8q", donor_device(n_qubits=8), 8),
    ]

    config_mps = SimConfig(noise_enabled=False, max_bond_dim=32, seed=42)

    for name, device, n_q in mps_configs:
        print(f"\n--- MPS Backend: {name} ({n_q} qubits, chi=32) ---")

        tracemalloc.start()
        t_init = time.perf_counter()
        sim = SiliQunSimulator(device, config_mps)
        t_init = time.perf_counter() - t_init

        # Apply same gates
        n_gates = min(100, 10 * n_q)
        for i in range(min(n_q - 1, 5)):
            sim.apply_ry(np.pi / 4, i)
            sim.apply_cnot(i, i + 1)

        t_single, t_two = benchmark_gate_time(sim, n_gates=n_gates)
        t_z, t_ent = benchmark_observables(sim)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        entry = {
            "name": name,
            "backend": "mps",
            "n_qubits": n_q,
            "bond_dim": 32,
            "init_time_s": t_init,
            "single_gate_time_s": t_single,
            "two_gate_time_s": t_two,
            "z_expectation_time_s": t_z,
            "entropy_time_s": t_ent,
            "peak_memory_mb": peak / 1024**2,
        }
        results["benchmarks"].append(entry)

        print(f"  Init:        {t_init*1000:.2f} ms")
        print(f"  1Q gate:     {t_single*1e6:.1f} us")
        print(f"  2Q gate:     {t_two*1e6:.1f} us")
        print(f"  <Z>:         {t_z*1e6:.1f} us")
        print(f"  S(A:B):      {t_ent*1000:.2f} ms")
        print(f"  Peak memory: {peak/1024**2:.1f} MB")

    # ---- CPU vs GPU comparison (if GPU available) ----
    if gpu_available and max_qubits >= 25:
        print("\n\n=== CPU vs GPU Comparison (25 qubits) ===")
        config_cpu = SVSimConfig(noise_enabled=False, use_gpu=False, seed=42)
        config_gpu = SVSimConfig(noise_enabled=False, use_gpu=True, seed=42)
        dev_5x5 = sledge_5x5()

        # CPU benchmark
        sim_cpu = StateVectorSimulator(dev_5x5, config_cpu)
        for i in range(4):
            sim_cpu.apply_ry(np.pi / 4, i)
            sim_cpu.apply_cnot(i, i + 1)

        t0 = time.perf_counter()
        for _ in range(5):
            sim_cpu.apply_rx(0.5, 0)
        t_cpu_1q = (time.perf_counter() - t0) / 5

        t0 = time.perf_counter()
        for _ in range(5):
            sim_cpu.apply_cnot(0, 1)
        t_cpu_2q = (time.perf_counter() - t0) / 5

        t0 = time.perf_counter()
        sim_cpu.expectation_z(0)
        t_cpu_z = time.perf_counter() - t0

        # GPU benchmark
        sim_gpu = StateVectorSimulator(dev_5x5, config_gpu)
        for i in range(4):
            sim_gpu.apply_ry(np.pi / 4, i)
            sim_gpu.apply_cnot(i, i + 1)

        # Warmup GPU
        sim_gpu.apply_rx(0.5, 0)
        import cupy
        cupy.cuda.Device().synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            sim_gpu.apply_rx(0.5, 0)
        cupy.cuda.Device().synchronize()
        t_gpu_1q = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        for _ in range(10):
            sim_gpu.apply_cnot(0, 1)
        cupy.cuda.Device().synchronize()
        t_gpu_2q = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        sim_gpu.expectation_z(0)
        cupy.cuda.Device().synchronize()
        t_gpu_z = time.perf_counter() - t0

        print(f"  25q 1Q gate: CPU={t_cpu_1q*1e3:.2f} ms, GPU={t_gpu_1q*1e3:.2f} ms, Speedup={t_cpu_1q/t_gpu_1q:.1f}x")
        print(f"  25q 2Q gate: CPU={t_cpu_2q*1e3:.2f} ms, GPU={t_gpu_2q*1e3:.2f} ms, Speedup={t_cpu_2q/t_gpu_2q:.1f}x")
        print(f"  25q <Z>:     CPU={t_cpu_z*1e3:.2f} ms, GPU={t_gpu_z*1e3:.2f} ms, Speedup={t_cpu_z/t_gpu_z:.1f}x")

        results["cpu_vs_gpu_25q"] = {
            "cpu_1q_ms": t_cpu_1q * 1e3,
            "gpu_1q_ms": t_gpu_1q * 1e3,
            "speedup_1q": t_cpu_1q / t_gpu_1q,
            "cpu_2q_ms": t_cpu_2q * 1e3,
            "gpu_2q_ms": t_gpu_2q * 1e3,
            "speedup_2q": t_cpu_2q / t_gpu_2q,
            "cpu_z_ms": t_cpu_z * 1e3,
            "gpu_z_ms": t_gpu_z * 1e3,
            "speedup_z": t_cpu_z / t_gpu_z,
        }

    # ---- DFS Leakage Analysis ----
    print("\n\n=== DFS Leakage Analysis ===")
    from siliqun.engine.statevector_simulator import DFSLogicalProjector

    proj = DFSLogicalProjector(2, [(0, 1)])
    angles = np.linspace(0.01, np.pi, 50)
    leakage_data = []
    for theta in angles:
        leak = proj.compute_leakage_rate(theta, 0, 1, 2, 0)
        leakage_data.append({"theta": float(theta), "leakage": float(leak)})
        if theta in [0.01, 0.1, 0.5, 1.0, np.pi]:
            print(f"  theta={theta:.3f}: leakage={leak:.6e}")

    results["leakage_analysis"] = leakage_data

    # ---- Save results ----
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiliQun SV Benchmark")
    parser.add_argument("--max-qubits", type=int, default=16)
    parser.add_argument("--output", type=str, default="benchmarks/sv_benchmark_results.json")
    args = parser.parse_args()

    os.makedirs("benchmarks", exist_ok=True)
    run_benchmark_suite(args.max_qubits, args.output)
