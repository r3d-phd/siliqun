# -*- coding: utf-8 -*-
"""
Comparison Benchmark: SiliQun vs Qiskit Aer vs QuTiP
Measures gate application throughput for fair comparison.

All tools apply the same sequence: H gates on each qubit in round-robin.
"""
import json
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/siliqun"))

H_MATRIX = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

# ============================================================
# SiliQun Benchmark (MPS-based)
# ============================================================
def bench_siliqun(n_qubits, n_gates):
    from siliqun.engine.simulator import SiliQunSimulator, SimConfig
    from siliqun.physics.devices.profiles import simos_device
    
    chi = min(64, 2 ** min(n_qubits, 6))
    dp = simos_device(n_qubits=n_qubits)
    config = SimConfig(noise_enabled=False, max_bond_dim=chi)
    
    # Warmup
    sim_w = SiliQunSimulator(device=dp, config=config)
    for _ in range(200):
        sim_w.apply_single_gate(H_MATRIX, 0)
    
    # Timed run
    sim = SiliQunSimulator(device=dp, config=config)
    start = time.perf_counter()
    for i in range(n_gates):
        sim.apply_single_gate(H_MATRIX, i % n_qubits)
    elapsed = time.perf_counter() - start
    
    return n_gates / elapsed, elapsed

# ============================================================
# Qiskit Aer Benchmark (statevector)
# ============================================================
def bench_qiskit_aer(n_qubits, n_gates):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    
    # Build circuit
    qc = QuantumCircuit(n_qubits)
    for i in range(n_gates):
        qc.h(i % n_qubits)
    qc.save_statevector()
    
    sim = AerSimulator(method='statevector')
    
    # Warmup
    warmup_qc = QuantumCircuit(n_qubits)
    for i in range(200):
        warmup_qc.h(i % n_qubits)
    warmup_qc.save_statevector()
    sim.run(warmup_qc, shots=1).result()
    
    # Timed run
    start = time.perf_counter()
    result = sim.run(qc, shots=1).result()
    elapsed = time.perf_counter() - start
    
    return n_gates / elapsed, elapsed

# ============================================================
# QuTiP Benchmark (full state vector with Qobj)
# ============================================================
def bench_qutip(n_qubits, n_gates):
    import qutip
    
    # Create initial state |00...0>
    psi = qutip.tensor([qutip.basis(2, 0)] * n_qubits)
    H_gate = qutip.Qobj([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Pre-build gate operators for each qubit
    gates = []
    for q in range(n_qubits):
        ops = [qutip.qeye(2)] * n_qubits
        ops[q] = H_gate
        gates.append(qutip.tensor(ops))
    
    # Warmup
    state = psi
    for i in range(200):
        state = gates[i % n_qubits] * state
    
    # Reset and timed run
    state = psi
    start = time.perf_counter()
    for i in range(n_gates):
        state = gates[i % n_qubits] * state
    elapsed = time.perf_counter() - start
    
    return n_gates / elapsed, elapsed

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 75)
    print("Comparison Benchmark: SiliQun (MPS) vs Qiskit Aer (SV) vs QuTiP (SV)")
    print("=" * 75)
    
    # Test configurations: (n_qubits, n_gates)
    configs = [
        (2, 5000),
        (4, 5000),
        (6, 5000),
        (8, 3000),
        (10, 2000),
        (12, 1000),
        (14, 500),
        (16, 500),
    ]
    
    all_results = []
    
    for n_qubits, n_gates in configs:
        print(f"\n--- {n_qubits} qubits, {n_gates} gates ---")
        row = {"n_qubits": n_qubits, "n_gates": n_gates}
        
        # SiliQun
        try:
            tp, elapsed = bench_siliqun(n_qubits, n_gates)
            row["siliqun_throughput"] = round(tp, 1)
            row["siliqun_time_s"] = round(elapsed, 4)
            print(f"  SiliQun:    {tp:>10.0f} gates/s  ({elapsed:.4f}s)")
        except Exception as e:
            row["siliqun_throughput"] = None
            row["siliqun_error"] = str(e)
            print(f"  SiliQun:    ERROR - {e}")
        
        # Qiskit Aer
        try:
            tp, elapsed = bench_qiskit_aer(n_qubits, n_gates)
            row["qiskit_throughput"] = round(tp, 1)
            row["qiskit_time_s"] = round(elapsed, 4)
            print(f"  Qiskit Aer: {tp:>10.0f} gates/s  ({elapsed:.4f}s)")
        except Exception as e:
            row["qiskit_throughput"] = None
            row["qiskit_error"] = str(e)
            print(f"  Qiskit Aer: ERROR - {e}")
        
        # QuTiP (skip for large systems - memory intensive with full state vector)
        if n_qubits <= 14:
            try:
                tp, elapsed = bench_qutip(n_qubits, n_gates)
                row["qutip_throughput"] = round(tp, 1)
                row["qutip_time_s"] = round(elapsed, 4)
                print(f"  QuTiP:      {tp:>10.0f} gates/s  ({elapsed:.4f}s)")
            except Exception as e:
                row["qutip_throughput"] = None
                row["qutip_error"] = str(e)
                print(f"  QuTiP:      ERROR - {e}")
        else:
            row["qutip_throughput"] = None
            row["qutip_note"] = "skipped (>14 qubits)"
            print(f"  QuTiP:      SKIPPED (>14 qubits, memory limit)")
        
        all_results.append(row)
    
    # Save results
    output_path = os.path.expanduser("~/siliqun/benchmarks/comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'=' * 75}")
    print(f"{'Qubits':>6} | {'SiliQun(MPS)':>14} | {'Qiskit(SV)':>14} | {'QuTiP(SV)':>14} | SQ/Qiskit")
    print("-" * 75)
    for r in all_results:
        sq = r.get("siliqun_throughput", 0) or 0
        qk = r.get("qiskit_throughput", 0) or 0
        qt = r.get("qutip_throughput", 0) or 0
        ratio = f"{sq/qk:.2f}x" if qk > 0 else "N/A"
        qt_str = f"{qt:>12.0f}/s" if qt else "       N/A  "
        print(f"{r['n_qubits']:>6} | {sq:>12.0f}/s | {qk:>12.0f}/s | {qt_str} | {ratio}")
    print(f"{'=' * 75}")
