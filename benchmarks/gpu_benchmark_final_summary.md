# SiliQun GPU Benchmark Results — Aziz A100

**Job:** 169637.khead2 | **Node:** kcn501 | **GPU:** NVIDIA A100-PCIE-40GB (CUDA 12.4)  
**CuPy:** 13.0.0 | **Python:** 3.12.4 | **Date:** 2026-04-03

## Raw CuPy GPU vs CPU Comparison

| Test | CPU (NumPy) | GPU (CuPy) | Speedup |
|------|-------------|------------|---------|
| 33M element-wise multiply | 142.65 ms | 47.05 ms | **3.0x** |
| 16-qubit 1Q gate | 232 μs | 2,774 μs | 0.1x (GPU overhead dominates) |
| 25-qubit 1Q gate | 78.55 ms | 7.64 ms | **10.3x** |

**Key insight:** GPU shines at 25 qubits (10x speedup) but has too much kernel launch overhead for small systems (16 qubits). This is expected — the A100 needs large arrays to saturate its bandwidth.

## SiliQun StateVectorSimulator Benchmark

The benchmark script used NumPy arrays (the simulator's CuPy auto-detection needs the `n_qubits` constructor parameter fixed). These are CPU-only numbers from the A100 node:

| Configuration | Qubits | Dim | 1Q Gate | 2Q Gate | ⟨Z⟩ | S(A:B) | SV Mem |
|---------------|--------|-----|---------|---------|------|--------|--------|
| donor_2q | 2 | 4 | 120 μs | 77 μs | 178 μs | 10.6 ms | 0 MB |
| donor_4q | 4 | 16 | 98 μs | 72 μs | 106 μs | 0.39 ms | 0 MB |
| donor_8q | 8 | 256 | 74 μs | 59 μs | 58 μs | 0.27 ms | 0 MB |
| sledge_2x2 | 4 | 16 | 49 μs | 39 μs | 56 μs | 0.16 ms | 0 MB |
| sledge_3x3 | 9 | 512 | 60 μs | 54 μs | 56 μs | 0.90 ms | 0 MB |
| **sledge_4x4** | **16** | **65K** | **1.13 ms** | **1.71 ms** | **386 μs** | **86 ms** | **1 MB** |
| **sledge_5x5** | **25** | **33.5M** | **408 ms** | **606 ms** | **163 ms** | **12.9 s** | **512 MB** |

## Projected GPU-Accelerated SiliQun Numbers

Based on the raw CuPy speedups measured above, the projected GPU-accelerated SiliQun numbers for 25 qubits:

| Metric | CPU (measured) | GPU (projected, 10x) |
|--------|---------------|---------------------|
| 1Q gate | 408 ms | ~41 ms |
| 2Q gate | 606 ms | ~61 ms |
| ⟨Z⟩ | 163 ms | ~16 ms |
| S(A:B) | 12.9 s | ~1.3 s |

## Next Steps

1. Fix the `StateVectorSimulator` constructor to accept `n_qubits` properly so the benchmark uses CuPy arrays
2. Re-run to get actual GPU-accelerated SiliQun numbers
3. The 10x speedup at 25 qubits makes DRL training feasible (~41 ms/gate vs 408 ms/gate)
