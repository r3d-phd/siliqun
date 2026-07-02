# SiliQun Comparison Benchmark Results

## Environment
- CPU-only sandbox (6 cores, ~3.85 GB RAM)
- SiliQun v0.1.0 (MPS/tensor network backend)
- Qiskit Aer 0.17.2 (statevector method)
- QuTiP 5.2.3 (full state vector)

## Results Summary

| Qubits | SiliQun (MPS) | Qiskit Aer (SV) | QuTiP (SV) | SiliQun/Qiskit |
|--------|---------------|------------------|-------------|----------------|
| 2      | 460,839/s     | 83,104/s         | 500,115/s   | 5.55x          |
| 4      | 468,882/s     | 84,028/s         | 447,117/s   | 5.58x          |
| 6      | 477,126/s     | 84,548/s         | 376,387/s   | 5.64x          |
| 8      | 465,071/s     | 80,815/s         | 283,657/s   | 5.75x          |
| 10     | 463,393/s     | 69,540/s         | 160,242/s   | 6.66x          |
| 12     | 458,520/s     | 42,314/s         | 49,203/s    | 10.84x         |
| 14     | 468,677/s     | 16,359/s         | 11,533/s    | 28.65x         |
| 16     | 444,857/s     | 61,824/s         | N/A         | 7.20x          |

## Key Findings

1. **SiliQun is consistently 5-29x faster than Qiskit Aer** for single-gate throughput
2. **SiliQun scales nearly flat** (~450K gates/s) across 2-16 qubits — MPS advantage
3. **Qiskit Aer and QuTiP degrade exponentially** with qubit count (statevector 2^n scaling)
4. **At 14 qubits, SiliQun is 28.65x faster** than Qiskit Aer and 40.6x faster than QuTiP
5. **SiliQun throughput is nearly constant** because MPS bond dimension stays bounded for H gates

## Important Caveats

- This compares **gate-level throughput only**, not full circuit simulation accuracy
- Qiskit Aer's statevector is exact; SiliQun's MPS is approximate (bounded chi)
- For highly entangled states, SiliQun would need higher chi and slow down
- Qiskit Aer has circuit-level optimizations that reduce overhead for full circuits
- The 16-qubit Qiskit anomaly (faster than 14q) may be due to Aer's internal optimizations
- These are sandbox CPU results; Aziz HPC results will be more authoritative
