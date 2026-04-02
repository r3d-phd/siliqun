# Figures V2 Notes

## fig_speed_scaling.png
- SiliQun throughput is nearly constant at ~91K steps/s across 2-16 qubits on Aziz HPC
- Theoretical statevector scaling drops exponentially
- Clean, publication-quality figure

## fig_comparison.png
- Left panel: Log-scale throughput comparison shows SiliQun flat, Qiskit/QuTiP declining
- Right panel: Speedup bars show 6x at 2q, growing to 29x (vs Qiskit) and 41x (vs QuTiP) at 14q
- Very compelling visualization

## fig_noise_overhead.png
- Noise overhead ranges from 20-39%, averaging ~32%
- Clean vs noisy throughput clearly shown

## fig_bond_dim.png
- Throughput nearly constant across chi=2 to chi=64 for 8 qubits
- Shows MPS efficiency for low-entanglement states
