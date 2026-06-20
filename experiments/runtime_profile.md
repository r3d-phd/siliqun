# SiliQun Runtime Profile

All experiments were executed on the **King Abdulaziz University Aziz HPC cluster**.

## Hardware Specifications
- **Node Type**: GPU Compute Node
- **GPU**: NVIDIA A100-PCIE-40GB (HBM2)
- **CPU**: Intel Xeon Gold 6248R (3.00 GHz)
- **RAM**: 64 GB allocated per job

## Software Environment
- **OS**: Red Hat Enterprise Linux 8.4
- **CUDA**: 12.1
- **PyTorch**: 2.5.1+cu121
- **Python**: 3.11.5

## Wall-Clock Execution Times (per 5 seeds)
- **Replication 1 (E1)**: ~4.0 hours
- **Replication 2 (E2)**: ~0.9 hours
- **Replication 3 (E3)**: ~0.8 hours
- **Extension 4 (E4)**: ~12.5 hours (up to N=12)

*Note: SiliQun's tensorised Hamiltonian engine achieves >10x speedup over standard QuTiP solvers on A100 hardware, enabling deep multi-seed DRL training.*
