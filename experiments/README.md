# SiliQun Experiments and Reproducibility

This directory contains the complete experimental record for the SiliQun framework as presented in the paper **"SiliQun: A Deep Reinforcement Learning Framework for Silicon Spin Qubit Control"** (IEEE Transactions on Quantum Engineering, 2026).

All experiments were conducted on the King Abdulaziz University Aziz HPC cluster (NVIDIA A100-PCIE-40GB, CUDA 12.1, PyTorch 2.5.1). Results are reported as mean ± std across five independent seeds.

## Directory Structure

```
experiments/
├── README.md                          ← This file
├── requirements.txt                   ← Pinned dependencies for exact reproduction
├── seeds.json                         ← All random seeds used across experiments
├── results_summary.csv                ← Aggregated fidelity statistics (machine-readable)
├── runtime_profile.md                 ← Hardware specs and wall-clock times
│
├── E1_replication_daraeizadeh/        ← Replication 1: DQL/SARSA/PPO on SiMOS
│   ├── README.md
│   ├── run_e1.py
│   ├── config_e1.json
│   ├── logs/                          ← Runtime logs
│   └── results/                       ← Raw JSON result files
│
├── E2_replication_he/                 ← Replication 2: DQN on Donor Qubits
│   ├── README.md
│   ├── run_e2.py
│   ├── config_e2.json
│   ├── logs/
│   └── results/
│
├── E3_cross_backend_validation/       ← Replication 3: Cross-Backend Validation
│   ├── README.md
│   ├── run_e3_train.py
│   ├── run_e3_qiskit.py
│   ├── run_e3_fakesherbrooke.py
│   ├── config_e3.json
│   ├── logs/
│   └── results/
│
├── E4_multitarget_scalability/        ← Extension 4: Multi-Target Scalability
│   ├── README.md
│   ├── run_e4.py
│   ├── config_e4.json
│   ├── siliqun_e4_v6/                 ← Original v6 source (historical reference)
│   ├── logs/
│   └── results/
│
└── plots/                             ← All publication-quality figures
    ├── ablation/                      ← Algorithm comparison (PPO, SAC, GRAPE, GAA)
    ├── fidelity_vs_episode/           ← Convergence curves
    ├── ppo_reward_curves/             ← PPO training metrics per seed
    ├── scalability/                   ← Fidelity vs qubit count (N=2 to 12)
    └── tensorboard_scalars/           ← TensorBoard exported scalars
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Replication 1 (DQL/SARSA/PPO on SiMOS)
cd E1_replication_daraeizadeh && python3 run_e1.py

# Run Extension 4 (Multi-Target Scalability)
cd E4_multitarget_scalability && python3 run_e4.py

# Regenerate all plots from raw JSON data
cd .. && python3 generate_plots.py
```

## Key Results Summary

| Experiment | Best Algorithm | Best Fidelity | Hardware |
|-----------|---------------|---------------|----------|
| E1 (SiMOS CZ gate) | PPO | F = 1.0000 | A100 |
| E2 (Donor Universal) | DQN | F = 0.9999 | A100 |
| E3 (Cross-Backend) | DQN | F = 0.9995 | A100 |
| E4 (3Q GHZ) | PPO | F = 0.9999 | A100 |

See `results_summary.csv` for the full per-seed breakdown and `runtime_profile.md` for hardware details.
