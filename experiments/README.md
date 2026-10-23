# SiliQun Experiments and Reproducibility

This directory contains the complete experimental record for the SiliQun framework as presented in the paper **"SiliQun: A Deep Reinforcement Learning Framework for Silicon Spin Qubit Control"** (IEEE Transactions on Quantum Engineering, 2026).

Experiments ran on the King Abdulaziz University Aziz HPC cluster (NVIDIA A100-PCIE-40GB for E1; H100 NVL for the E4 multi-target jobs 180521/180658); software versions per `requirements.txt` and the run-log headers (PyTorch 2.4.0, Stable-Baselines3 2.8.0). Seed sets per `seeds.json` (primary seeds 0--4; additional seeds 42/123/456/789/1000/2024).

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

| Experiment | Algorithm | Best Fidelity | Status | Hardware |
|-----------|-----------|---------------|--------|----------|
| E1 (SiMOS CZ gate) | PPO | 0.8215 | NEGATIVE vs published 0.999 (0/3 seeds) | A100 |
| E1-R4 (donor TQC 4-qubit) | DQN | 0.8444 | NEGATIVE vs 0.99 (0/4 seeds) | A100/H100 |
| E2 (Donor Universal) | DQN | — | PENDING_REGENERATION (no per-seed artifact) | A100 |
| E3 (GAA nanowire) | SAC | 0.9705 final; ~1.0 in-training | PARTIAL (0/4 runs final >= 0.99) | A100 |
| E4 2Q cluster | SAC | 0.9994 | PASS (1/3 seeds) | A100 |
| E4 2Q GHZ / Dicke-k3 | SAC | 0.9869 / 0.9876 | FAIL (< 0.99) | A100 |
| E4 3Q multi-target | SAC | 0.4438 (best callback) | FAIL (non-converged) | H100 NVL |

The original summary table (E1 PPO 1.0000, E2 DQN 0.9999, E3 DQN 0.9995, E4 3Q PPO 0.9999) and the DQL/SARSA rows have been removed: no per-seed artifacts exist for those values (see `results_summary.csv`, 2026-10-13 audit).

See `results_summary.csv` for the full per-seed breakdown and `runtime_profile.md` for hardware details.
