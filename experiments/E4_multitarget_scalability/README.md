# Extension 4: Multi-Target Scalability

This directory contains the core scalability experiments for the SiliQun framework, training PPO and SAC agents across multiple target states and qubit counts, with GRAPE/CRAB oracle baselines on the two-qubit SiMOS CZ task.

## Objective
Evaluate the scalability of PPO and SAC in generating high-fidelity (F > 0.999) preparation sequences for GHZ, W, Cluster-1D, and Dicke-k3 states at N=2 (three-seed scaling runs: `scaling__2Q_*.json`) and N=3 (four targets × three seeds, H100 NVL jobs 180521/180658) on the `simos_nominal` profile. The N=4--12 extension advertised in the original draft was not executed; no artifacts exist for it (2026-10-13 audit).

## Running the Experiment
```bash
python3 run_e4.py --config config_e4.json
```
*(Note: The original v6 script is preserved in the `siliqun_e4_v6/` directory for historical reference).*
