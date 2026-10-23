# Replication 2: DQN Universal State Preparation on Donor Qubits

This directory contains the replication of the He et al. (2025) universal state preparation experiment using SiliQun's `donor_electron` profile.

## Objective
Replicate the DQN-based universal state preparation task to validate SiliQun's donor spin Hamiltonian dynamics against published baselines.

## Running the Experiment
```bash
python3 run_e2.py --config config_e2.json
```
Results will be saved to the `results/` directory and logs to `logs/`.
