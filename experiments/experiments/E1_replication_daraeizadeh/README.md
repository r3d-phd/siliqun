# Replication 1: DQL and SARSA on SiMOS

This directory contains the replication of the Daraeizadeh et al. (2020) CZ gate-synthesis experiment using SiliQun's `simos_nominal` profile.

## Objective
Replicate the continuous-parameter search for a CZ gate using Deep Q-Learning (DQL) and tabular SARSA, demonstrating SiliQun's ability to match published physics baselines.

## Running the Experiment
```bash
python3 run_e1.py --config config_e1.json
```
Results will be saved to the `results/` directory and logs to `logs/`.
