# Extension 4: Multi-Target Scalability

This directory contains the core scalability experiments for the SiliQun framework, testing DRL agents against classical baselines (GRAPE, GAA) across multiple target states and qubit counts.

## Objective
Evaluate the scalability of PPO and SAC in generating high-fidelity (F > 0.999) preparation sequences for GHZ, W, Cluster-1D, and Dicke-k3 states from N=2 up to N=12 qubits on the `simos_nominal` profile.

## Running the Experiment
```bash
python3 run_e4.py --config config_e4.json
```
*(Note: The original v6 script is preserved in the `siliqun_e4_v6/` directory for historical reference).*
