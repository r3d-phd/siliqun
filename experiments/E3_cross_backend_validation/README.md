# Replication 3: Cross-Backend Validation on Singlet-Triplet Qubits

This directory contains the replication of the Abedi & Schmitt (2025) singlet-triplet qubit control experiment, extended to validate SiliQun's policy transferability across Qiskit Aer and IBM FakeSherbrooke backends.

## Objective
Train a DQN agent on SiliQun's internal `singlet_triplet` Hamiltonian dynamics, then execute the resulting control pulse sequence on external quantum simulators to verify fidelity consistency.

## Running the Experiment
```bash
# Run training on SiliQun internal engine
python3 run_e3_train.py --config config_e3.json

# Validate policy on Qiskit Aer Statevector
python3 run_e3_qiskit.py --config config_e3.json

# Validate policy on IBM FakeSherbrooke (noisy)
python3 run_e3_fakesherbrooke.py --config config_e3.json
```
