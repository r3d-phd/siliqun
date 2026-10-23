# SiliQun E4 v6 — SAC Training Harness

Standalone SAC training harness for silicon spin qubit state preparation.

## Key Changes from v5
- **Fully decoupled from QUASAR**: replaced `from quasar_v9.envs import QuasarEnv`
  with `from siliqun.envs import SiliQunEnv`
- All log messages, result metadata, and PBS references updated
- `env` field in result JSON now reads `"SiliQunEnv-v1.0"` (was `"QuasarEnv_v9.54"`)
- Version bumped to v6.0.0

## Usage
```bash
# Single cell
python3 siliqun_e4_v6.py --n-qubits 3 --target ghz --seed 42

# Full sweep (2Q–6Q, all targets, 3 seeds)
python3 siliqun_e4_v6.py --sweep --qubit-range 2 6 --targets ghz w cluster_linear dicke_k3
```

## Dependencies
- `siliqun` package (this directory or `pip install -e /path/to/siliqun_package`)
- `torch`, `numpy`
