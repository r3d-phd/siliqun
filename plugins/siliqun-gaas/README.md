# siliqun-gaas

SiliQun plugin for **gate-all-around (GAA) silicon spin qubits**, calibrated to
Tanamoto and Ono, *Phys. Rev. Applied* 23, 034001 (2025).

This plugin demonstrates the **minimum plugin contract**: 47 lines of
platform-specific code that integrate a new hardware technology into the
SiliQun framework.  The Lindblad solver, Gymnasium interface, and RL agent
are provided by the SiliQun core — no modifications required.

## Installation

```bash
pip install siliqun          # install SiliQun core first
pip install -e .             # install this plugin in development mode
```

## Usage

```python
from siliqun_gaas import GAAProfile
import gymnasium as gym

# Validate and register the Gymnasium environment
profile = GAAProfile()
profile.register()           # runs 3 PGIRS Phase-1 benchmarks; raises if any fail

# Use with any Gymnasium-compatible RL library
env = gym.make("SiliQun-GAA-v1")
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"Final fidelity: {info.get('fidelity', 'N/A'):.4f}")
```

## Plugin Contract (47 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `siliqun_gaas/gaa_profile.py` | 28 | `TechnologyProfile` subclass with 5 abstract methods |
| `siliqun_gaas/gaa_calibration.json` | 12 | Calibration data with source DOIs |
| `siliqun_gaas/gaa_benchmarks.py` | 7 | Three PGIRS Phase-1 benchmark calls |

## Validation

```python
from siliqun_gaas import run_pgirs_phase1
results = run_pgirs_phase1()
print(results)  # {"rabi": True, "bell": True, "t2_echo": True}
```

## Hardware Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| T₁ | 0.5 ms | Tanamoto & Ono 2025 |
| T₂ | 0.1 ms | Tanamoto & Ono 2025 |
| σ_ε (charge noise) | 2.0 µeV | Tanamoto & Ono 2025 |
| σ_J (exchange noise) | 1.0 MHz | Tanamoto & Ono 2025 |
| τ₁ (single-qubit gate) | 5 ns | Tanamoto & Ono 2025 |
| τ₂ (two-qubit gate) | 50 ns | Tanamoto & Ono 2025 |
| p₂ (two-qubit error) | 1.5 % | Tanamoto & Ono 2025 |

## Citation

```bibtex
@article{Tanamoto2025,
  author  = {Tanamoto, Tetsufumi and Ono, Keiji},
  title   = {Noise characterization of gate-all-around silicon spin qubits via TCAD simulation},
  journal = {Physical Review Applied},
  volume  = {23},
  pages   = {034001},
  year    = {2025},
  doi     = {10.1103/PhysRevApplied.23.034001}
}
```
