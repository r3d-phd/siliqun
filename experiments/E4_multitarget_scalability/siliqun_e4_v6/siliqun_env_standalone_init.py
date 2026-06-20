"""
SiliQun — Silicon Spin Qubit Simulator
=======================================
A standalone, physics-accurate simulator for silicon spin qubits under
realistic noise models (charge noise, phonon dephasing, ZZ Ising coupling).

Designed for use as a Gym-compatible reinforcement learning environment.
Completely independent of QUASAR, ANDROMEDA, or any other project.

Package layout:
  siliqun/
  ├── __init__.py     ← this file: version, public API
  ├── envs.py         ← SiliQunEnv (core Gym-compatible environment)
  ├── targets.py      ← target state factory (GHZ, W, Cluster, Dicke-k)
  └── noise.py        ← noise model helpers (charge noise, T1/T2 dephasing)

Author: Raad Al-Shehri | KAU FCIT PhD
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__  = "Raad Al-Shehri"

from siliqun.envs import SiliQunEnv

__all__ = ["SiliQunEnv"]
