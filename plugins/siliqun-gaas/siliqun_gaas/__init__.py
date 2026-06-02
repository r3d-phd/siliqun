"""
siliqun-gaas
============
SiliQun plugin for gate-all-around (GAA) silicon spin qubits.

Minimum plugin contract (47 lines):
  - gaa_profile.py        (28 lines) — TechnologyProfile subclass
  - gaa_calibration.json  (12 lines) — calibration data with DOIs
  - gaa_benchmarks.py     ( 7 lines) — three PGIRS Phase-1 benchmarks

Usage::

    from siliqun_gaas import GAAProfile
    profile = GAAProfile()
    profile.register()                    # validates + registers Gymnasium env
    import gymnasium as gym
    env = gym.make("SiliQun-GAA-v1")

Source: Tanamoto and Ono, Phys. Rev. Applied 23, 034001 (2025).
"""

from .gaa_profile import GAAProfile
from .gaa_benchmarks import run_pgirs_phase1, assert_pgirs_phase1

__all__ = ["GAAProfile", "run_pgirs_phase1", "assert_pgirs_phase1"]
__version__ = "1.0.0"
