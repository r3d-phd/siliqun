"""
SiliQun - Silicon Qubits Simulator
A high-performance, modular simulator for silicon spin qubit quantum
computers, built on tensor network methods (MPS/MPO) for scalable
simulation of noisy quantum systems.
"""
__version__ = "1.0.0"
__author__ = "Raad Alshehri"

from .engine.gym_env import SiliQunEnv
from .physics.devices.profiles import DEVICE_REGISTRY as DEVICE_PROFILES

__all__ = ["SiliQunEnv", "DEVICE_PROFILES"]
