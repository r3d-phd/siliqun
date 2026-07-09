"""
SiliQun v2.2 - Universal Silicon Spin Qubit Simulator

A standards-compliant, pulse-level simulation platform for silicon spin qubits.
Provides a Lindblad master equation solver, OpenQASM 3.0 and OpenPulse support,
Qiskit BackendV2 and PennyLane Device API plugins, and a Gymnasium RL environment
for deep reinforcement learning research on silicon quantum hardware.

Architecture:
  siliqun.engine       — Gymnasium RL environment, MPS/MPO and state-vector simulators
  siliqun.pulse        — Lindblad solver, OpenPulse schedule representation
  siliqun.compiler     — OpenQASM 3.0 parser, gate-to-pulse compiler
  siliqun.physics      — DFS encoding, noise channels, device profiles (SiMOS, P:Si, GAA)
  siliqun.plugins      — Qiskit BackendV2, PennyLane Device API
  siliqun.backend      — NumPy (CPU) and CuPy (CUDA GPU) numerical backends
  siliqun.tensor       — MPS/MPO tensor network primitives
  siliqun.tomography   — Quantum process and state tomography
  siliqun.hpc          — PBS/SLURM job generation for Aziz HPC and similar clusters
  siliqun.library      — SiliQunLib: 50 pre-trained primitive gate policy checkpoints
"""

__version__ = "2.3.0"
__author__ = "Raad Alshehri"
__email__ = "ralshehri0468@stu.kau.edu.sa"
__license__ = "MIT"
