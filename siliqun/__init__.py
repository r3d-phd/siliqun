"""
SiliQun v2.0 — Silicon Qubits Simulator
========================================

A modular, tensor-network-based simulator for silicon spin qubit quantum
computers. SiliQun provides physically accurate simulation of donor (P:Si),
SiMOS, and gate-all-around (GAA) qubit architectures using Matrix Product
State (MPS) and Matrix Product Operator (MPO) representations.

Design Principles
-----------------
SiliQun is structured around three levels of abstraction:

1. **Core Simulator** — backend-agnostic tensor network engine (MPS/MPO)
   with NumPy, JAX, and CUDA backends.
2. **Technology Modules** — physics models specific to silicon spin qubits:
   Hamiltonians, native gate sets, and calibrated noise channels.
3. **Plugin Interface** — standardised extension points for new device
   types, noise models, evolution methods, and third-party integrations
   (PennyLane, OpenPulse, QASM3).

Quick Start
-----------
>>> from siliqun import SiliQunEnv, SiliQunSimulator, SimConfig
>>> from siliqun.physics.devices import get_device_profile
>>>
>>> # Create a 4-qubit SiMOS environment for DRL training
>>> env = SiliQunEnv(n_qubits=4, device="simos", target="GHZ",
...                  noise_stage=3, seed=42)
>>> obs, info = env.reset()
>>>
>>> # Or use the low-level simulator directly
>>> device = get_device_profile("donor", n_qubits=3)
>>> sim = SiliQunSimulator(device, SimConfig(noise_enabled=True))
>>> sim.reset()
>>> sim.apply_ry(1.5708, qubit=0)
>>> sim.apply_cnot(control=0, target=1)
>>> fidelity = sim.fidelity_to_target("GHZ")

References
----------
Weinstein et al., Nature 615, 817-822 (2023) — HRL SLEDGE device parameters.
Al-Shehri, R. (2025) — SiliQun: A Modular Tensor-Network Simulator for
Silicon Spin Qubit Quantum Computers. PhD Thesis, King Abdulaziz University.
"""

__version__ = "2.0.0"
__author__ = "Raad Al-Shehri"
__email__ = "ralshehri0468@stu.kau.edu.sa"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Primary public API — flat imports for convenience
# ---------------------------------------------------------------------------
from .engine.gym_env import SiliQunEnv
from .engine.simulator import SiliQunSimulator, SimConfig
from .engine.statevector_simulator import StatevectorSimulator
from .physics.devices.profiles import get_device_profile, DEVICE_REGISTRY
from .physics.gates import (
    # Single-qubit standard gates
    identity,
    pauli_x,
    pauli_y,
    pauli_z,
    hadamard,
    phase_gate,
    rx,
    ry,
    rz,
    t_gate,
    # Silicon-specific single-qubit gates
    esr_rotation,
    edsr_rotation,
    # Two-qubit gates
    cnot,
    cz,
    swap,
    sqrt_swap,
    exchange_gate,
    exchange_sqrt_swap,
    # MPO conversion utilities
    single_qubit_mpo_tensor,
    two_qubit_gate_to_mpo_tensors,
)
from .physics.noise.channels import (
    NoiseParams,
    default_noise_params,
    amplitude_damping_kraus,
    phase_damping_kraus,
    depolarizing_kraus,
    leakage_kraus,
    CrosstalkModel,
    ChargeNoiseGenerator,
    DFSNoiseModel,
    apply_t1_noise,
    apply_dephasing_noise,
    apply_leakage_noise,
    apply_crosstalk_noise,
    apply_charge_noise_dephasing,
)
from .tensor.mps import MPS
from .tensor.mpo import MPO
from .backend import active_backend, set_backend

# ---------------------------------------------------------------------------
# Target state builders (extracted from QUASAR codebase)
# ---------------------------------------------------------------------------
from .states import (
    bell_state,
    ghz_state,
    w_state,
    cluster_state,
    dicke_state,
    build_target_state,
    compute_fidelity,
)

# ---------------------------------------------------------------------------
# Noise curriculum (extracted from QUASAR v26/v27)
# ---------------------------------------------------------------------------
from .noise_curriculum import (
    NoiseCurriculum,
    get_noise_prob,
    NOISE_STAGE_PARAMS,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__ = [
    # Version
    "__version__",
    # Simulators
    "SiliQunEnv",
    "SiliQunSimulator",
    "SimConfig",
    "StatevectorSimulator",
    # Device profiles
    "get_device_profile",
    "DEVICE_REGISTRY",
    # Gates
    "identity", "pauli_x", "pauli_y", "pauli_z",
    "hadamard", "phase_gate", "rx", "ry", "rz", "t_gate",
    "esr_rotation", "edsr_rotation",
    "cnot", "cz", "swap", "sqrt_swap",
    "exchange_gate", "exchange_sqrt_swap",
    "single_qubit_mpo_tensor", "two_qubit_gate_to_mpo_tensors",
    # Noise
    "NoiseParams", "default_noise_params",
    "amplitude_damping_kraus", "phase_damping_kraus",
    "depolarizing_kraus", "leakage_kraus",
    "CrosstalkModel", "ChargeNoiseGenerator", "DFSNoiseModel",
    "apply_t1_noise", "apply_dephasing_noise", "apply_leakage_noise",
    "apply_crosstalk_noise", "apply_charge_noise_dephasing",
    # Tensor networks
    "MPS", "MPO",
    # Backends
    "active_backend", "set_backend",
    # Target states
    "bell_state", "ghz_state", "w_state", "cluster_state",
    "dicke_state", "build_target_state", "compute_fidelity",
    # Noise curriculum
    "NoiseCurriculum", "get_noise_prob", "NOISE_STAGE_PARAMS",
]
