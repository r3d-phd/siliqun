# SiliQun — Silicon Qubits Simulator

## Architecture Design Document v1.0

---

## 1. Overview

SiliQun is a high-performance, modular, open-source simulator for silicon spin qubit quantum computers. It is designed to serve as the foundational physics engine for the QUASAR, MOZAIQ, and SeQurAIty research pillars, providing hardware-accurate pulse-level simulation of silicon spin qubit devices at scales up to 100 noisy qubits.

The simulator leverages Matrix Product Operator (MPO) tensor network representations to exploit the bounded entanglement structure of noisy quantum states, enabling polynomial-time simulation of systems that would otherwise require exponential resources.

---

## 2. Design Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Each layer is independently testable and replaceable via clean interfaces |
| **Backend Agnosticism** | Core algorithms work with NumPy, JAX, or cuQuantum backends |
| **Physics Fidelity** | Hamiltonians and noise models are derived from published experimental data |
| **DRL Integration** | Native Gymnasium environment wrapper for seamless QUASAR integration |
| **Scalability** | MPO-based evolution scales polynomially with qubit count for noisy systems |
| **Extensibility** | Plugin architecture for new device types, noise models, and evolution methods |

---

## 3. Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 5: API & Integration            │
│  Gymnasium Env │ CLI │ REST API │ Visualization │ I/O   │
├─────────────────────────────────────────────────────────┤
│                    Layer 4: Simulation Engine            │
│  TimeEvolver │ PulseScheduler │ MeasurementEngine       │
├─────────────────────────────────────────────────────────┤
│                    Layer 3: Physics Models               │
│  Hamiltonians │ NoiseModels │ DeviceFactory │ Topology  │
├─────────────────────────────────────────────────────────┤
│                    Layer 2: Tensor Network Engine        │
│  MPO │ MPS │ Contraction │ Truncation │ Decomposition   │
├─────────────────────────────────────────────────────────┤
│                    Layer 1: Compute Backend              │
│  NumPy │ JAX │ cuQuantum │ MPI Distribution             │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Package Structure

```
siliqun/
├── __init__.py                 # Version, public API
├── backend/                    # Layer 1: Compute Backend
│   ├── __init__.py
│   ├── base.py                 # Abstract backend interface
│   ├── numpy_backend.py        # NumPy reference implementation
│   ├── jax_backend.py          # JAX GPU-accelerated backend
│   └── cuquantum_backend.py    # NVIDIA cuQuantum backend
├── tensor/                     # Layer 2: Tensor Network Engine
│   ├── __init__.py
│   ├── tensor.py               # Core Tensor class
│   ├── mps.py                  # Matrix Product State
│   ├── mpo.py                  # Matrix Product Operator
│   ├── contraction.py          # Contraction strategies
│   ├── truncation.py           # SVD truncation & bond management
│   └── decomposition.py        # Tensor decompositions (SVD, QR, etc.)
├── physics/                    # Layer 3: Physics Models
│   ├── __init__.py
│   ├── constants.py            # Physical constants (g-factors, μ_B, etc.)
│   ├── hamiltonian.py          # Hamiltonian builder (Zeeman, exchange, etc.)
│   ├── noise/                  # Noise model sub-package
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract noise model
│   │   ├── charge_noise.py     # 1/f charge noise (TLS ensemble)
│   │   ├── nuclear_noise.py    # 29Si nuclear spin bath
│   │   ├── phonon.py           # Phonon-induced T1 relaxation
│   │   └── crosstalk.py        # Inter-qubit crosstalk
│   ├── devices/                # Device variant sub-package
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract device specification
│   │   ├── donor.py            # 31P donor in silicon
│   │   ├── simos.py            # SiMOS quantum dot
│   │   └── gaa.py              # Gate-all-around nanowire
│   └── topology.py             # Qubit connectivity graphs
├── engine/                     # Layer 4: Simulation Engine
│   ├── __init__.py
│   ├── time_evolver.py         # TEBD/TDVP time evolution
│   ├── pulse_scheduler.py      # Pulse sequence compilation
│   ├── measurement.py          # Projective & weak measurement
│   ├── fidelity.py             # Fidelity metrics (process, state, gate)
│   └── channel.py              # Quantum channel representations
├── api/                        # Layer 5: API & Integration
│   ├── __init__.py
│   ├── gym_env.py              # Gymnasium environment wrapper
│   ├── cli.py                  # Command-line interface
│   ├── visualization.py        # State & circuit visualization
│   └── io.py                   # Import/export (OpenQASM, Qiskit, etc.)
├── config/                     # Configuration
│   ├── __init__.py
│   ├── defaults.py             # Default simulation parameters
│   └── device_params/          # Device-specific parameter files
│       ├── donor_31P.yaml
│       ├── simos_default.yaml
│       └── gaa_default.yaml
└── tests/                      # Test suite
    ├── test_backend.py
    ├── test_tensor.py
    ├── test_physics.py
    ├── test_engine.py
    ├── test_gym_env.py
    └── benchmarks/
        ├── bench_scaling.py
        └── bench_fidelity.py
```

---

## 5. Layer Details

### 5.1 Layer 1: Compute Backend

The compute backend provides an abstract interface for array operations, enabling the same simulation code to run on CPU (NumPy), GPU (JAX), or multi-GPU clusters (cuQuantum). The interface follows the adapter pattern.

```python
class Backend(ABC):
    """Abstract compute backend interface."""
    @abstractmethod
    def zeros(self, shape, dtype): ...
    @abstractmethod
    def eye(self, n, dtype): ...
    @abstractmethod
    def svd(self, tensor, full_matrices=False): ...
    @abstractmethod
    def qr(self, tensor): ...
    @abstractmethod
    def tensordot(self, a, b, axes): ...
    @abstractmethod
    def expm(self, matrix): ...    # Matrix exponential for time evolution
    @abstractmethod
    def eigh(self, matrix): ...    # Hermitian eigendecomposition
```

### 5.2 Layer 2: Tensor Network Engine

The tensor network layer implements MPS and MPO data structures with efficient contraction, truncation, and decomposition algorithms. This is the core of SiliQun's scalability.

**Key classes:**
- `MPS`: Matrix Product State representation of pure quantum states
- `MPO`: Matrix Product Operator representation of mixed states and operators
- `MPOEvolver`: Applies MPO gates/channels to MPS/MPO states with truncation

**Bond dimension management** is critical: after each gate application, the bond dimension is truncated via SVD to maintain a maximum bond dimension χ_max, with a configurable truncation error threshold ε.

### 5.3 Layer 3: Physics Models

The physics layer constructs the Hamiltonian and noise models specific to silicon spin qubits.

**Hamiltonian builder** constructs the full system Hamiltonian as an MPO:

```
H = H_Zeeman + H_Exchange + H_Hyperfine + H_SOC + H_Valley
```

Each term is constructed as a local MPO and combined using MPO addition.

**Noise models** generate stochastic realizations of noise processes:
- `ChargeNoise`: Generates 1/f noise trajectories via TLS ensemble sampling
- `NuclearNoise`: Models 29Si nuclear spin bath using cluster expansion
- `PhononRelaxation`: T1 decay via Lindblad master equation in MPO form

**Device factory** creates pre-configured device specifications:
```python
device = DeviceFactory.create("donor_31P", n_qubits=8, B0=1.4)
device = DeviceFactory.create("simos", n_qubits=20, valley_splitting=0.1)
```

### 5.4 Layer 4: Simulation Engine

The simulation engine orchestrates time evolution, pulse scheduling, and measurement.

**Time evolution methods:**
- `TEBD`: Time-Evolving Block Decimation (2nd and 4th order Trotter)
- `TDVP`: Time-Dependent Variational Principle (1-site and 2-site)
- `WII`: Wiener-Itô Integration for stochastic noise

**Pulse scheduler** compiles high-level gate operations into time-dependent Hamiltonian parameters:
```python
schedule = PulseScheduler(device)
schedule.add_gate("CNOT", qubits=[0, 1], duration=100e-9)
schedule.add_gate("Rz", qubits=[2], angle=np.pi/4, duration=10e-9)
```

### 5.5 Layer 5: API & Integration

**Gymnasium environment** provides the DRL interface for QUASAR:
```python
env = SiliQunEnv(device="donor_31P", n_qubits=4, target_gate="CNOT")
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
```

The observation space includes fidelity estimates, spectral features, and noise diagnostics. The action space represents pulse parameters (amplitude, frequency, phase, duration).

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **quimb as primary TN backend** | Best balance of features, GPU support, and quantum circuit integration |
| **MPO for mixed states** | Enables efficient simulation of noisy systems with bounded entanglement |
| **TEBD as default evolver** | Well-understood, parallelizable, efficient for nearest-neighbor interactions |
| **YAML device configs** | Human-readable, version-controllable device parameter files |
| **Gymnasium API** | Standard DRL interface, direct QUASAR compatibility |
| **Plugin noise models** | Strategy pattern allows swapping noise models without changing engine code |

---

## 7. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| 2-qubit gate simulation | < 1 ms | Real-time DRL training feasibility |
| 8-qubit full simulation | < 100 ms | QUASAR target system size |
| 50-qubit noisy MPO (χ=64) | < 10 s | MOZAIQ modular validation |
| 100-qubit noisy MPO (χ=32) | < 60 s | Maximum scale demonstration |
| Memory (50 qubits, χ=64) | < 4 GB | Fits on single A100 GPU |

---

## 8. Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | Reference backend | Yes |
| scipy | Linear algebra, special functions | Yes |
| quimb | Tensor network operations | Yes |
| gymnasium | DRL environment interface | Yes |
| pyyaml | Device configuration files | Yes |
| jax | GPU-accelerated backend | Optional |
| cuquantum | NVIDIA GPU tensor networks | Optional |
| mpi4py | Distributed simulation | Optional |
| matplotlib | Visualization | Optional |

---

## 9. Integration Points

### 9.1 QUASAR Integration
SiliQun's `SiliQunEnv` replaces the current `SiliconSpinEnv` as a drop-in Gymnasium environment. QUASAR's CAMEL-Q, SDFT, SWDFT, and ERL modules interact with SiliQun through the standard `step()` / `reset()` API.

### 9.2 MOZAIQ Integration
SiliQun's topology module supports modular architectures. Multiple SiliQun instances can represent individual MOZAIQ modules, with inter-module entanglement tracked via the MPO bond dimensions at partition boundaries.

### 9.3 SeQurAIty Integration
SiliQun's noise model API allows programmatic injection of adversarial noise patterns. SeQurAIty's threat models can modify the `ChargeNoise` or `CrosstalkNoise` parameters mid-simulation to model physical-layer attacks.
