# SiliQun

**A GPU-Accelerated Tensor-Network Gymnasium Environment for Deep Reinforcement Learning-Based Control of Silicon Spin Qubits**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange.svg)](https://gymnasium.farama.org/)

SiliQun is an open-source quantum simulation platform purpose-built for training deep reinforcement learning (DRL) agents on silicon spin qubit control tasks. It provides physically calibrated device models, decoherence-free subspace (DFS) encoding, and a dual-engine architecture combining matrix product state (MPS) tensor networks with a GPU-accelerated exact state vector backend.

As of v2.2, SiliQun supports the **1D Cluster state** (`cluster1d`) as a target for generalisation studies, enabling curriculum-based training to transfer from GHZ entanglement to graph-state entanglement under the same SiMOS noise model.

## Key Features

- **Dual Simulation Backends:** MPS tensor network engine for approximate simulation and GPU-accelerated state vector engine for exact logical-subspace simulation up to 25 qubits (5x5 grids).
- **Silicon-Specific Device Physics:** Three calibrated device profiles (Donor, SiMOS, GAA) with realistic T1, T2*, charge noise (1/f spectrum), and crosstalk models.
- **Formal Plugin Interface:** Add any new qubit platform in under 50 lines via the `TechnologyProfile` ABC — the Lindblad solver, Gymnasium interface, and RL agent require no modification.
- **DFS Logical Encoding:** Automatic mapping from physical spin triplets to logical qubits with perturbative leakage tracking.
- **Native Gymnasium Interface:** Fully compliant with the [Gymnasium](https://gymnasium.farama.org/) API for seamless integration with standard DRL libraries (Stable-Baselines3, Ray RLlib, CleanRL).
- **Automatic Backend Selection:** The `"auto"` mode selects MPS for small systems (4 qubits or fewer) and GPU state vector for larger grids.
- **SLEDGE Grid Topologies:** Built-in support for 2x2, 3x3, 4x4, and 5x5 2D spin qubit arrays with nearest-neighbour exchange couplings.
- **1D Cluster State Target:** Native support for the open-boundary 1D Cluster state (`cluster1d`) in both MPS and state vector modes, enabling generalisation benchmarks across entanglement structures.

## Installation

### Prerequisites

- Python 3.9 or later
- NumPy 1.24 or later
- SciPy 1.10 or later

### Basic Installation (CPU only)

```bash
git clone https://github.com/r3d-phd/siliqun.git
cd siliqun
pip install -e .
```

### GPU Installation (recommended for 16 qubits or more)

For NVIDIA GPUs with CUDA 12.x:

```bash
pip install cupy-cuda12x
pip install -e .
```

For CUDA 11.x:

```bash
pip install cupy-cuda11x
pip install -e .
```

SiliQun automatically detects CuPy at runtime and uses GPU acceleration when available. No code changes are required.

### Verifying the Installation

```bash
python -c "from siliqun.engine import StateVectorSimulator; print('SiliQun OK')"
```

To verify GPU support:

```python
from siliqun.engine import StateVectorSimulator
from siliqun.physics.devices.profiles import sledge_5x5

sim = StateVectorSimulator(sledge_5x5(), use_gpu=True)
print(f"Backend: {sim.backend_name}")  # Should print "cupy" if GPU is available
```

## Quick Start

### 1. Direct Simulator Usage

```python
import numpy as np
from siliqun.engine import StateVectorSimulator
from siliqun.physics.devices.profiles import sledge_3x3

# Create a 9-qubit (3x3 grid) simulator
config = sledge_3x3()
sim = StateVectorSimulator(config, use_gpu=True)

# Apply quantum gates
sim.apply_rx(qubit=0, theta=np.pi / 2)
sim.apply_cnot(control=0, target=1)

# Measure observables
z_exp = sim.expectation_z(qubit=0)
fidelity = sim.fidelity(target_state=sim.psi)
entropy = sim.entanglement_entropy(qubit=0)

print(f"<Z_0> = {z_exp:.4f}")
print(f"Fidelity = {fidelity:.6f}")
print(f"S(0) = {entropy:.4f}")
```

### 2. Gymnasium RL Environment

```python
from siliqun.engine.gym_env import make_siliqun_env

# Create a 4-qubit environment with auto backend selection
env = make_siliqun_env(
    device="donor_2q",
    target="bell",
    sim_mode="auto",
    max_steps=100
)

obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Replace with your DRL agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final fidelity: {info.get('fidelity', 'N/A')}")
```

### 3. Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from siliqun.engine.gym_env import make_siliqun_env

env = make_siliqun_env(device="sledge_3x3", target="ghz", sim_mode="auto")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_siliqun_3x3")
```

## Architecture

SiliQun follows a four-layer modular architecture:

```
+---------------------------------------------+
|         Gymnasium RL Environment             |  <- gym_env.py
|   (observations, actions, rewards, info)     |
+---------------------------------------------+
|           Simulation Core                    |
|  +--------------+  +---------------------+  |
|  |  MPS Engine   |  |  State Vector Engine |  |  <- simulator.py / statevector_simulator.py
|  |  (<=4 qubits) |  |  (5-25 qubits, GPU) |  |
|  +--------------+  +---------------------+  |
+---------------------------------------------+
|            Physics Layer                     |
|  DFS Encoding - Noise Channels - Devices     |  <- physics/
+---------------------------------------------+
|          Backend (NumPy / CuPy)              |  <- Auto-detected at runtime
+---------------------------------------------+
```

**Layer 1 (Backend):** Provides array operations via NumPy (CPU) or CuPy (GPU). SiliQun detects CuPy at import time and transparently routes all linear algebra to the GPU when available.

**Layer 2 (Physics):** Contains the silicon-specific device profiles, DFS encoder, Hamiltonian construction, quantum gate definitions, and noise channel models. Each device profile encapsulates experimentally measured parameters (T1, T2*, charge noise power spectral density, exchange coupling strengths) from published literature.

**Layer 3 (Simulation Core):** Two interchangeable engines share a common API. The MPS engine uses adaptive-bond-dimension tensor networks for approximate simulation of small systems. The state vector engine operates in the DFS logical subspace (2^n dimensions instead of 2^3n) and leverages GPU-accelerated `einsum` contractions for exact simulation of systems up to 25 logical qubits.

**Layer 4 (Gymnasium Environment):** Wraps either engine in a standard Gymnasium interface, exposing continuous action spaces (gate parameters), structured observation spaces (state amplitudes, bond dimensions, noise estimates), and physics-informed reward signals (state fidelity, leakage penalties).

## Device Profiles

SiliQun includes three experimentally calibrated silicon spin qubit device profiles:

| Profile | T1 | T2* | Gate Fidelity | Source |
|---------|-----|------|---------------|--------|
| **Donor** (P in 28Si) | 30 s | 0.5 ms | 99.9% | Muhonen et al., Nature Nanotech. (2014) |
| **SiMOS** (MOS QD) | 1 s | 20 us | 99.6% | Yoneda et al., Nature Nanotech. (2018) |
| **GAA** (Gate-All-Around FET) | 10 ms | 1 us | 99.0% | Geyer et al. (2022) |

Grid topologies are available via convenience functions:

```python
from siliqun.physics.devices.profiles import (
    donor_2q,      # 2 qubits (linear)
    sledge_3x3,    # 9 qubits (3x3 grid)
    sledge_4x4,    # 16 qubits (4x4 grid)
    sledge_5x5     # 25 qubits (5x5 grid)
)
```

## Writing a Plugin

SiliQun v5 ships a formal **plugin interface** that lets you integrate any new
qubit platform in **under 50 lines of Python**, without modifying the core
framework.  The Lindblad solver, Gymnasium interface, and RL agent are all
provided — you only write the platform-specific physics.

### Plugin Contract (4 mandatory components)

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Profile class | `my_profile.py` | ~28 | `TechnologyProfile` subclass with 5 abstract methods |
| Calibration data | `my_calibration.json` | ~12 | Raw measurements with source DOIs |
| Validation benchmarks | `my_benchmarks.py` | ~7 | Three PGIRS Phase-1 benchmark calls |
| Gymnasium env ID | (in profile class) | 1 line | Unique `SiliQun-{Name}-v{N}` string |

### Step 1 — Subclass `TechnologyProfile`

```python
# my_plugin/my_profile.py  (~28 lines)
import math, json, pathlib
import numpy as np
from siliqun.core.abc import TechnologyProfile, CalibrationRecord

class MyProfile(TechnologyProfile):
    technology_name  = "MyTech-Nominal"
    gymnasium_env_id = "SiliQun-MyTech-v1"

    def device_parameters(self):
        return {"T1": 1e-3, "T2": 5e-4, "tau1": 10e-9, "tau2": 100e-9, "p2": 0.01, "n_qubits": 2}

    def drift_hamiltonian(self):
        return np.zeros((4, 4), dtype=complex)

    def control_hamiltonians(self):
        Sx = np.array([[0,1],[1,0]], dtype=complex)/2
        I  = np.eye(2, dtype=complex)
        return [np.kron(Sx,I), np.kron(I,Sx)]

    def noise_channels(self):
        p  = self.device_parameters()
        Sm = np.array([[0,1],[0,0]], dtype=complex)
        I  = np.eye(2, dtype=complex)
        return [math.sqrt(1/p["T1"])*np.kron(Sm,I), math.sqrt(1/p["T1"])*np.kron(I,Sm)]

    def gate_library(self):
        return ["Rx", "Ry", "Rz", "CNOT"]

    @property
    def calibration_record(self):
        data = json.loads((pathlib.Path(__file__).parent/"my_calibration.json").read_text())
        return CalibrationRecord(**data)
```

### Step 2 — Add calibration data

```json
{
  "source_doi": "10.1000/example.doi",
  "source_description": "Author et al., Journal, Year",
  "raw_values": {"T1_ms": 1.0, "T2_ms": 0.5, "F_2Q_percent": 99.0},
  "additional_sources": []
}
```

### Step 3 — Add validation benchmarks

```python
# my_plugin/my_benchmarks.py  (~7 lines)
from siliqun.core.abc import PGIRSValidator
from .my_profile import MyProfile

def run_pgirs_phase1(tolerance=0.02):
    return PGIRSValidator(MyProfile()).run(tolerance=tolerance)

def assert_pgirs_phase1(tolerance=0.02):
    results = run_pgirs_phase1(tolerance)
    failures = [k for k, v in results.items() if not v]
    assert not failures, f"Failed PGIRS Phase-1: {failures}"
```

### Step 4 — Register and use

```python
from my_plugin.my_profile import MyProfile
import gymnasium as gym

profile = MyProfile()
profile.register()                         # runs benchmarks; raises if any fail
env = gym.make("SiliQun-MyTech-v1")       # works with any Gymnasium-compatible RL library
```

### Reference implementation

The `plugins/siliqun-gaas/` directory contains the complete reference
implementation for GAA silicon spin qubits (Tanamoto & Ono 2025), demonstrating
the minimum 47-line contract.  Use it as a template for new platforms.

```
plugins/siliqun-gaas/
├── siliqun_gaas/
│   ├── __init__.py
│   ├── gaa_profile.py        # 28 lines — TechnologyProfile subclass
│   ├── gaa_calibration.json  # 12 lines — calibration data with DOIs
│   └── gaa_benchmarks.py     #  7 lines — PGIRS Phase-1 benchmarks
├── setup.py
└── README.md
```

---

## GPU Benchmarks

Measured on NVIDIA A100-PCIE-40GB (Aziz HPC, King Abdulaziz University):

| Grid | Qubits | Hilbert Dim | 1Q Gate (CPU) | 1Q Gate (GPU) | Speedup |
|------|--------|-------------|---------------|---------------|---------|
| 2x2 | 4 | 16 | 51 us | N/A | CPU faster |
| 3x3 | 9 | 512 | 61 us | 1.88 ms | CPU faster |
| 4x4 | 16 | 65,536 | 1.07 ms | 1.90 ms | ~1x |
| **5x5** | **25** | **33.5M** | **317 ms** | **10.6 ms** | **30x** |

For 2-qubit gates at 25 qubits, GPU achieves **89x speedup** (556 ms to 6.2 ms). For expectation values, GPU achieves **93x speedup** (123 ms to 1.3 ms).

The crossover point where GPU becomes faster than CPU is approximately 16 qubits, which aligns with the automatic backend selection threshold.

## Reproducing the Benchmarks

```bash
# CPU-only benchmark (all grid sizes up to 16 qubits)
python benchmarks/benchmark_sv.py

# Full GPU benchmark (requires CuPy + NVIDIA GPU)
python benchmarks/benchmark_sv.py --gpu
```

Results are saved to `benchmarks/sv_benchmark_results.json` for analysis.

## Running the Tests

```bash
# Run the full test suite (45 tests)
python -m pytest tests/ -v

# Run only the state vector simulator tests
python -m pytest tests/test_statevector_simulator.py -v
```

The test suite covers gate unitarity, noise channel correctness, DFS encoding fidelity, observable accuracy, multi-qubit entanglement, and backend auto-detection.

## API Reference

### StateVectorSimulator

The core simulation engine for exact logical-subspace simulation.

```python
StateVectorSimulator(config, use_gpu=True)
```

**Parameters:**
- `config` (dict): Device configuration from `profiles.py` containing grid dimensions, noise parameters, and coupling strengths.
- `use_gpu` (bool): Whether to use CuPy GPU arrays. Falls back to NumPy if CuPy is not available.

**Gate Methods:**

| Method | Description |
|--------|-------------|
| `apply_rx(qubit, theta)` | Single-qubit X rotation by angle theta |
| `apply_ry(qubit, theta)` | Single-qubit Y rotation by angle theta |
| `apply_rz(qubit, theta)` | Single-qubit Z rotation by angle theta |
| `apply_cnot(control, target)` | Controlled-NOT gate |
| `apply_cz(control, target)` | Controlled-Z gate |
| `apply_sqrt_swap(q1, q2)` | Square-root SWAP gate |
| `apply_exchange(q1, q2, J, t)` | Exchange interaction gate with coupling J and time t |

**Observable Methods:**

| Method | Description |
|--------|-------------|
| `expectation_z(qubit)` | Single-qubit Z expectation value |
| `expectation_zz(q1, q2)` | Two-qubit ZZ correlator |
| `fidelity(target_state)` | State fidelity against a target state |
| `entanglement_entropy(qubit)` | Von Neumann entanglement entropy across a bipartition |

**Properties:**

| Property | Description |
|----------|-------------|
| `psi` | Current state vector (NumPy or CuPy array) |
| `n_qubits` | Number of logical qubits |
| `backend_name` | `"cupy"` or `"numpy"` |
| `leakage` | Cumulative DFS leakage estimate |

### SiliQunSimulator (MPS Engine)

The tensor network engine for approximate simulation of small systems.

```python
SiliQunSimulator(device_profile, config)
```

Shares the same gate and observable API as `StateVectorSimulator`. Additionally provides:
- `state.bond_dims` — Current MPS bond dimensions
- Adaptive SVD truncation with configurable `max_bond_dim`

### make_siliqun_env

Factory function for creating Gymnasium-compatible environments.

```python
make_siliqun_env(device, target, sim_mode="auto", max_steps=100, **kwargs)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | str | Device profile name: `"donor_2q"`, `"sledge_3x3"`, `"sledge_4x4"`, `"sledge_5x5"` |
| `target` | str | Target quantum state: `"bell"`, `"ghz"`, `"w"`, `"random"`, `"cluster1d"` |
| `sim_mode` | str | Simulation backend: `"mps"`, `"sv"`, `"auto"` |
| `max_steps` | int | Maximum steps per episode |

## Project Structure

```
siliqun/
├── siliqun/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── simulator.py              # MPS tensor network engine
│   │   ├── statevector_simulator.py   # GPU state vector engine (1,123 lines)
│   │   └── gym_env.py                # Gymnasium environment wrapper
│   ├── physics/
│   │   ├── dfs_encoding.py           # DFS logical subspace encoder
│   │   ├── noise/
│   │   │   └── channels.py           # Noise channel models (T1, T2*, 1/f)
│   │   └── devices/
│   │       └── profiles.py           # Device profiles (Donor, SiMOS, GAA)
│   └── tensor/
│       └── mps.py                    # Matrix product state implementation
├── tests/
│   └── test_statevector_simulator.py  # Comprehensive test suite (45 tests)
├── benchmarks/
│   ├── benchmark_sv.py               # Performance benchmarking script
│   └── sv_benchmark_results.json     # Benchmark results
├── core/
│   ├── __init__.py
│   └── abc.py                        # TechnologyProfile ABC + PGIRSValidator
├── paper/
│   └── softwarex_siliqun_v2.tex      # SoftwareX manuscript
└── README.md
plugins/
└── siliqun-gaas/                     # Reference plugin (GAA, 47 lines)
    ├── siliqun_gaas/
    │   ├── gaa_profile.py
    │   ├── gaa_calibration.json
    │   └── gaa_benchmarks.py
    └── setup.py
```

## Integration with Research Frameworks

SiliQun serves as the foundational simulation layer for several research projects:

| Project | Integration Point |
|---------|-------------------|
| **QUASAR** | Provides the training environment for DRL agents learning scalable quantum control (2 to 25 qubits) |
| **SeQurAIty** | Models physical-layer adversarial attacks via charge noise injection for security evaluation |
| **DYNAMO** | Supplies the quantum environment for classical DRL convergence benchmarking |

## Citation

If you use SiliQun in your research, please cite:

```bibtex
@article{alshehri2026siliqun,
  title={SiliQun: A GPU-Accelerated Tensor-Network Gymnasium Environment for 
         Deep Reinforcement Learning-Based Control of Silicon Spin Qubits},
  author={Al-Shehri, Raad},
  journal={SoftwareX},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

The author acknowledges the use of the Aziz Supercomputer at King Abdulaziz University for computational benchmarking. This work is part of a PhD research programme at the Faculty of Computing and Information Technology (FCIT), King Abdulaziz University.
