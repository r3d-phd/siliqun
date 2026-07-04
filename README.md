# SiliQun v2.0 — Silicon Qubits Simulator

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org)

**SiliQun** is a modular, tensor-network-based Python library for simulating silicon spin qubit quantum computers. It provides physically accurate simulation of donor (P:Si), SiMOS, and gate-all-around (GAA) qubit architectures using Matrix Product State (MPS) and Matrix Product Operator (MPO) representations, enabling scalable simulation of noisy quantum systems beyond the limits of full state-vector methods.

SiliQun is the simulation backbone of the [QUASAR](https://github.com/ralshehri0468/quasar) project and is designed as a standalone library that can be used independently for quantum simulation, noise characterisation, and reinforcement learning research.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Tensor Network Core** | MPS/MPO representations with adaptive SVD truncation — simulates 8+ qubit systems that would require 2^N memory with state vectors |
| **Physically Accurate Noise** | Correlated 1/f charge noise, T1 relaxation, T2* dephasing, leakage, and crosstalk — calibrated to Weinstein et al., Nature 2023 |
| **Three Device Architectures** | Donor (P:Si), SiMOS (micromagnet), and GAA (gate-all-around) with architecture-specific Hamiltonians and native gate sets |
| **Gymnasium Integration** | Drop-in `SiliQunEnv` for DRL training with configurable observation spaces, action spaces, and reward functions |
| **Noise Curriculum** | Five-stage progressive noise curriculum for stable DRL convergence from near-ideal to realistic noise |
| **Target State Library** | Analytically exact Bell, GHZ, W, Cluster, and Dicke states with fidelity computation |
| **Plugin Architecture** | Standardised extension points for new device types, noise models, and third-party integrations (PennyLane, OpenPulse, QASM3) |
| **HPC Ready** | PBS job generation, multi-node sweep scripts, CUDA/cuQuantum backend for GPU acceleration |

---

## Installation

### Minimal (CPU only)

```bash
pip install siliqun
```

### With GPU support (JAX + CUDA)

```bash
pip install "siliqun[gpu]"
```

### Full installation (all extras)

```bash
pip install "siliqun[all]"
```

### From source

```bash
git clone https://github.com/ralshehri0468/siliqun.git
cd siliqun
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Gymnasium Environment for DRL Training

```python
from siliqun import SiliQunEnv

# Create a 3-qubit SiMOS environment targeting the GHZ state
# with stage-3 noise (medium noise, p=0.01)
env = SiliQunEnv(
    n_qubits=3,
    device="simos",
    target="GHZ",
    noise_stage=3,
    max_steps=200,
    seed=42,
)

obs, info = env.reset()
print(f"Observation dim: {env.observation_space.shape}")  # (obs_dim,)
print(f"Action dim:      {env.action_space.shape}")       # (act_dim,)

# Standard Gymnasium loop
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Best fidelity: {info['best_F']:.4f}")
```

### 2. Low-Level Simulator

```python
from siliqun import SiliQunSimulator, SimConfig
from siliqun.physics.devices import get_device_profile

# 4-qubit donor device with noise enabled
device = get_device_profile("donor", n_qubits=4)
sim = SiliQunSimulator(device, SimConfig(noise_enabled=True, max_bond_dim=32))
sim.reset()

# Apply gates
sim.apply_ry(1.5708, qubit=0)           # π/2 rotation
sim.apply_cnot(control=0, target=1)     # Entangle qubits 0-1
sim.apply_cnot(control=1, target=2)     # Extend entanglement
sim.apply_cnot(control=2, target=3)

# Measure fidelity to GHZ target
fidelity = sim.fidelity_to_target("GHZ")
print(f"GHZ fidelity: {fidelity:.4f}")
```

### 3. Target States and Fidelity

```python
from siliqun import build_target_state, compute_fidelity, ghz_state
import numpy as np

# Build target states directly
bell = build_target_state("Bell", n=2)
ghz3 = build_target_state("GHZ", n=3)
dicke = build_target_state("Dicke-k2", n=4)

# Compute fidelity between two state vectors
sv = ghz_state(3)
F = compute_fidelity(sv, "GHZ", n=3)
print(f"Self-fidelity: {F:.6f}")  # 1.000000
```

### 4. Noise Curriculum

```python
from siliqun import NoiseCurriculum, get_noise_prob

# Five-stage progressive curriculum
curriculum = NoiseCurriculum(
    initial_stage=1,
    advance_threshold=0.90,
    window=100,
)

# Training loop with automatic stage advancement
for episode in range(10000):
    # ... train agent ...
    episode_fidelity = 0.95  # from your DRL agent
    advanced = curriculum.step(episode_fidelity)
    if advanced:
        print(f"Advanced to stage {curriculum.current_stage}!")
        print(f"New noise prob: {curriculum.noise_prob:.4f}")

# Or use fixed stage for ablation studies
from siliqun import get_noise_prob
p_ns1 = get_noise_prob(1)  # 0.001
p_ns5 = get_noise_prob(5)  # 0.050
```

### 5. Noise Channels

```python
from siliqun import (
    NoiseParams, default_noise_params,
    apply_t1_noise, apply_dephasing_noise,
    depolarizing_kraus,
)
import numpy as np

# Get calibrated noise parameters for a SiMOS device
params = default_noise_params(n_qubits=3, device_type="simos")
print(f"T1 times: {params.t1_times}")
print(f"T2* times: {params.t2_star_times}")

# Apply depolarising noise as Kraus operators
p = 0.01
kraus_ops = depolarizing_kraus(p)
print(f"Number of Kraus operators: {len(kraus_ops)}")  # 4
```

---

## Architecture

SiliQun follows a five-layer modular architecture:

```
┌─────────────────────────────────────────────────────────┐
│               Layer 5: API & Integration                │
│  Gymnasium Env │ CLI │ REST API │ Visualization │ I/O   │
├─────────────────────────────────────────────────────────┤
│               Layer 4: Simulation Engine                │
│  SiliQunSimulator │ StatevectorSimulator │ MPOSimulator  │
├─────────────────────────────────────────────────────────┤
│               Layer 3: Physics Models                   │
│  Hamiltonians │ NoiseChannels │ DeviceProfiles │ Gates   │
├─────────────────────────────────────────────────────────┤
│               Layer 2: Tensor Network Engine            │
│  MPS │ MPO │ Contraction │ SVD Truncation               │
├─────────────────────────────────────────────────────────┤
│               Layer 1: Compute Backend                  │
│  NumPy │ JAX │ cuQuantum │ MPI                          │
└─────────────────────────────────────────────────────────┘
```

### Package Structure

```
siliqun/
├── __init__.py              # Public API
├── states.py                # Target state builders (Bell, GHZ, W, Cluster, Dicke)
├── noise_curriculum.py      # Five-stage noise curriculum controller
├── backend/                 # Layer 1: Compute backends
│   ├── base.py              # Abstract backend interface
│   ├── numpy_backend.py     # NumPy reference implementation
│   ├── jax_backend.py       # JAX GPU-accelerated backend
│   └── cuda_backend.py      # NVIDIA cuQuantum backend
├── tensor/                  # Layer 2: Tensor network engine
│   ├── tensor.py            # Core Tensor class
│   ├── mps.py               # Matrix Product State
│   └── mpo.py               # Matrix Product Operator
├── physics/                 # Layer 3: Physics models
│   ├── constants.py         # Physical constants (g-factors, μ_B, etc.)
│   ├── gates.py             # Quantum gates (standard + silicon-specific)
│   ├── hamiltonian.py       # Hamiltonian builder (Zeeman, exchange, SOC)
│   ├── dfs_encoding.py      # Decoherence-free subspace encoding
│   ├── sequential_pulsing.py # Sequential pulse scheduling
│   ├── devices/
│   │   └── profiles.py      # Device profiles (Donor, SiMOS, GAA)
│   └── noise/
│       └── channels.py      # Noise channels (T1, T2, 1/f, crosstalk)
├── engine/                  # Layer 4: Simulation engines
│   ├── simulator.py         # MPS-based simulator
│   ├── mpo_simulator.py     # MPO density matrix simulator
│   ├── statevector_simulator.py  # Full state-vector (≤12 qubits)
│   └── gym_env.py           # Gymnasium environment wrapper
├── compiler/                # QASM3 and gate compiler
│   ├── gate_compiler.py
│   └── qasm3_compiler.py
├── pulse/                   # Pulse-level simulation
│   ├── lindblad.py          # Lindblad master equation solver
│   └── openpulse_schedule.py
├── tomography/              # Quantum state and process tomography
│   └── tomography.py
├── plugins/                 # Third-party integrations
│   └── pennylane_device.py  # PennyLane device plugin
├── hpc/                     # HPC utilities
│   └── runner.py            # PBS job generation
└── api/                     # REST API server
    └── server.py
```

---

## Supported Device Architectures

| Device | Key Physics | Native Gates | Typical T1 | Typical T2* |
|--------|-------------|-------------|-----------|------------|
| `donor` | P:Si hyperfine + exchange | ESR, sqrtSWAP | ~1 s | ~1 ms |
| `simos` | SiMOS micromagnet + EDSR | EDSR, CZ | ~10 ms | ~100 μs |
| `gaa` | Gate-all-around SOC | EDSR, CZ | ~1 ms | ~10 μs |

---

## Plugin Interface

SiliQun supports custom extensions through a standardised plugin interface.

### Adding a Custom Device

```python
from siliqun.physics.devices.profiles import DeviceProfile, DEVICE_REGISTRY
from siliqun.physics.noise.channels import NoiseParams

@DEVICE_REGISTRY.register("my_device")
class MyDeviceProfile(DeviceProfile):
    name = "my_device"
    n_qubits_max = 8

    def get_noise_params(self, n_qubits: int) -> NoiseParams:
        return NoiseParams(
            t1_times=[50e-3] * n_qubits,
            t2_star_times=[5e-3] * n_qubits,
            charge_noise_amplitude=2e-6,
        )
```

### Adding a Custom Noise Model

```python
from siliqun.physics.noise.channels import NoiseParams
import numpy as np

def my_custom_noise(sv, qubit, n, params):
    """Apply custom noise channel."""
    # Return modified state vector
    return sv
```

---

## Citation

If you use SiliQun in your research, please cite:

```bibtex
@phdthesis{alshehri2025siliqun,
  author  = {Al-Shehri, Raad},
  title   = {{SiliQun}: A Modular Tensor-Network Simulator for Silicon
             Spin Qubit Quantum Computers},
  school  = {King Abdulaziz University},
  year    = {2025},
  note    = {Available at: https://github.com/ralshehri0468/siliqun}
}
```

---

## References

1. Weinstein, A. J. et al. (2023). Universal logic with encoded spin qubits in silicon. *Nature*, 615, 817–822. https://doi.org/10.1038/s41586-023-05777-3
2. Briegel, H. J. & Raussendorf, R. (2001). Persistent Entanglement in Arrays of Interacting Particles. *Physical Review Letters*, 86(5), 910–913.
3. Dicke, R. H. (1954). Coherence in Spontaneous Radiation Processes. *Physical Review*, 93(1), 99–110.
4. Orus, R. (2014). A practical introduction to tensor networks. *Annals of Physics*, 349, 117–158.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
