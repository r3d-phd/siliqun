# SiliQun — Silicon Qubits Simulator

**Version 1.0** | A modular, tensor-network-based simulator for silicon spin qubit systems.

SiliQun is purpose-built for the PhD research project on quantum deep reinforcement learning (DRL) for silicon spin qubit control. It provides physically accurate simulation of donor (P:Si), SiMOS, and gate-all-around (GAA) qubit architectures using Matrix Product State (MPS) and Matrix Product Operator (MPO) representations, enabling scalable simulation beyond the 5-qubit barrier that limits full state-vector approaches.

## Architecture

SiliQun follows a 4-layer modular architecture:

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: HPC Backend                               │
│  ├── HPCRunner (PBS job generation, sweep scripts)   │
│  ├── CUDABackend (cuQuantum/cuTensorNet)            │
│  └── CheckpointConfig (fault-tolerant checkpointing)│
├─────────────────────────────────────────────────────┤
│  Layer 3: Simulation Engine                         │
│  ├── SiliQunSimulator (TEBD time evolution)         │
│  └── SiliQunEnv (Gymnasium wrapper for DRL)         │
├─────────────────────────────────────────────────────┤
│  Layer 2: Physics Models                            │
│  ├── Hamiltonians (Zeeman, exchange, SOC, HF)       │
│  ├── Quantum Gates (ESR, EDSR, √SWAP, CZ)          │
│  ├── Noise Channels (T1, T2, 1/f charge noise)     │
│  └── Device Profiles (Donor, SiMOS, GAA)            │
├─────────────────────────────────────────────────────┤
│  Layer 1: Tensor Network Engine                     │
│  ├── MPS (Matrix Product State)                     │
│  ├── MPO (Matrix Product Operator)                  │
│  └── Compute Backends (NumPy, JAX, CUDA)            │
└─────────────────────────────────────────────────────┘
```

## Key Features

- **Tensor Network Core**: MPS/MPO representations with adaptive bond dimension truncation via SVD, enabling simulation of 8+ qubit systems that would require 2^N memory with state vectors.
- **Physically Accurate Noise**: Correlated 1/f charge noise (the dominant decoherence mechanism in silicon), T1 relaxation, T2* dephasing, and depolarizing channels — all parameterized from experimental data.
- **Three Device Architectures**: Donor (P:Si), SiMOS (micromagnet), and GAA (gate-all-around) with architecture-specific Hamiltonians, native gate sets, and noise profiles.
- **Gymnasium Integration**: Drop-in replacement for QUASAR's `SiliconSpinEnv`, providing a physics-based environment for DRL training with configurable observation spaces, action spaces, and reward functions.
- **HPC Ready**: PBS job generation, multi-node sweep scripts, CUDA/cuQuantum backend for GPU acceleration on Aziz A100 nodes.

## Quick Start

```python
from siliqun.engine.simulator import SiliQunSimulator, SimConfig
from siliqun.physics.devices.profiles import get_device_profile

# Create a 4-qubit SiMOS device
device = get_device_profile("simos", n_qubits=4)

# Initialize simulator (noise-free for debugging)
sim = SiliQunSimulator(device, SimConfig(noise_enabled=False, max_bond_dim=32))
sim.reset()

# Apply gates
sim.apply_ry(3.14159 / 2, qubit=0)   # Hadamard-like rotation
sim.apply_cnot(control=0, target=1)    # Entangle qubits 0-1

# Measure observables
print(f"<Z_0> = {sim.expectation_z(0):.4f}")
print(f"Entanglement entropy = {sim.compute_entanglement_entropy(1):.4f}")
print(f"Bond dimensions: {sim.state.bond_dims}")
```

## DRL Integration (QUASAR)

```python
from siliqun.engine.gym_env import SiliQunEnv
from siliqun.engine.simulator import SimConfig

# Create Gymnasium environment for DRL training
env = SiliQunEnv(
    device="donor",
    n_qubits=2,
    target_state="bell",
    max_steps=200,
    config=SimConfig(noise_enabled=True, max_bond_dim=16),
    reward_type="dense",
)

# Standard Gymnasium loop
obs, info = env.reset(seed=42)
for step in range(200):
    action = env.action_space.sample()  # Replace with DRL policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final fidelity: {info['fidelity']:.4f}")
```

## HPC Deployment (Aziz)

```python
from siliqun.hpc import HPCRunner, PBSConfig

runner = HPCRunner(PBSConfig(
    queue="A100",
    nodes=1,
    gpus=1,
    walltime="24:00:00",
    job_name="siliqun_8q_benchmark",
))

# Generate sweep script for all 3 device types
configs = [
    {"device": "donor", "n_qubits": 8, "seed": 42},
    {"device": "simos", "n_qubits": 8, "seed": 42},
    {"device": "gaa", "n_qubits": 8, "seed": 42},
]
script = runner.generate_sweep_script("train.py", configs)
runner.write_and_submit(script, "sweep.pbs", dry_run=False)
```

## Package Structure

```
siliqun/
├── __init__.py
├── backend/
│   ├── __init__.py          # Backend factory (get_backend, set_backend)
│   ├── base.py              # Abstract Backend interface
│   ├── numpy_backend.py     # NumPy reference backend
│   ├── jax_backend.py       # JAX GPU backend
│   └── cuda_backend.py      # CUDA/cuQuantum backend
├── tensor/
│   ├── __init__.py
│   ├── tensor.py            # Core Tensor class
│   ├── mps.py               # Matrix Product State
│   └── mpo.py               # Matrix Product Operator
├── physics/
│   ├── __init__.py
│   ├── gates.py             # Quantum gates (Pauli, rotations, 2Q)
│   ├── hamiltonian.py       # Hamiltonian construction as MPO
│   ├── noise/
│   │   ├── __init__.py
│   │   └── channels.py      # Noise channels + 1/f charge noise
│   └── devices/
│       ├── __init__.py
│       └── profiles.py      # Device profiles (Donor, SiMOS, GAA)
├── engine/
│   ├── __init__.py
│   ├── simulator.py         # Core simulation engine (TEBD)
│   └── gym_env.py           # Gymnasium environment wrapper
├── hpc/
│   ├── __init__.py
│   └── runner.py            # HPC runner (PBS scripts, sweeps)
└── tests/
    └── test_integration.py  # 42-test integration suite
```

## Test Suite

```bash
cd siliqun && python3 tests/test_integration.py
```

Tests cover 7 categories across all 4 layers:
1. **Backend** (4 tests): NumPy ops, SVD, matrix exponential, backend switching
2. **Tensor Networks** (9 tests): MPS states (computational, GHZ, Bell, W, random), inner products, expectations, MPO
3. **Physics** (8 tests): Gate unitarity, rotations, CNOT, Hamiltonian MPO, device profiles, noise channels
4. **Simulator** (6 tests): Reset, single-qubit gates, entanglement, noise, snapshots, fidelity
5. **Gymnasium** (6 tests): Creation, reset, step, episodes, all devices, rendering
6. **HPC** (4 tests): PBS config, script generation, sweeps, dry run
7. **Cross-Layer** (5 tests): Full stack 2Q/4Q/noisy, 8Q scalability, Hamiltonian MPO

## Integration with PhD Research

| Project | Integration Point |
|---------|-------------------|
| **QUASAR** | `SiliQunEnv` replaces `SiliconSpinEnv` for physics-based DRL training beyond 5 qubits |
| **MOZAIQ** | Tensor network tracks entanglement across modular partitions for distributed quantum control |
| **SeQurAIty** | `ChargeNoiseGenerator` models physical-layer adversarial attacks for security validation |

## Dependencies

- **Required**: `numpy`, `scipy`, `gymnasium`
- **Optional**: `jax[cuda]` (GPU backend), `quimb` (advanced tensor network ops), `cuquantum` (NVIDIA GPU acceleration)

## License

Internal research tool — King Abdulaziz University, FCIT.

## Citation

If you use SiliQun in your research, please cite:

```bibtex
@software{siliqun2026,
  title={SiliQun: A Tensor-Network Simulator for Silicon Spin Qubit Systems},
  author={Al-Shehri, Raad},
  year={2026},
  institution={King Abdulaziz University}
}
```
