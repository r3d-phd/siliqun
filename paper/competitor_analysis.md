# Competing Quantum Simulation Tools - Analysis for SiliQun Paper

## Key Competitors

### 1. Qiskit Aer (IBM)
- General-purpose quantum circuit simulator
- GPU support via cuStateVec (NVIDIA cuQuantum SDK)
- Statevector, density matrix, MPS backends
- NO silicon-specific device physics (T1, T2*, charge noise)
- NO DFS encoding
- NO Gymnasium integration
- Scales to ~32 qubits on single GPU (general circuits)
- Gangapuram et al. (2024) benchmarked at SciPost

### 2. QuTiP (Quantum Toolbox in Python)
- Open-source, widely used for quantum dynamics
- QuTiP 5 released Jan 2026 with GPU support
- Pulse-level simulation via qutip-qip
- Spin chain simulation supported
- NO silicon-specific device profiles
- NO DFS encoding
- NO Gymnasium/RL integration
- Scales to ~15-20 qubits (dense matrix)
- Li & Johansson (2013), Lambert et al. (2024)

### 3. NVIDIA cuQuantum / cuStateVec
- Low-level GPU library for state vector simulation
- Used by Qiskit Aer, Cirq, PennyLane as backend
- Extremely fast but NO physics layer
- NO device profiles, NO noise models, NO RL integration
- Pure computational backend
- Bayraktar et al. (2023), cited 130 times

### 4. Cirq (Google)
- General-purpose quantum circuit simulator
- Supports noisy simulation
- NO silicon-specific physics
- NO DFS encoding, NO Gymnasium integration

### 5. PennyLane (Xanadu)
- Differentiable quantum programming
- Good for variational quantum algorithms
- NO silicon spin qubit focus
- NO DFS encoding, NO RL environment

### 6. SQC Quantum Twins (Silicon Quantum Computing)
- Application-specific silicon quantum simulator
- Atomically precise silicon qubit arrays
- Commercial, NOT open-source
- Focused on materials simulation, NOT control/DRL
- Launched Feb 2026

### 7. Merino (2025) - UWaterloo thesis
- Simulation tool for silicon quantum dot design
- Device-level simulation (electrostatics, etc.)
- NOT for DRL training, NOT Gymnasium compatible

## SiliQun's Unique Position
SiliQun is the ONLY tool that combines:
1. Silicon-specific device physics (3 calibrated profiles)
2. DFS logical subspace encoding
3. GPU-accelerated state vector for 2D grids
4. Native Gymnasium RL environment
5. Dual MPS + SV backends with auto-selection
6. Perturbative leakage tracking
7. Open-source

## Comparison Table for Paper

| Feature | SiliQun | Qiskit Aer | QuTiP | cuQuantum | Cirq | PennyLane |
|---------|---------|------------|-------|-----------|------|-----------|
| Silicon device profiles | Yes (3) | No | No | No | No | No |
| DFS encoding | Yes | No | No | No | No | No |
| GPU state vector | Yes (CuPy) | Yes (cuSV) | Yes (v5) | Yes | Yes | Yes |
| MPS backend | Yes | Yes | No | No | No | No |
| Gymnasium RL env | Yes | No | No | No | No | No |
| Perturbative leakage | Yes | No | No | No | No | No |
| 2D grid topology | Yes | Generic | Generic | Generic | Generic | Generic |
| Max qubits (SV, 1 GPU) | 25 (logical) | ~32 | ~20 | ~32 | ~32 | ~30 |
| Open source | Yes | Yes | Yes | No (lib) | Yes | Yes |
| Charge noise model | Yes (1/f) | Generic | Generic | No | Generic | No |
