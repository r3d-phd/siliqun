# SiliQun Research Notes

## Tensor Network Libraries (Python)

### 1. quimb (Recommended Primary Backend)
- Fast Python library for "quantum information many-body" calculations
- Native tensor network support: MPS, MPO, PEPS, arbitrary TN
- GPU backend support via JAX, TensorFlow, PyTorch
- Circuit simulation via tensor networks built-in
- Contraction optimization (cotengra)
- Active development, well-documented
- Can interface with cuQuantum via QiboTN

### 2. TeNPy (Tensor Network Python) v1.0
- Mature library (2024 SciPost publication, 60+ citations)
- Focus on strongly correlated quantum systems
- MPS/MPO algorithms: DMRG, TEBD, TDVP
- Good for time evolution (critical for pulse simulation)
- Well-tested, production-quality

### 3. NVIDIA cuQuantum / cuTensorNet
- GPU-accelerated tensor network contractions
- State vector and tensor network simulation
- Python bindings available
- Optimal for HPC deployment on A100s

### 4. QiboTN
- Bridges Qibo quantum framework with tensor networks
- Uses both cuTensorNet and quimb as backends

## Design Decision: Use quimb as primary TN backend
- Reason: Best balance of features, GPU support, and quantum circuit integration
- cuQuantum can be used as an optional accelerator via quimb's backend system
- TeNPy's TEBD/TDVP algorithms can be referenced for time evolution

## Silicon Spin Qubit Physics

### Hamiltonian Components
1. **Zeeman splitting**: H_Z = g*μ_B*B_0*σ_z (external magnetic field)
2. **Exchange coupling**: H_ex = J(ε)*S_1·S_2 (voltage-tunable)
3. **Hyperfine interaction**: H_hf = A*I·S (donor qubits, 31P)
4. **Spin-orbit coupling**: H_SO (SiMOS, GAA devices)
5. **Valley splitting**: Δ_v (interface disorder)

### Noise Models
1. **1/f charge noise**: Power spectrum S(f) ∝ 1/f^α (α ≈ 0.7-1.3)
   - Causes detuning fluctuations → dephasing
   - Modeled as ensemble of two-level systems (TLS)
   - Kepa et al. 2023: realistic Si/SiGe simulation
2. **Johnson-Nyquist noise**: Thermal noise from control electronics
3. **Nuclear spin noise**: 29Si isotope fluctuations
4. **Phonon relaxation**: T1 processes

### SpinPulse (Jan 2026) - Key Competitor
- Open-source Python package by Quobly
- Pulse-level simulation of spin qubit quantum computers
- Models exchange coupling, Zeeman splitting, noise
- Limitations: Pure Python (slow), no GPU, no tensor networks for scaling
- SiliQun advantage: Rust/C++ backend, GPU, tensor networks, DRL integration

## Modular Architecture Design Principles
1. **Adapter pattern**: Abstract backends (numpy, jax, cuQuantum)
2. **Strategy pattern**: Swappable noise models, evolution methods
3. **Factory pattern**: Device variant creation (Donor, SiMOS, GAA)
4. **Observer pattern**: Monitoring hooks for DRL integration
5. **Clean separation**: Physics layer ↔ TN engine ↔ HPC backend
