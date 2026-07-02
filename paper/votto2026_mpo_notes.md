# Votto et al. (2026) — Learning Mixed Quantum States in Large-Scale Experiments
## PRL 136, 090801 (2026), Published 4 March 2026

### Key Contribution
- Protocol to LEARN the MPO representation of an experimentally prepared quantum state
- Takes classical shadows (randomized measurements) as input
- Outputs tensors of an MPO that maximizes fidelity with the experimental state
- Tensor optimization is sequential (like DMRG)
- Provably efficient under short-range correlation conditions

### Key Results
- Experimentally demonstrated on IBM Brisbane superconducting processor
- Scaled to N = 96 qubits (previous randomized measurement approaches limited to N = 13)
- For N=96, D=1: F_max ~ 75% and ~50% respectively
- Per-qubit fidelity > 99% (F_max^{1/N} > 99%)
- Successfully captures experimental noise as mixed state
- Application: Quantum error mitigation via DMRG-QPCA (no additional experiments needed)

### MPO Representation
- State σ = Σ M^(1)_{s1,s1'} M^(2)_{s2,s2'} ... M^(N)_{sN,sN'} |{sj}><{sj'}|
- Bond dimension χ controls expressivity: χ = 2^N for exact, χ = O(1) for 1D states
- Tensors are χ_j × χ_{j+1} matrices
- Sequential optimization via DMRG-like sweeps (N_S = 20 sweeps)
- Uses geometric-mean (GM) fidelity, which is differentiable

### Relevance to SiliQun
1. SiliQun already uses MPS/MPO as core representation — this paper validates MPO as the RIGHT choice for representing quantum states in silicon
2. The paper shows MPO can represent MIXED states (density matrices), not just pure states — SiliQun currently uses MPS for pure states only
3. Key insight: MPO with χ = O(1) efficiently represents noisy 1D quantum states — this is EXACTLY what silicon spin qubit arrays produce
4. The 96-qubit scaling demonstrates MPO can go far beyond SiliQun's current 16-qubit benchmarks
5. Error mitigation via DMRG-QPCA could be integrated into SiliQun for noise-aware training

### Implications for SiliQun's 100-Qubit Goal
- MPO representation CAN scale to 96+ qubits for 1D systems (experimentally proven)
- Bond dimension χ = O(1) suffices for short-range correlated states (typical of silicon spin qubits)
- This validates SiliQun's architectural choice of tensor networks
- BUT: the paper works with 1D chains — 2D scaling still requires PEPS or other approaches

### Error Mitigation Application (DMRG-QPCA)
- Run DMRG on H = -σ to get principal component |ψ_0^σ⟩ as MPS
- No additional experiments needed — works purely on the learned MPO
- Fidelity between |ψ_0^σ⟩ and |ψ⟩ exceeds 90% even for large systems
- This is a form of noise-free state extraction from noisy data

### Conclusions and Outlook (from paper)
- MPO tomography can be combined with MPS preparation algorithms
- Modular quantum computing: fault-tolerant QC analyzes noisy state via MPO, then prepares purified MPS
- Quantum circuit cutting: save intermediate MPO, perform QPCA, resume algorithm
- Data deposited on Zenodo [106]

### Critical Insight for SiliQun
The paper proves that MPO is not just a computational convenience — it is the NATURAL representation for experimentally prepared quantum states. SiliQun's choice of MPS/MPO is now backed by a PRL-level experimental demonstration at 96 qubits.
