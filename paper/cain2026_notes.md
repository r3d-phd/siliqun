# Paper Notes: Cain et al. (2026) - arXiv:2603.28627

## Title
Shor's algorithm is possible with as few as 10,000 reconfigurable atomic qubits

## Authors
Madelyn Cain, Qian Xu, Robbie King, Lewis R. B. Picard, Harry Levine, Manuel Endres, John Preskill, Hsin-Yuan Huang, Dolev Bluvstein

## Key Findings
- Shor's algorithm at cryptographically relevant scales with only 10,000 reconfigurable atomic (neutral-atom) qubits
- Leverages high-rate quantum error-correcting codes, efficient logical instruction sets, circuit design
- P-256 elliptic curve discrete logarithm: just a few days with 26,000 physical qubits
- RSA-2048 factoring: 1-2 orders of magnitude longer
- Neutral-atom experiments already demonstrated: universal fault-tolerant ops below threshold, hundreds of qubits, trapping arrays >6,000 qubits

## Platform
- **Neutral atoms** (NOT silicon spin qubits)
- Reconfigurable atomic arrays

## Relevance to SiliQun
This paper is about neutral atoms, not silicon. However, it's highly relevant because:
1. It dramatically lowers the qubit count needed for practical quantum computing (from millions to ~10K)
2. It creates competitive pressure on silicon spin qubits — silicon needs to show it can compete
3. It strengthens the MOTIVATION for SiliQun: if neutral atoms can do Shor's with 10K qubits, silicon needs better control/error mitigation tools (like DRL-based control via SiliQun) to stay competitive
4. It validates the importance of fault-tolerant quantum computing research broadly
5. Could be cited in the Motivation section to argue urgency of silicon spin qubit control optimization

## Detailed Technical Points
- High-rate error-correcting codes encode ~1000 logical qubits at ~30% encoding rate
- Order of magnitude reduction vs small quasi-local codes (~4% encoding rate)
- Two orders of magnitude reduction vs planar surface codes
- ECC-256 discrete log: ~10 days with 26K qubits (1ms cycle time)
- RSA-2048: 11K-14K qubits, ~2 orders of magnitude longer runtime
- Neutral atom arrays already demonstrated: 6,100 qubits trapped, 500-qubit fault-tolerant processing
- Key assumption: 1ms stabilizer measurement cycle time
- Authors from Oratomic (startup) + Caltech + UC Berkeley
- John Preskill is a co-author — very high credibility

## Strategic Relevance to SiliQun Paper

### Direct citation opportunity
This paper is a BOMBSHELL for the quantum computing community (March 30, 2026 — just 2 days ago!). It dramatically lowers the bar for practical quantum computing on neutral atoms. This creates:

1. **Competitive pressure narrative**: Silicon spin qubits must accelerate their control quality to remain competitive. SiliQun enables this by providing DRL-based control optimization.
2. **Urgency argument**: If neutral atoms can do Shor's with 10K qubits, the window for silicon to establish its niche is narrowing. Tools like SiliQun that accelerate silicon qubit control research become more urgent.
3. **Scale argument**: The paper shows ~10K physical qubits are enough. Silicon spin qubits need to demonstrate they can scale to this level with high fidelity — DRL control via SiliQun helps.

### How to cite in SiliQun paper
In the Motivation section, add a paragraph like:
"The urgency of developing high-fidelity control for silicon spin qubits has been further underscored by recent theoretical results demonstrating that practical quantum algorithms, including cryptographically relevant instances of Shor's algorithm, may be executable with as few as 10,000 physical qubits on competing neutral-atom platforms [Cain et al., 2026]. This dramatic reduction in resource estimates intensifies the need for silicon spin qubit platforms to achieve competitive gate fidelities through advanced control techniques, including those discovered by deep reinforcement learning."
