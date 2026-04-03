# DFS Validation Study Results Summary

## Key Findings

### Test 1: Intra-Qubit Exchange (EXACT)
- J_12 (logical Z): max leakage = 7.77e-16 (machine epsilon)
- J_23 (logical X): max leakage = 3.33e-16 (machine epsilon)
- **Conclusion:** Intra-qubit exchanges are exactly subspace-preserving

### Test 2: Inter-Qubit Exchange Leakage
- Max leakage at theta=pi: 83.95%
- Average leakage: 41.98%
- **This is physically correct** — raw Heisenberg exchange between boundary spins of adjacent DFS-encoded qubits causes significant leakage

### Test 3: Perturbative Regime (theta < 0.3 rad)
- Leakage scaling exponent: **2.00** (exactly quadratic, as expected for perturbative regime)
- Max leakage at theta=0.3: 1.87%
- Min projected fidelity: 98.13%
- **Conclusion:** The perturbative leakage model is valid for small exchange angles

### Test 4: Fong-Wandzura CNOT
- High leakage (42-76%) — the simplified FW sequence in our code is not the optimized version
- This is expected: the real FW-CNOT uses 18 carefully calibrated pulses; our simplified version is for testing

### Test 5: Encoded SWAP (EXACT)
- All 4 basis states: leakage = 0.0000, fidelity = 1.0000
- SWAP correctly maps |01_L> → |10_L> and vice versa
- **Conclusion:** The 3-physical-SWAP decomposition is exactly subspace-preserving

### Uniq MCP Cross-Validation
- Qiskit simulator confirmed DFS |0_L> state preparation:
  - |011>: 65.7% (theory 66.7%) ✓
  - |101>: 17.1% (theory 16.7%) ✓
  - |110>: 17.2% (theory 16.7%) ✓
- J_12 exchange confirmed to preserve DFS subspace (only m_s=-1/2 states populated)

## Implications for SiliQun Paper
1. The logical subspace projection is **exact** for intra-qubit operations
2. Inter-qubit leakage follows **theta^2 scaling** — perturbative model is valid
3. The encoded SWAP is **exact** — key for SLEDGE grid routing
4. The FW-CNOT needs the full optimized pulse sequence for high fidelity
