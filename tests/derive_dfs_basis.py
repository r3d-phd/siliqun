"""
Derive the correct DFS basis states for 3 electron spins.

The 8-dimensional Hilbert space of 3 spin-1/2 particles decomposes as:
    (1/2 ⊗ 1/2) ⊗ 1/2 = (0 ⊕ 1) ⊗ 1/2 = 1/2 ⊕ 1/2 ⊕ 3/2

The S=1/2 subspace is 4-dimensional (two doublets).
The m_s = -1/2 sector within S=1/2 has 2 states — these are our
encoded |0_L⟩ and |1_L⟩.

The S=3/2 subspace is 4-dimensional (one quartet).
The m_s = -1/2 sector within S=3/2 has 1 state — this is leakage.

Total m_s = -1/2 sector: 3 states (2 encoded + 1 leaked... wait,
actually 2 from S=1/2 and 1 from S=3/2, but there are also gauge states).

Let me derive everything from scratch using the total spin operator.
"""

import numpy as np
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def kron3(A, B, C):
    return np.kron(A, np.kron(B, C))

# Build total spin operators S_total = S_1 + S_2 + S_3
Sx_total = 0.5 * (kron3(X, I, I) + kron3(I, X, I) + kron3(I, I, X))
Sy_total = 0.5 * (kron3(Y, I, I) + kron3(I, Y, I) + kron3(I, I, Y))
Sz_total = 0.5 * (kron3(Z, I, I) + kron3(I, Z, I) + kron3(I, I, Z))

# S^2 = Sx^2 + Sy^2 + Sz^2
S2_total = Sx_total @ Sx_total + Sy_total @ Sy_total + Sz_total @ Sz_total

print("=== S^2 eigenvalues ===")
eigenvalues, eigenvectors = np.linalg.eigh(S2_total)
for i, (ev, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    S = (-1 + np.sqrt(1 + 4*ev)) / 2
    mz = float(np.real(vec.conj() @ Sz_total @ vec))
    print(f"  State {i}: S^2 = {ev:.4f} → S = {S:.2f}, m_s = {mz:.4f}")

print("\n=== m_s = -1/2 subspace ===")
# Find all states with m_s = -1/2
# The computational basis states with m_s = -1/2 are:
# |↑↓↓⟩ (index 3), |↓↑↓⟩ (index 5), |↓↓↑⟩ (index 6)
ms_minus_half_indices = []
for idx in range(8):
    bits = [(idx >> (2-j)) & 1 for j in range(3)]  # 0=up, 1=down
    mz = sum(-0.5 if b else 0.5 for b in bits)
    if abs(mz - (-0.5)) < 1e-10:
        label = ''.join('↓' if b else '↑' for b in bits)
        print(f"  Index {idx}: |{label}⟩, m_s = {mz}")
        ms_minus_half_indices.append(idx)

print(f"\n  m_s = -1/2 basis indices: {ms_minus_half_indices}")

# Extract the S^2 matrix in the m_s = -1/2 subspace
n_sub = len(ms_minus_half_indices)
S2_sub = np.zeros((n_sub, n_sub), dtype=np.complex128)
for i, ri in enumerate(ms_minus_half_indices):
    for j, rj in enumerate(ms_minus_half_indices):
        S2_sub[i, j] = S2_total[ri, rj]

print(f"\n=== S^2 in m_s = -1/2 subspace ({n_sub}×{n_sub}) ===")
print(np.round(np.real(S2_sub), 6))

sub_evals, sub_evecs = np.linalg.eigh(S2_sub)
print("\n=== Eigenstates of S^2 in m_s = -1/2 subspace ===")
for i, (ev, vec) in enumerate(zip(sub_evals, sub_evecs.T)):
    S = (-1 + np.sqrt(1 + 4*ev)) / 2
    # Expand to full 8-dim space
    full_vec = np.zeros(8, dtype=np.complex128)
    for k, idx in enumerate(ms_minus_half_indices):
        full_vec[idx] = vec[k]
    
    # Verify m_s
    mz = float(np.real(full_vec.conj() @ Sz_total @ full_vec))
    # Verify S^2
    s2_check = float(np.real(full_vec.conj() @ S2_total @ full_vec))
    
    print(f"\n  State {i}: S = {S:.4f}, S^2 = {s2_check:.4f}, m_s = {mz:.4f}")
    print(f"    Coefficients in {{|↑↓↓⟩, |↓↑↓⟩, |↓↓↑⟩}}:")
    print(f"    {np.round(np.real(vec), 8)}")
    
    # Check if it's S=1/2 or S=3/2
    if abs(S - 0.5) < 0.1:
        print(f"    → S=1/2 (ENCODED or GAUGE)")
    elif abs(S - 1.5) < 0.1:
        print(f"    → S=3/2 (LEAKAGE)")

# Now build the exchange operators
print("\n\n=== Exchange operators ===")

# Exchange H_12 = (1/2)(σ_1·σ_2 + I) = P_12 (SWAP operator for spins 1,2)
# Actually, Heisenberg exchange: H_12 = J(S_1·S_2) = (J/2)(P_12 - 1/2)
# The partial swap is: U_12(θ) = exp(-i θ H_12)

# S_1 · S_2
S1S2 = 0.25 * (kron3(X, X, I) + kron3(Y, Y, I) + kron3(Z, Z, I))
S2S3 = 0.25 * (kron3(I, X, X) + kron3(I, Y, Y) + kron3(I, Z, Z))

# Exchange unitary: U_12(θ) = exp(-i θ S_1·S_2)
from scipy.linalg import expm

print("\n=== Testing exchange J_12 on encoded states ===")
for theta in [0.1, np.pi/4, np.pi/2, np.pi]:
    U12 = expm(-1j * theta * S1S2)
    
    # Get the S=1/2 eigenstates
    s_half_states = []
    for i, (ev, vec) in enumerate(zip(sub_evals, sub_evecs.T)):
        S = (-1 + np.sqrt(1 + 4*ev)) / 2
        if abs(S - 0.5) < 0.1:
            full_vec = np.zeros(8, dtype=np.complex128)
            for k, idx in enumerate(ms_minus_half_indices):
                full_vec[idx] = vec[k]
            s_half_states.append(full_vec)
    
    # Test on each S=1/2 state
    for j, state in enumerate(s_half_states):
        evolved = U12 @ state
        # Project onto S=1/2 subspace
        P_half = sum(np.outer(s, s.conj()) for s in s_half_states)
        p_in = float(np.real(evolved.conj() @ P_half @ evolved))
        print(f"  θ={theta:.2f}, S=1/2 state {j}: P(S=1/2) = {p_in:.8f}")

print("\n=== Testing exchange J_23 on encoded states ===")
for theta in [0.1, np.pi/4, np.pi/2, np.pi]:
    U23 = expm(-1j * theta * S2S3)
    
    for j, state in enumerate(s_half_states):
        evolved = U23 @ state
        P_half = sum(np.outer(s, s.conj()) for s in s_half_states)
        p_in = float(np.real(evolved.conj() @ P_half @ evolved))
        print(f"  θ={theta:.2f}, S=1/2 state {j}: P(S=1/2) = {p_in:.8f}")

# Print the correct basis states
print("\n\n" + "=" * 60)
print("CORRECT DFS BASIS STATES")
print("=" * 60)
s_half_states_clean = []
for i, (ev, vec) in enumerate(zip(sub_evals, sub_evecs.T)):
    S = (-1 + np.sqrt(1 + 4*ev)) / 2
    if abs(S - 0.5) < 0.1:
        full_vec = np.zeros(8, dtype=np.complex128)
        for k, idx in enumerate(ms_minus_half_indices):
            full_vec[idx] = vec[k]
        s_half_states_clean.append((full_vec, vec))
        
        print(f"\nS=1/2 state {len(s_half_states_clean)-1}:")
        print(f"  Full 8-dim vector: {np.round(np.real(full_vec), 8)}")
        print(f"  Subspace vector:   {np.round(np.real(vec), 8)}")
        
        # Check orthogonality with our current |0_L⟩
        from siliqun.physics.dfs_encoding import encoded_zero, encoded_one, gauge_zero
        z = encoded_zero()
        o = encoded_one()
        g = gauge_zero()
        
        overlap_z = abs(full_vec.conj() @ z)
        overlap_o = abs(full_vec.conj() @ o)
        overlap_g = abs(full_vec.conj() @ g)
        print(f"  Overlap with current |0_L⟩: {overlap_z:.6f}")
        print(f"  Overlap with current |1_L⟩: {overlap_o:.6f}")
        print(f"  Overlap with current |g0⟩:  {overlap_g:.6f}")

# Also find the S=3/2 state in m_s=-1/2
for i, (ev, vec) in enumerate(zip(sub_evals, sub_evecs.T)):
    S = (-1 + np.sqrt(1 + 4*ev)) / 2
    if abs(S - 1.5) < 0.1:
        full_vec = np.zeros(8, dtype=np.complex128)
        for k, idx in enumerate(ms_minus_half_indices):
            full_vec[idx] = vec[k]
        print(f"\nS=3/2 state (LEAKAGE):")
        print(f"  Full 8-dim vector: {np.round(np.real(full_vec), 8)}")
        print(f"  Subspace vector:   {np.round(np.real(vec), 8)}")
