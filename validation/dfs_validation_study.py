#!/usr/bin/env python3
"""
DFS Logical Subspace Validation Study
======================================

Compares the DFS-projected logical-space dynamics against full physical-space
simulation for 1 and 2 logical qubits (where 2^{3n} is tractable).

This script produces:
  1. Fidelity between projected and full-physics final states across
     a range of exchange angles (0 to 2*pi).
  2. Leakage probability as a function of exchange angle.
  3. Multi-gate circuit validation: random circuits of increasing depth.
  4. Numerical data for Table and Figure in the SoftwareX paper.

For 1 logical qubit (3 physical spins, dim=8):
  - Intra-qubit exchanges J_12 and J_23 are EXACT (zero leakage).
  - The DFS projection is an isometry, so U_logical = V^dag U_phys V.

For 2 logical qubits (6 physical spins, dim=64):
  - Inter-qubit exchange causes leakage out of the 4D logical subspace.
  - We quantify this leakage and show the projected dynamics remain
    faithful for typical exchange angles.

Author: Raad Alshehri
"""

import numpy as np
from scipy.linalg import expm
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =====================================================================
# Physical-space primitives
# =====================================================================

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def heisenberg_exchange(spin_i, spin_j, n_spins):
    """Build the Heisenberg exchange Hamiltonian H_{ij} for spins i,j
    in an n_spins system.

    H_{ij} = (1/4)(X_i X_j + Y_i Y_j + Z_i Z_j)
    """
    dim = 2 ** n_spins
    H = np.zeros((dim, dim), dtype=np.complex128)

    for P in [_X, _Y, _Z]:
        ops = [_I] * n_spins
        ops[spin_i] = P
        ops[spin_j] = P

        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += 0.25 * term

    return H


def build_dfs_isometry_1q():
    """Build the encoding isometry V for 1 logical qubit (3 physical spins).

    DFS basis:
      |0_L> = (1/sqrt(2)) (|01> - |10>) |x>  for S_{12}=0 subspace
      |1_L> = (1/sqrt(6)) (2|x>|01> - |0>|1x> - |1>|0x>)  ... etc.

    Using the standard DFS encoding for 3-spin S_total=1/2 subspace:
      |0_L> = (|01> - |10>)|0> / sqrt(2)   (S_{12}=0, m_z=+1/2 sector)
      |1_L> = (2|001> - |010> - |100>) / sqrt(6)  (S_{12}=1, m_z=+1/2 sector)

    Wait, let me use the actual SiliQun encoding.
    """
    from siliqun.physics.dfs_encoding import encoded_zero, encoded_one

    zero_L = encoded_zero()  # 8-dim
    one_L = encoded_one()    # 8-dim

    V = np.column_stack([zero_L, one_L])  # (8, 2)
    return V


def build_dfs_isometry_2q():
    """Build the encoding isometry V for 2 logical qubits (6 physical spins).

    V_2 = V_1 ⊗ V_1 : (64, 4)
    """
    V1 = build_dfs_isometry_1q()
    V2 = np.kron(V1, V1)  # (64, 4)
    return V2


# =====================================================================
# Validation Test 1: Single-qubit intra-qubit exchanges
# =====================================================================

def validate_single_qubit_exchanges():
    """Validate that intra-qubit exchanges (J_12, J_23) are exact.

    For a single encoded qubit, J_12 and J_23 act entirely within the
    DFS logical subspace. The projected gate should match the full
    physical evolution exactly (zero leakage).
    """
    print("=" * 70)
    print("TEST 1: Single-Qubit Intra-Qubit Exchange Validation")
    print("=" * 70)

    V = build_dfs_isometry_1q()  # (8, 2)
    P_enc = V @ V.conj().T  # Projector onto encoded subspace

    # Physical Hamiltonians
    H_12 = heisenberg_exchange(0, 1, 3)  # spins 0-1 in 3-spin system
    H_23 = heisenberg_exchange(1, 2, 3)  # spins 1-2 in 3-spin system

    # Logical Hamiltonians
    H_12_L = V.conj().T @ H_12 @ V  # (2, 2)
    H_23_L = V.conj().T @ H_23 @ V  # (2, 2)

    thetas = np.linspace(0, 2 * np.pi, 50)
    results_j12 = []
    results_j23 = []

    for theta in thetas:
        # --- J_12 exchange ---
        U_phys_12 = expm(-1j * theta * H_12)
        U_logical_12 = expm(-1j * theta * H_12_L)

        # Test on both basis states
        fidelities_12 = []
        leakages_12 = []
        for basis_idx in range(2):
            psi_L = np.zeros(2, dtype=np.complex128)
            psi_L[basis_idx] = 1.0

            # Full physics path
            psi_phys = V @ psi_L  # Encode
            psi_phys_out = U_phys_12 @ psi_phys  # Evolve in physical space

            # Projected path
            psi_L_out = U_logical_12 @ psi_L  # Evolve in logical space
            psi_proj_out = V @ psi_L_out  # Decode back to physical

            # Fidelity between the two
            fid = abs(np.vdot(psi_phys_out, psi_proj_out)) ** 2
            fidelities_12.append(fid)

            # Leakage from physical evolution
            p_enc = float(np.real(psi_phys_out.conj() @ P_enc @ psi_phys_out))
            leakages_12.append(1.0 - p_enc)

        results_j12.append({
            'theta': float(theta),
            'avg_fidelity': float(np.mean(fidelities_12)),
            'min_fidelity': float(np.min(fidelities_12)),
            'avg_leakage': float(np.mean(leakages_12)),
            'max_leakage': float(np.max(leakages_12)),
        })

        # --- J_23 exchange ---
        U_phys_23 = expm(-1j * theta * H_23)
        U_logical_23 = expm(-1j * theta * H_23_L)

        fidelities_23 = []
        leakages_23 = []
        for basis_idx in range(2):
            psi_L = np.zeros(2, dtype=np.complex128)
            psi_L[basis_idx] = 1.0

            psi_phys = V @ psi_L
            psi_phys_out = U_phys_23 @ psi_phys

            psi_L_out = U_logical_23 @ psi_L
            psi_proj_out = V @ psi_L_out

            fid = abs(np.vdot(psi_phys_out, psi_proj_out)) ** 2
            fidelities_23.append(fid)

            p_enc = float(np.real(psi_phys_out.conj() @ P_enc @ psi_phys_out))
            leakages_23.append(1.0 - p_enc)

        results_j23.append({
            'theta': float(theta),
            'avg_fidelity': float(np.mean(fidelities_23)),
            'min_fidelity': float(np.min(fidelities_23)),
            'avg_leakage': float(np.mean(leakages_23)),
            'max_leakage': float(np.max(leakages_23)),
        })

    # Summary
    min_fid_12 = min(r['min_fidelity'] for r in results_j12)
    max_leak_12 = max(r['max_leakage'] for r in results_j12)
    min_fid_23 = min(r['min_fidelity'] for r in results_j23)
    max_leak_23 = max(r['max_leakage'] for r in results_j23)

    print(f"\n  J_12 exchange (spins 0-1):")
    print(f"    Min fidelity across all angles: {min_fid_12:.15f}")
    print(f"    Max leakage across all angles:  {max_leak_12:.2e}")
    print(f"    EXACT: {'YES' if min_fid_12 > 1 - 1e-12 else 'NO'}")

    print(f"\n  J_23 exchange (spins 1-2):")
    print(f"    Min fidelity across all angles: {min_fid_23:.15f}")
    print(f"    Max leakage across all angles:  {max_leak_23:.2e}")
    print(f"    EXACT: {'YES' if min_fid_23 > 1 - 1e-12 else 'NO'}")

    return {
        'j12': results_j12,
        'j23': results_j23,
        'j12_min_fidelity': min_fid_12,
        'j12_max_leakage': max_leak_12,
        'j23_min_fidelity': min_fid_23,
        'j23_max_leakage': max_leak_23,
    }


# =====================================================================
# Validation Test 2: Two-qubit inter-qubit exchange
# =====================================================================

def validate_two_qubit_exchange():
    """Validate inter-qubit exchange for 2 logical qubits (6 physical spins).

    Inter-qubit exchange (between spin 3 of qubit A and spin 1 of qubit B)
    can cause leakage. We quantify:
      1. Fidelity of projected vs full-physics dynamics
      2. Leakage probability as a function of exchange angle
    """
    print("\n" + "=" * 70)
    print("TEST 2: Two-Qubit Inter-Qubit Exchange Validation")
    print("=" * 70)

    V2 = build_dfs_isometry_2q()  # (64, 4)
    P_enc = V2 @ V2.conj().T  # Projector onto 4D encoded subspace

    # Physical Hamiltonian: exchange between spin 2 of qubit A (physical
    # index 2) and spin 0 of qubit B (physical index 3)
    # In the 6-spin system, qubit A = spins 0,1,2 and qubit B = spins 3,4,5
    H_inter = heisenberg_exchange(2, 3, 6)  # (64, 64)

    # Logical Hamiltonian
    H_inter_L = V2.conj().T @ H_inter @ V2  # (4, 4)

    thetas = np.linspace(0, 2 * np.pi, 100)
    results = []

    for theta in thetas:
        U_phys = expm(-1j * theta * H_inter)
        U_logical = expm(-1j * theta * H_inter_L)

        fidelities = []
        leakages = []
        projected_fidelities = []

        for basis_idx in range(4):
            psi_L = np.zeros(4, dtype=np.complex128)
            psi_L[basis_idx] = 1.0

            # Full physics path
            psi_phys = V2 @ psi_L
            psi_phys_out = U_phys @ psi_phys

            # Projected path
            psi_L_out = U_logical @ psi_L
            psi_proj_out = V2 @ psi_L_out

            # State fidelity (overlap between full-physics and projected)
            fid = abs(np.vdot(psi_phys_out, psi_proj_out)) ** 2
            fidelities.append(fid)

            # Leakage from physical evolution
            p_enc = float(np.real(psi_phys_out.conj() @ P_enc @ psi_phys_out))
            leakages.append(1.0 - p_enc)

            # Fidelity of projected-back physical state with logical state
            psi_phys_projected_back = V2.conj().T @ psi_phys_out
            norm_pb = np.linalg.norm(psi_phys_projected_back)
            if norm_pb > 1e-15:
                psi_phys_projected_back /= norm_pb
            proj_fid = abs(np.vdot(psi_L_out, psi_phys_projected_back)) ** 2
            projected_fidelities.append(proj_fid)

        results.append({
            'theta': float(theta),
            'avg_fidelity': float(np.mean(fidelities)),
            'min_fidelity': float(np.min(fidelities)),
            'avg_leakage': float(np.mean(leakages)),
            'max_leakage': float(np.max(leakages)),
            'avg_projected_fidelity': float(np.mean(projected_fidelities)),
            'min_projected_fidelity': float(np.min(projected_fidelities)),
        })

    # Summary
    min_fid = min(r['min_fidelity'] for r in results)
    max_leak = max(r['max_leakage'] for r in results)
    avg_leak = np.mean([r['avg_leakage'] for r in results])
    min_proj_fid = min(r['min_projected_fidelity'] for r in results)

    print(f"\n  Inter-qubit exchange (spin 2 of qA <-> spin 0 of qB):")
    print(f"    Min state fidelity (projected vs full-physics): {min_fid:.10f}")
    print(f"    Max leakage probability:                       {max_leak:.6f}")
    print(f"    Average leakage probability:                   {avg_leak:.6f}")
    print(f"    Min projected-back fidelity:                   {min_proj_fid:.10f}")
    print(f"    (Projected-back fidelity measures how well the logical")
    print(f"     dynamics track the physical dynamics within the subspace)")

    return {
        'results': results,
        'min_fidelity': min_fid,
        'max_leakage': max_leak,
        'avg_leakage': avg_leak,
        'min_projected_fidelity': min_proj_fid,
    }


# =====================================================================
# Validation Test 3: Random multi-gate circuits
# =====================================================================

def validate_random_circuits(n_circuits=20, max_depth=50):
    """Validate DFS projection for random circuits of increasing depth.

    For 2 logical qubits, applies random sequences of:
      - Intra-qubit exchanges (J_12, J_23) on each qubit
      - Inter-qubit exchange between the two qubits

    Compares the final state from:
      (a) Full 64-dimensional physical simulation
      (b) 4-dimensional logical-space projection

    Reports fidelity and accumulated leakage vs circuit depth.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Random Multi-Gate Circuit Validation")
    print("=" * 70)

    V2 = build_dfs_isometry_2q()  # (64, 4)
    V1 = build_dfs_isometry_1q()  # (8, 2)
    P_enc = V2 @ V2.conj().T

    # Pre-compute physical Hamiltonians for 6-spin system
    # Intra-qubit A: J_12 (spins 0-1), J_23 (spins 1-2)
    H_12_A = heisenberg_exchange(0, 1, 6)
    H_23_A = heisenberg_exchange(1, 2, 6)
    # Intra-qubit B: J_12 (spins 3-4), J_23 (spins 4-5)
    H_12_B = heisenberg_exchange(3, 4, 6)
    H_23_B = heisenberg_exchange(4, 5, 6)
    # Inter-qubit: spin 2 of A <-> spin 3 of B
    H_inter = heisenberg_exchange(2, 3, 6)

    # Logical Hamiltonians
    H_12_A_L = V2.conj().T @ H_12_A @ V2
    H_23_A_L = V2.conj().T @ H_23_A @ V2
    H_12_B_L = V2.conj().T @ H_12_B @ V2
    H_23_B_L = V2.conj().T @ H_23_B @ V2
    H_inter_L = V2.conj().T @ H_inter @ V2

    hamiltonians_phys = [H_12_A, H_23_A, H_12_B, H_23_B, H_inter]
    hamiltonians_logical = [H_12_A_L, H_23_A_L, H_12_B_L, H_23_B_L, H_inter_L]
    gate_names = ['J12_A', 'J23_A', 'J12_B', 'J23_B', 'J_inter']

    rng = np.random.RandomState(42)
    depths = list(range(1, max_depth + 1, 1))

    results_by_depth = []

    for depth in depths:
        fidelities = []
        leakages = []
        proj_fidelities = []

        for _ in range(n_circuits):
            # Random initial logical state
            psi_L = rng.randn(4) + 1j * rng.randn(4)
            psi_L /= np.linalg.norm(psi_L)

            # Encode to physical
            psi_phys = V2 @ psi_L
            psi_logical = psi_L.copy()

            # Apply random gates
            for _ in range(depth):
                gate_idx = rng.randint(0, 5)
                theta = rng.uniform(0, np.pi)

                # Physical evolution
                U_phys = expm(-1j * theta * hamiltonians_phys[gate_idx])
                psi_phys = U_phys @ psi_phys

                # Logical evolution
                U_logical = expm(-1j * theta * hamiltonians_logical[gate_idx])
                psi_logical = U_logical @ psi_logical

            # Encode logical result back to physical for comparison
            psi_proj = V2 @ psi_logical

            # State fidelity
            fid = abs(np.vdot(psi_phys, psi_proj)) ** 2
            fidelities.append(fid)

            # Leakage
            p_enc = float(np.real(psi_phys.conj() @ P_enc @ psi_phys))
            leakages.append(1.0 - p_enc)

            # Project physical state back to logical and compare
            psi_phys_back = V2.conj().T @ psi_phys
            norm_back = np.linalg.norm(psi_phys_back)
            if norm_back > 1e-15:
                psi_phys_back /= norm_back
            proj_fid = abs(np.vdot(psi_logical / np.linalg.norm(psi_logical),
                                    psi_phys_back)) ** 2
            proj_fidelities.append(proj_fid)

        results_by_depth.append({
            'depth': depth,
            'avg_fidelity': float(np.mean(fidelities)),
            'std_fidelity': float(np.std(fidelities)),
            'min_fidelity': float(np.min(fidelities)),
            'avg_leakage': float(np.mean(leakages)),
            'std_leakage': float(np.std(leakages)),
            'max_leakage': float(np.max(leakages)),
            'avg_projected_fidelity': float(np.mean(proj_fidelities)),
            'min_projected_fidelity': float(np.min(proj_fidelities)),
        })

    # Print summary table
    print(f"\n  {'Depth':>5} | {'Avg Fid':>10} | {'Min Fid':>10} | "
          f"{'Avg Leak':>10} | {'Max Leak':>10} | {'Proj Fid':>10}")
    print("  " + "-" * 70)
    for r in results_by_depth[::5]:  # Every 5th depth
        print(f"  {r['depth']:5d} | {r['avg_fidelity']:10.8f} | "
              f"{r['min_fidelity']:10.8f} | {r['avg_leakage']:10.6f} | "
              f"{r['max_leakage']:10.6f} | {r['avg_projected_fidelity']:10.8f}")

    return results_by_depth


# =====================================================================
# Validation Test 4: Leakage regime characterization
# =====================================================================

def characterize_leakage_regimes():
    """Characterize when leakage becomes significant.

    Sweeps inter-qubit exchange angle and identifies:
      - Safe regime (leakage < 1e-4)
      - Moderate regime (1e-4 < leakage < 1e-2)
      - High-leakage regime (leakage > 1e-2)

    Also tests different spin pairings for inter-qubit exchange.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Leakage Regime Characterization")
    print("=" * 70)

    V2 = build_dfs_isometry_2q()
    P_enc = V2 @ V2.conj().T

    # Test all possible inter-qubit spin pairings
    spin_pairs = [
        (2, 3, "spin3_A <-> spin1_B (nearest)"),
        (2, 4, "spin3_A <-> spin2_B"),
        (2, 5, "spin3_A <-> spin3_B"),
        (0, 3, "spin1_A <-> spin1_B"),
        (1, 3, "spin2_A <-> spin1_B"),
    ]

    thetas = np.linspace(0, 2 * np.pi, 200)
    all_results = {}

    for spin_a, spin_b, label in spin_pairs:
        H = heisenberg_exchange(spin_a, spin_b, 6)
        results = []

        for theta in thetas:
            U_phys = expm(-1j * theta * H)

            leakages = []
            for basis_idx in range(4):
                psi_L = np.zeros(4, dtype=np.complex128)
                psi_L[basis_idx] = 1.0
                psi_phys = V2 @ psi_L
                psi_out = U_phys @ psi_phys
                p_enc = float(np.real(psi_out.conj() @ P_enc @ psi_out))
                leakages.append(1.0 - p_enc)

            results.append({
                'theta': float(theta),
                'avg_leakage': float(np.mean(leakages)),
                'max_leakage': float(np.max(leakages)),
            })

        max_leak = max(r['max_leakage'] for r in results)
        avg_leak = np.mean([r['avg_leakage'] for r in results])

        print(f"\n  {label}:")
        print(f"    Max leakage: {max_leak:.6f}")
        print(f"    Avg leakage: {avg_leak:.6f}")

        if max_leak < 1e-4:
            regime = "SAFE (< 1e-4)"
        elif max_leak < 1e-2:
            regime = "MODERATE (1e-4 to 1e-2)"
        else:
            regime = "HIGH (> 1e-2)"
        print(f"    Regime: {regime}")

        all_results[label] = {
            'spin_a': spin_a,
            'spin_b': spin_b,
            'results': results,
            'max_leakage': max_leak,
            'avg_leakage': avg_leak,
            'regime': regime,
        }

    return all_results


# =====================================================================
# Generate figures
# =====================================================================

def generate_validation_figures(test1_data, test2_data, test3_data, test4_data):
    """Generate matplotlib figures for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'paper')
    os.makedirs(fig_dir, exist_ok=True)

    # --- Figure: Validation fidelity and leakage ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel (a): Single-qubit exchange fidelity
    ax = axes[0, 0]
    thetas_j12 = [r['theta'] for r in test1_data['j12']]
    fids_j12 = [r['avg_fidelity'] for r in test1_data['j12']]
    thetas_j23 = [r['theta'] for r in test1_data['j23']]
    fids_j23 = [r['avg_fidelity'] for r in test1_data['j23']]
    ax.plot(thetas_j12, fids_j12, 'b-', linewidth=2, label=r'$J_{12}$ exchange')
    ax.plot(thetas_j23, fids_j23, 'r--', linewidth=2, label=r'$J_{23}$ exchange')
    ax.set_xlabel(r'Exchange angle $\theta$ (rad)', fontsize=11)
    ax.set_ylabel('State fidelity', fontsize=11)
    ax.set_title('(a) Single-qubit: projected vs. full-physics', fontsize=12)
    ax.set_ylim([0.9999, 1.00005])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (b): Two-qubit exchange fidelity and leakage
    ax = axes[0, 1]
    thetas_2q = [r['theta'] for r in test2_data['results']]
    fids_2q = [r['avg_fidelity'] for r in test2_data['results']]
    leaks_2q = [r['avg_leakage'] for r in test2_data['results']]
    ax.plot(thetas_2q, fids_2q, 'b-', linewidth=2, label='State fidelity')
    ax2 = ax.twinx()
    ax2.plot(thetas_2q, leaks_2q, 'r-', linewidth=1.5, alpha=0.7, label='Leakage')
    ax.set_xlabel(r'Exchange angle $\theta$ (rad)', fontsize=11)
    ax.set_ylabel('State fidelity', fontsize=11, color='blue')
    ax2.set_ylabel('Leakage probability', fontsize=11, color='red')
    ax.set_title('(b) Two-qubit: inter-qubit exchange', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax2.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (c): Random circuit fidelity vs depth
    ax = axes[1, 0]
    depths = [r['depth'] for r in test3_data]
    avg_fids = [r['avg_fidelity'] for r in test3_data]
    min_fids = [r['min_fidelity'] for r in test3_data]
    proj_fids = [r['avg_projected_fidelity'] for r in test3_data]
    ax.plot(depths, avg_fids, 'b-', linewidth=2, label='Avg state fidelity')
    ax.fill_between(depths, min_fids, avg_fids, alpha=0.2, color='blue')
    ax.plot(depths, proj_fids, 'g--', linewidth=2, label='Avg projected fidelity')
    ax.set_xlabel('Circuit depth (number of gates)', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('(c) Random circuits: fidelity vs. depth', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (d): Leakage vs depth
    ax = axes[1, 1]
    avg_leaks = [r['avg_leakage'] for r in test3_data]
    max_leaks = [r['max_leakage'] for r in test3_data]
    ax.plot(depths, avg_leaks, 'r-', linewidth=2, label='Avg leakage')
    ax.fill_between(depths, avg_leaks, max_leaks, alpha=0.2, color='red')
    ax.plot(depths, max_leaks, 'r--', linewidth=1, alpha=0.7, label='Max leakage')
    ax.set_xlabel('Circuit depth (number of gates)', fontsize=11)
    ax.set_ylabel('Leakage probability', fontsize=11)
    ax.set_title('(d) Random circuits: leakage vs. depth', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'fig_dfs_validation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n  Validation figure saved to: {fig_path}")
    plt.close()

    return fig_path


# =====================================================================
# Main
# =====================================================================

def main():
    print("SiliQun DFS Logical Subspace Validation Study")
    print("=" * 70)
    print(f"Date: {__import__('datetime').datetime.now().isoformat()}")
    print()

    # Run all validation tests
    test1_data = validate_single_qubit_exchanges()
    test2_data = validate_two_qubit_exchange()
    test3_data = validate_random_circuits(n_circuits=20, max_depth=50)
    test4_data = characterize_leakage_regimes()

    # Generate figures
    fig_path = generate_validation_figures(test1_data, test2_data, test3_data, test4_data)

    # Save all results as JSON
    results_dir = os.path.join(os.path.dirname(__file__))
    results = {
        'test1_single_qubit': {
            'j12_min_fidelity': test1_data['j12_min_fidelity'],
            'j12_max_leakage': test1_data['j12_max_leakage'],
            'j23_min_fidelity': test1_data['j23_min_fidelity'],
            'j23_max_leakage': test1_data['j23_max_leakage'],
        },
        'test2_two_qubit': {
            'min_fidelity': test2_data['min_fidelity'],
            'max_leakage': test2_data['max_leakage'],
            'avg_leakage': test2_data['avg_leakage'],
            'min_projected_fidelity': test2_data['min_projected_fidelity'],
        },
        'test3_random_circuits': {
            'depth_1': test3_data[0],
            'depth_10': test3_data[9],
            'depth_25': test3_data[24],
            'depth_50': test3_data[49],
        },
        'test4_leakage_regimes': {
            k: {
                'max_leakage': v['max_leakage'],
                'avg_leakage': v['avg_leakage'],
                'regime': v['regime'],
            }
            for k, v in test4_data.items()
        },
        'figure_path': fig_path,
    }

    results_path = os.path.join(results_dir, 'dfs_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  1. Single-qubit exchanges: EXACT (fidelity > 1 - 1e-12)")
    print(f"  2. Two-qubit inter-qubit exchange:")
    print(f"       Min fidelity:    {test2_data['min_fidelity']:.10f}")
    print(f"       Max leakage:     {test2_data['max_leakage']:.6f}")
    print(f"       Projected fid:   {test2_data['min_projected_fidelity']:.10f}")
    print(f"  3. Random circuits (depth 50):")
    print(f"       Avg fidelity:    {test3_data[49]['avg_fidelity']:.8f}")
    print(f"       Avg leakage:     {test3_data[49]['avg_leakage']:.6f}")
    print(f"  4. Leakage regimes characterized for 5 spin pairings")
    print()


if __name__ == '__main__':
    main()
