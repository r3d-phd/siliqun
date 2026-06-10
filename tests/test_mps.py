"""
Tests for siliqun.tensor.mps — covering all classmethods and operations.
Run with: python3 -m pytest tests/test_mps.py -v
"""
import pytest
import numpy as np
from math import comb, sqrt

from siliqun.tensor.mps import MPS


# ─── Helpers ────────────────────────────────────────────────────────────────

def fidelity_sv(mps: MPS, sv: np.ndarray) -> float:
    """Compute |<mps|sv>|^2 by converting MPS to dense."""
    psi = mps.to_dense().flatten()
    return float(abs(np.dot(psi.conj(), sv)) ** 2)


def hamming_weight(x: int, n: int) -> int:
    return bin(x).count("1")


# ─── Computational basis ─────────────────────────────────────────────────────

class TestComputationalBasis:
    def test_zero_state(self):
        mps = MPS.computational_basis(3, state=0)
        sv = mps.to_dense().flatten()
        assert abs(sv[0] - 1.0) < 1e-12
        assert np.allclose(sv[1:], 0)

    def test_arbitrary_state(self):
        mps = MPS.computational_basis(3, state=5)  # |101>
        sv = mps.to_dense().flatten()
        assert abs(sv[5] - 1.0) < 1e-12

    def test_norm_is_one(self):
        mps = MPS.computational_basis(4, state=7)
        assert abs(mps.norm() - 1.0) < 1e-12


# ─── GHZ state ───────────────────────────────────────────────────────────────

class TestGHZState:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_norm(self, n):
        mps = MPS.ghz_state(n)
        assert abs(mps.norm() - 1.0) < 1e-10

    def test_2qubit_amplitudes(self):
        mps = MPS.ghz_state(2)
        sv = mps.to_dense().flatten()
        assert abs(sv[0] - 1 / np.sqrt(2)) < 1e-10
        assert abs(sv[3] - 1 / np.sqrt(2)) < 1e-10
        assert abs(sv[1]) < 1e-10
        assert abs(sv[2]) < 1e-10

    def test_3qubit_amplitudes(self):
        mps = MPS.ghz_state(3)
        sv = mps.to_dense().flatten()
        assert abs(sv[0] - 1 / np.sqrt(2)) < 1e-10
        assert abs(sv[7] - 1 / np.sqrt(2)) < 1e-10
        assert all(abs(sv[i]) < 1e-10 for i in [1, 2, 3, 4, 5, 6])


# ─── Bell state ──────────────────────────────────────────────────────────────

class TestBellState:
    def test_bell_equals_ghz_2q(self):
        bell = MPS.bell_state(2)
        ghz = MPS.ghz_state(2)
        assert abs(abs(bell.inner(ghz)) - 1.0) < 1e-10

    def test_bell_norm(self):
        for n in [2, 3, 4]:
            assert abs(MPS.bell_state(n).norm() - 1.0) < 1e-10


# ─── W state ─────────────────────────────────────────────────────────────────

class TestWState:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_norm(self, n):
        mps = MPS.w_state(n)
        assert abs(mps.norm() - 1.0) < 1e-10

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_single_excitation_structure(self, n):
        mps = MPS.w_state(n)
        sv = mps.to_dense().flatten()
        # All amplitudes with Hamming weight != 1 should be ~0
        for i in range(2 ** n):
            hw = hamming_weight(i, n)
            if hw == 1:
                assert abs(abs(sv[i]) - 1 / sqrt(n)) < 1e-9, f"n={n}, i={i}: {sv[i]}"
            else:
                assert abs(sv[i]) < 1e-9, f"n={n}, i={i}: {sv[i]}"

    def test_w_is_dicke_k1(self):
        """W state should equal Dicke(n, k=1)."""
        for n in [3, 4, 5]:
            w = MPS.w_state(n)
            d = MPS.dicke_state(n, k=1)
            sv_w = w.to_dense().flatten()
            sv_d = d.to_dense().flatten()
            assert np.allclose(np.abs(sv_w), np.abs(sv_d), atol=1e-8), f"n={n}"


# ─── Dicke state ─────────────────────────────────────────────────────────────

class TestDickeState:
    @pytest.mark.parametrize("n,k", [
        (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2), (4, 3),
        (5, 2), (6, 3),
    ])
    def test_norm(self, n, k):
        mps = MPS.dicke_state(n, k)
        assert abs(mps.norm() - 1.0) < 1e-8, f"n={n}, k={k}: norm={mps.norm()}"

    @pytest.mark.parametrize("n,k", [
        (3, 1), (3, 2), (4, 2), (5, 2),
    ])
    def test_only_weight_k_amplitudes(self, n, k):
        mps = MPS.dicke_state(n, k)
        sv = mps.to_dense().flatten()
        n_states = comb(n, k)
        expected_amp = 1.0 / sqrt(n_states)
        for i in range(2 ** n):
            hw = hamming_weight(i, n)
            if hw == k:
                assert abs(abs(sv[i]) - expected_amp) < 1e-7, \
                    f"n={n}, k={k}, i={i}: |amp|={abs(sv[i]):.6f}, expected={expected_amp:.6f}"
            else:
                assert abs(sv[i]) < 1e-7, \
                    f"n={n}, k={k}, i={i}: |amp|={abs(sv[i]):.6f} should be 0"

    def test_k0_is_zero_state(self):
        mps = MPS.dicke_state(4, 0)
        sv = mps.to_dense().flatten()
        assert abs(sv[0] - 1.0) < 1e-10

    def test_kn_is_all_ones(self):
        n = 4
        mps = MPS.dicke_state(n, n)
        sv = mps.to_dense().flatten()
        assert abs(sv[-1] - 1.0) < 1e-10

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            MPS.dicke_state(3, 4)
        with pytest.raises(ValueError):
            MPS.dicke_state(3, -1)


# ─── Cluster linear state ────────────────────────────────────────────────────

class TestClusterLinearState:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
    def test_norm(self, n):
        mps = MPS.cluster_linear_state(n)
        assert abs(mps.norm() - 1.0) < 1e-9, f"n={n}: norm={mps.norm()}"

    def test_2qubit_matches_cluster1d(self):
        """For n=2, cluster_linear_state and cluster1d_state should be the same."""
        cl = MPS.cluster_linear_state(2)
        c1 = MPS.cluster1d_state(2)
        sv_cl = cl.to_dense().flatten()
        sv_c1 = c1.to_dense().flatten()
        assert np.allclose(np.abs(sv_cl), np.abs(sv_c1), atol=1e-9)

    def test_4qubit_matches_cluster1d(self):
        """For n=4, both methods should give the same state (up to global phase)."""
        cl = MPS.cluster_linear_state(4)
        c1 = MPS.cluster1d_state(4)
        overlap = abs(cl.inner(c1))
        assert abs(overlap - 1.0) < 1e-8, f"overlap={overlap}"

    def test_arbitrary_n_works(self):
        """cluster_linear_state must work for n=3, 5, 6, 7 (non-power-of-2)."""
        for n in [3, 5, 6, 7]:
            mps = MPS.cluster_linear_state(n)
            assert abs(mps.norm() - 1.0) < 1e-9

    def test_cluster1d_rejects_non_power2(self):
        """cluster1d_state must raise for n=3, 5, 6."""
        for n in [3, 5, 6, 7]:
            with pytest.raises(ValueError, match="power of 2"):
                MPS.cluster1d_state(n)


# ─── Cluster1D state (power-of-2 only) ───────────────────────────────────────

class TestCluster1DState:
    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_norm(self, n):
        mps = MPS.cluster1d_state(n)
        assert abs(mps.norm() - 1.0) < 1e-9

    def test_rejects_non_power2(self):
        with pytest.raises(ValueError):
            MPS.cluster1d_state(3)


# ─── Random state ────────────────────────────────────────────────────────────

class TestRandomState:
    def test_random_state_alias(self):
        """random_state() and random() should produce same-shaped MPS."""
        np.random.seed(42)
        mps1 = MPS.random_state(4, bond_dim=4)
        assert abs(mps1.norm() - 1.0) < 1e-10

    def test_random_norm(self):
        for n in [2, 3, 4, 5]:
            mps = MPS.random(n, bond_dim=4)
            assert abs(mps.norm() - 1.0) < 1e-10


# ─── from_dense ──────────────────────────────────────────────────────────────

class TestFromDense:
    def test_round_trip_ghz(self):
        mps = MPS.ghz_state(4)
        sv = mps.to_dense().flatten()
        mps2 = MPS.from_dense(sv, 4)
        assert abs(abs(mps.inner(mps2)) - 1.0) < 1e-9

    def test_round_trip_w(self):
        mps = MPS.w_state(4)
        sv = mps.to_dense().flatten()
        mps2 = MPS.from_dense(sv, 4)
        assert abs(abs(mps.inner(mps2)) - 1.0) < 1e-9

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            MPS.from_dense(np.ones(5), 3)


# ─── Operations ──────────────────────────────────────────────────────────────

class TestOperations:
    def test_inner_self_is_one(self):
        mps = MPS.ghz_state(4)
        assert abs(mps.inner(mps) - 1.0) < 1e-10

    def test_inner_orthogonal(self):
        """GHZ and W states are orthogonal for n>=3."""
        ghz = MPS.ghz_state(3)
        w = MPS.w_state(3)
        assert abs(ghz.inner(w)) < 1e-10

    def test_norm_consistency(self):
        mps = MPS.w_state(5)
        assert abs(mps.norm() - 1.0) < 1e-10

    def test_expectation_local_z(self):
        """<Z_0> for |0> should be +1."""
        mps = MPS.computational_basis(3, state=0)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        val = mps.expectation_local(Z, site=0)
        assert abs(val - 1.0) < 1e-10

    def test_expectation_local_z_excited(self):
        """<Z_0> for |100> should be -1."""
        mps = MPS.computational_basis(3, state=4)  # |100>
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        val = mps.expectation_local(Z, site=0)
        assert abs(val - (-1.0)) < 1e-10

    def test_copy_independence(self):
        mps = MPS.ghz_state(3)
        mps2 = mps.copy()
        mps2[0][:] = 0
        assert abs(mps.norm() - 1.0) < 1e-10  # original unchanged


# ─── TEBD operations ─────────────────────────────────────────────────────────

class TestTEBD:
    def test_apply_one_site_gate_identity(self):
        mps = MPS.ghz_state(3)
        sv_before = mps.to_dense().flatten().copy()
        I = np.eye(2, dtype=complex)
        mps.apply_one_site_gate(I, site=0)
        sv_after = mps.to_dense().flatten()
        assert np.allclose(sv_before, sv_after, atol=1e-12)

    def test_apply_one_site_gate_x(self):
        """Applying X to qubit 0 of |000> should give |100>."""
        mps = MPS.computational_basis(3, state=0)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        mps.apply_one_site_gate(X, site=0)
        sv = mps.to_dense().flatten()
        assert abs(sv[4] - 1.0) < 1e-10  # |100> = index 4

    def test_apply_two_site_gate_cnot(self):
        """CNOT on |10> should give |11>."""
        mps = MPS.computational_basis(2, state=2)  # |10>
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        mps.apply_two_site_gate(CNOT, site=0)
        sv = mps.to_dense().flatten()
        assert abs(sv[3] - 1.0) < 1e-10  # |11> = index 3

    def test_apply_two_site_gate_identity(self):
        mps = MPS.w_state(4)
        sv_before = mps.to_dense().flatten().copy()
        I4 = np.eye(4, dtype=complex)
        mps.apply_two_site_gate(I4, site=1)
        sv_after = mps.to_dense().flatten()
        assert np.allclose(sv_before, sv_after, atol=1e-12)

    def test_tebd_sweep_identity(self):
        mps = MPS.ghz_state(4)
        sv_before = mps.to_dense().flatten().copy()
        I4 = np.eye(4, dtype=complex)
        gates = [(i, I4) for i in range(3)]
        mps.tebd_sweep(gates, max_bond_dim=8)
        sv_after = mps.to_dense().flatten()
        assert np.allclose(sv_before, sv_after, atol=1e-12)

    def test_apply_two_site_gate_invalid_site(self):
        mps = MPS.ghz_state(3)
        with pytest.raises(ValueError):
            mps.apply_two_site_gate(np.eye(4), site=2)  # site 2 is last, no site+1


# ─── Repr ────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr(self):
        mps = MPS.ghz_state(3)
        r = repr(mps)
        assert "MPS" in r
        assert "n_sites=3" in r
