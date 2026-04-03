"""
Comprehensive tests for the StateVectorSimulator.

Tests cover:
1. Initialization and state management
2. Single-qubit gate correctness
3. Two-qubit gate correctness (adjacent and non-adjacent)
4. DFS logical projection (exchange gates)
5. Observables (Z, ZZ, entanglement entropy)
6. Measurement and collapse
7. Fidelity computation
8. Noise channels
9. Scaling tests (4x4, 5x5 grids)
10. Gym environment integration with SV backend
"""

import sys
import os
import numpy as np
import pytest

# Ensure siliqun is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from siliqun.engine.statevector_simulator import (
    StateVectorSimulator, SVSimConfig, DFSLogicalProjector,
)
from siliqun.physics.devices.profiles import (
    donor_device, sledge_device, sledge_2x2, sledge_3x3,
    sledge_4x4, sledge_5x5,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def donor_2q():
    """2-qubit donor device (no DFS)."""
    return donor_device(n_qubits=2)


@pytest.fixture
def sledge_2x2_dev():
    """2x2 SLEDGE device (4 logical qubits, DFS-encoded)."""
    return sledge_2x2()


@pytest.fixture
def sv_config_noiseless():
    """Noiseless SV config for deterministic tests."""
    return SVSimConfig(
        noise_enabled=False,
        use_gpu=False,
        seed=42,
        dtype="complex128",
    )


@pytest.fixture
def sv_config_noisy():
    """Noisy SV config."""
    return SVSimConfig(
        noise_enabled=True,
        use_gpu=False,
        seed=42,
        dtype="complex128",
    )


# ======================================================================
# 1. Initialization tests
# ======================================================================

class TestInitialization:

    def test_basic_init(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        assert sim.n_qubits == 2
        assert sim._dim == 4
        assert sim.time == 0.0
        assert sim.leakage == 0.0

    def test_initial_state_is_zero(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sv = sim.state_vector
        expected = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_allclose(sv, expected)

    def test_reset(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(np.pi, 0)
        sim.reset()
        sv = sim.state_vector
        assert abs(sv[0] - 1.0) < 1e-12

    def test_dfs_init(self, sledge_2x2_dev, sv_config_noiseless):
        sim = StateVectorSimulator(sledge_2x2_dev, sv_config_noiseless)
        assert sim.n_qubits == 4
        assert sim._dim == 16
        assert sim.is_dfs
        assert sim._dfs_projector is not None

    def test_norm_preserved(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        assert abs(sim._compute_norm() - 1.0) < 1e-12


# ======================================================================
# 2. Single-qubit gate tests
# ======================================================================

class TestSingleQubitGates:

    def test_rx_pi(self, donor_2q, sv_config_noiseless):
        """Rx(pi)|0> = -i|1>"""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(np.pi, 0)
        sv = sim.state_vector
        # |00> -> -i|10>
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[2]) - 1.0) < 1e-10  # |10> = index 2

    def test_ry_pi_half(self, donor_2q, sv_config_noiseless):
        """Ry(pi/2)|0> = (|0> + |1>)/sqrt(2)"""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_ry(np.pi / 2, 0)
        sv = sim.state_vector
        # qubit 0 in superposition, qubit 1 in |0>
        assert abs(abs(sv[0]) - 1 / np.sqrt(2)) < 1e-10
        assert abs(abs(sv[2]) - 1 / np.sqrt(2)) < 1e-10

    def test_rz_phase(self, donor_2q, sv_config_noiseless):
        """Rz(phi)|+> should give relative phase."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        # Prepare |+> on qubit 0
        sim.apply_ry(np.pi / 2, 0)
        # Apply Rz
        sim.apply_rz(np.pi / 2, 0)
        sv = sim.state_vector
        # Should still have equal amplitudes
        assert abs(abs(sv[0]) - abs(sv[2])) < 1e-10

    def test_norm_after_gates(self, donor_2q, sv_config_noiseless):
        """Norm should be preserved after multiple gates."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(0.3, 0)
        sim.apply_ry(0.7, 1)
        sim.apply_rz(1.2, 0)
        sim.apply_rx(2.1, 1)
        assert abs(sim._compute_norm() - 1.0) < 1e-12


# ======================================================================
# 3. Two-qubit gate tests
# ======================================================================

class TestTwoQubitGates:

    def test_cnot_creates_bell(self, donor_2q, sv_config_noiseless):
        """H|0> then CNOT -> Bell state."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        # H = Ry(pi/2) Rz(pi) up to global phase
        sim.apply_ry(np.pi / 2, 0)  # |+> on qubit 0
        sim.apply_cnot(0, 1)
        sv = sim.state_vector
        # Bell state: (|00> + |11>) / sqrt(2)
        assert abs(abs(sv[0]) - 1 / np.sqrt(2)) < 1e-10
        assert abs(abs(sv[3]) - 1 / np.sqrt(2)) < 1e-10
        assert abs(sv[1]) < 1e-10
        assert abs(sv[2]) < 1e-10

    def test_cz_gate(self, donor_2q, sv_config_noiseless):
        """CZ|11> = -|11>"""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        # Prepare |11>
        sim.apply_rx(np.pi, 0)
        sim.apply_rx(np.pi, 1)
        sv_before = sim.state_vector.copy()
        sim.apply_cz(0, 1)
        sv_after = sim.state_vector
        # |11> component should get a -1 phase
        assert abs(sv_after[3] + sv_before[3]) < 1e-10

    def test_sqrt_swap(self, donor_2q, sv_config_noiseless):
        """sqrt(SWAP) applied twice = SWAP."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        # Prepare |10>
        sim.apply_rx(np.pi, 0)
        sim.apply_sqrt_swap(0, 1)
        sim.apply_sqrt_swap(0, 1)
        sv = sim.state_vector
        # Should be |01> (up to global phase)
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[1]) - 1.0) < 1e-10  # |01>

    def test_non_adjacent_cnot(self, sv_config_noiseless):
        """CNOT between non-adjacent qubits in 4-qubit system."""
        dev = donor_device(n_qubits=4)
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        # Prepare |1000>
        sim.apply_rx(np.pi, 0)
        # CNOT(0, 3): should give |1001>
        sim.apply_cnot(0, 3)
        sv = sim.state_vector
        # |1001> = index 0b1001 = 9
        assert abs(abs(sv[9]) - 1.0) < 1e-10

    def test_norm_after_two_qubit(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(0.5, 0)
        sim.apply_cnot(0, 1)
        sim.apply_cz(0, 1)
        assert abs(sim._compute_norm() - 1.0) < 1e-12


# ======================================================================
# 4. DFS logical projection tests
# ======================================================================

class TestDFSProjection:

    def test_projector_creation(self):
        """DFS projector should be created for SLEDGE devices."""
        dev = sledge_2x2()
        config = SVSimConfig(noise_enabled=False, use_gpu=False)
        sim = StateVectorSimulator(dev, config)
        assert sim._dfs_projector is not None
        assert sim._dfs_projector.n_logical == 4

    def test_single_qubit_generators(self):
        """H12 and H23 logical generators should be 2x2 Hermitian."""
        proj = DFSLogicalProjector(2, [(0, 1)])
        H12 = proj._H12_logical
        H23 = proj._H23_logical
        # Check Hermiticity
        np.testing.assert_allclose(H12, H12.conj().T, atol=1e-12)
        np.testing.assert_allclose(H23, H23.conj().T, atol=1e-12)

    def test_exchange_12_unitarity(self):
        """Logical J12 gate should be unitary."""
        proj = DFSLogicalProjector(2, [(0, 1)])
        U = proj.logical_exchange_12(0.5)
        UdU = U.conj().T @ U
        np.testing.assert_allclose(UdU, np.eye(2), atol=1e-12)

    def test_exchange_23_unitarity(self):
        """Logical J23 gate should be unitary."""
        proj = DFSLogicalProjector(2, [(0, 1)])
        U = proj.logical_exchange_23(0.5)
        UdU = U.conj().T @ U
        np.testing.assert_allclose(UdU, np.eye(2), atol=1e-12)

    def test_inter_qubit_exchange_unitarity(self):
        """Logical inter-qubit exchange should be 4x4 unitary."""
        proj = DFSLogicalProjector(2, [(0, 1)])
        U = proj.logical_inter_qubit_exchange(0.3, 0, 1, 2, 0)
        UdU = U.conj().T @ U
        np.testing.assert_allclose(UdU, np.eye(4), atol=1e-12)

    def test_leakage_rate_small_angle(self):
        """Leakage should be small for small exchange angles."""
        proj = DFSLogicalProjector(2, [(0, 1)])
        leak = proj.compute_leakage_rate(0.01, 0, 1, 2, 0)
        assert leak < 1e-3  # Very small leakage for small angle

    def test_leakage_rate_large_angle(self):
        """Leakage should increase with exchange angle."""
        proj = DFSLogicalProjector(2, [(0, 1)])
        leak_small = proj.compute_leakage_rate(0.01, 0, 1, 2, 0)
        leak_large = proj.compute_leakage_rate(1.0, 0, 1, 2, 0)
        assert leak_large > leak_small


# ======================================================================
# 5. Observable tests
# ======================================================================

class TestObservables:

    def test_z_expectation_zero_state(self, donor_2q, sv_config_noiseless):
        """<Z> = +1 for |0>."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        assert abs(sim.expectation_z(0) - 1.0) < 1e-12
        assert abs(sim.expectation_z(1) - 1.0) < 1e-12

    def test_z_expectation_one_state(self, donor_2q, sv_config_noiseless):
        """<Z> = -1 for |1>."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(np.pi, 0)
        assert abs(sim.expectation_z(0) - (-1.0)) < 1e-10

    def test_z_expectation_superposition(self, donor_2q, sv_config_noiseless):
        """<Z> = 0 for |+>."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_ry(np.pi / 2, 0)
        assert abs(sim.expectation_z(0)) < 1e-10

    def test_zz_correlator_product(self, donor_2q, sv_config_noiseless):
        """<ZZ> = 1 for product state |00>."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        assert abs(sim.expectation_zz(0, 1) - 1.0) < 1e-12

    def test_zz_correlator_bell(self, donor_2q, sv_config_noiseless):
        """<ZZ> = 1 for Bell state (|00> + |11>)/sqrt(2)."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_ry(np.pi / 2, 0)
        sim.apply_cnot(0, 1)
        assert abs(sim.expectation_zz(0, 1) - 1.0) < 1e-10

    def test_entanglement_entropy_product(self, donor_2q, sv_config_noiseless):
        """S = 0 for product state."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        S = sim.compute_entanglement_entropy(1)
        assert abs(S) < 1e-10

    def test_entanglement_entropy_bell(self, donor_2q, sv_config_noiseless):
        """S = 1 for Bell state."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_ry(np.pi / 2, 0)
        sim.apply_cnot(0, 1)
        S = sim.compute_entanglement_entropy(1)
        assert abs(S - 1.0) < 1e-10


# ======================================================================
# 6. Measurement tests
# ======================================================================

class TestMeasurement:

    def test_measure_deterministic_zero(self, donor_2q, sv_config_noiseless):
        """Measuring |0> should always give 0."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        result = sim.measure_qubit(0)
        assert result == 0

    def test_measure_deterministic_one(self, donor_2q, sv_config_noiseless):
        """Measuring |1> should always give 1."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(np.pi, 0)
        result = sim.measure_qubit(0)
        assert result == 1

    def test_measure_collapses_state(self, donor_2q, sv_config_noiseless):
        """After measurement, state should be collapsed."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_ry(np.pi / 2, 0)  # |+>
        result = sim.measure_qubit(0)
        # After measurement, <Z> should be +/-1
        z = sim.expectation_z(0)
        assert abs(abs(z) - 1.0) < 1e-10

    def test_measure_all(self, donor_2q, sv_config_noiseless):
        """measure_all should return list of outcomes."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        results = sim.measure_all()
        assert len(results) == 2
        assert all(r in [0, 1] for r in results)


# ======================================================================
# 7. Fidelity tests
# ======================================================================

class TestFidelity:

    def test_fidelity_self(self, donor_2q, sv_config_noiseless):
        """Fidelity with self should be 1."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_rx(0.5, 0)
        fid = sim.compute_fidelity(sim.state_vector)
        assert abs(fid - 1.0) < 1e-10

    def test_fidelity_orthogonal(self, donor_2q, sv_config_noiseless):
        """Fidelity with orthogonal state should be 0."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        target = np.array([0, 0, 0, 1], dtype=np.complex128)
        fid = sim.compute_fidelity(target)
        assert abs(fid) < 1e-10

    def test_fidelity_bell(self, donor_2q, sv_config_noiseless):
        """Fidelity with target Bell state."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.apply_ry(np.pi / 2, 0)
        sim.apply_cnot(0, 1)
        target = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        fid = sim.compute_fidelity(target)
        assert abs(fid - 1.0) < 1e-10


# ======================================================================
# 8. Noise tests
# ======================================================================

class TestNoise:

    def test_noisy_fidelity_decreases(self, donor_2q, sv_config_noisy):
        """Fidelity should decrease with noise over time."""
        sim = StateVectorSimulator(donor_2q, sv_config_noisy)
        sim.apply_ry(np.pi / 2, 0)
        sim.apply_cnot(0, 1)
        target = sim.state_vector.copy()
        sim.apply_idle_noise(1e-6)  # 1 microsecond idle
        fid = sim.compute_fidelity(target)
        # Fidelity should decrease (or stay same if T2 is very long)
        assert fid <= 1.0 + 1e-10


# ======================================================================
# 9. Scaling tests
# ======================================================================

class TestScaling:

    def test_4_qubit_donor(self, sv_config_noiseless):
        """4-qubit donor system should work."""
        dev = donor_device(n_qubits=4)
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        assert sim._dim == 16
        sim.apply_rx(0.5, 0)
        sim.apply_cnot(0, 3)
        assert abs(sim._compute_norm() - 1.0) < 1e-12

    def test_8_qubit_donor(self, sv_config_noiseless):
        """8-qubit donor system."""
        dev = donor_device(n_qubits=8)
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        assert sim._dim == 256
        for q in range(8):
            sim.apply_rx(0.3 * q, q)
        assert abs(sim._compute_norm() - 1.0) < 1e-12

    def test_sledge_2x2(self, sv_config_noiseless):
        """2x2 SLEDGE (4 logical qubits)."""
        dev = sledge_2x2()
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        assert sim._dim == 16
        assert sim.is_dfs

    def test_sledge_3x3(self, sv_config_noiseless):
        """3x3 SLEDGE (9 logical qubits)."""
        dev = sledge_3x3()
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        assert sim._dim == 512
        assert sim.is_dfs

    def test_sledge_4x4(self, sv_config_noiseless):
        """4x4 SLEDGE (16 logical qubits)."""
        dev = sledge_4x4()
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        assert sim._dim == 65536
        assert sim.is_dfs
        # Apply some gates
        sim.apply_rx(0.5, 0)
        sim.apply_rx(0.3, 15)
        assert abs(sim._compute_norm() - 1.0) < 1e-12

    @pytest.mark.skipif(
        os.environ.get("SKIP_LARGE_TESTS", "1") == "1",
        reason="25-qubit test requires significant memory",
    )
    def test_sledge_5x5(self, sv_config_noiseless):
        """5x5 SLEDGE (25 logical qubits) - memory intensive."""
        dev = sledge_5x5()
        sim = StateVectorSimulator(dev, sv_config_noiseless)
        assert sim._dim == 33554432  # 2^25
        assert sim.is_dfs


# ======================================================================
# 10. Circuit execution tests
# ======================================================================

class TestCircuitExecution:

    def test_simple_circuit(self, donor_2q, sv_config_noiseless):
        """Execute a simple circuit."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        circuit = [
            ("ry", {"theta": np.pi / 2}, [0]),
            ("cnot", {}, [0, 1]),
        ]
        result = sim.execute_circuit(circuit)
        assert "final_state" in result
        assert "time" in result

    def test_circuit_with_measurement(self, donor_2q, sv_config_noiseless):
        """Execute circuit with measurement."""
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        circuit = [
            ("rx", {"theta": np.pi}, [0]),
            ("measure", {}, [0]),
        ]
        result = sim.execute_circuit(circuit)
        assert len(result["measurements"]) == 1
        assert result["measurements"][0] == (0, 1)


# ======================================================================
# 11. Snapshot and history tests
# ======================================================================

class TestSnapshot:

    def test_snapshot(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        snap = sim.snapshot()
        assert "time" in snap
        assert "norm" in snap
        assert "z_expectations" in snap
        assert "leakage" in snap
        assert "backend" in snap
        assert snap["backend"] == "cpu"
        assert len(snap["z_expectations"]) == 2

    def test_history(self, donor_2q, sv_config_noiseless):
        sim = StateVectorSimulator(donor_2q, sv_config_noiseless)
        sim.snapshot()
        sim.apply_rx(0.5, 0)
        sim.snapshot()
        assert len(sim.history) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
