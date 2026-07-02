"""
SiliQun MPO Density Matrix Simulator — Comprehensive Test Suite.

Validates the MPO-based mixed-state simulator against:
    1. Analytical results for known quantum states
    2. The MPS simulator (pure-state limit)
    3. Noise channel correctness (exact vs stochastic)
    4. Gymnasium environment integration in MPO mode
"""

import sys
import os
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# ═══════════════════════════════════════════════════════════════
# 1. MPO DENSITY MATRIX BASICS
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. MPO DENSITY MATRIX BASICS")
print("=" * 60)

from siliqun.engine.mpo_simulator import MPODensityMatrixSimulator, MPOSimConfig
from siliqun.engine.simulator import SiliQunSimulator, SimConfig
from siliqun.physics.devices.profiles import get_device_profile
from siliqun.tensor.mps import MPS
from siliqun.tensor.mpo import MPO
from siliqun.physics.gates import rx, ry, rz, cnot, cz, hadamard
from siliqun.physics.noise.channels import (
    amplitude_damping_kraus, phase_damping_kraus, depolarizing_kraus,
)


def test_mpo_init_pure_state():
    """Initial state should be |0...0><0...0| with Tr(rho) = 1."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-10, f"Tr(rho) = {tr}, expected 1.0"


def test_mpo_init_z_expectations():
    """Initial |00> state should have <Z> = +1 for all qubits."""
    dev = get_device_profile("donor", 3)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    for q in range(3):
        z = sim.expectation_z(q)
        assert abs(z - 1.0) < 1e-10, f"<Z_{q}> = {z}, expected 1.0"


def test_mpo_purity_pure_state():
    """Pure state should have purity Tr(rho^2) = 1."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    purity = sim.compute_purity()
    assert abs(purity - 1.0) < 1e-8, f"Purity = {purity}, expected 1.0"


def test_mpo_reset():
    """Reset should restore initial state."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim.apply_rx(np.pi, 0)
    sim.reset()
    z0 = sim.expectation_z(0)
    assert abs(z0 - 1.0) < 1e-10, f"After reset, <Z_0> = {z0}"


test("MPO init pure state Tr(rho)=1", test_mpo_init_pure_state)
test("MPO init Z expectations", test_mpo_init_z_expectations)
test("MPO purity of pure state", test_mpo_purity_pure_state)
test("MPO reset", test_mpo_reset)


# ═══════════════════════════════════════════════════════════════
# 2. SINGLE-QUBIT GATE TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("2. SINGLE-QUBIT GATE TESTS (MPO)")
print("=" * 60)


def test_mpo_rx_pi():
    """Rx(pi)|0> = i|1>, so <Z> should be -1."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(rx(np.pi), 0)
    z0 = sim.expectation_z(0)
    z1 = sim.expectation_z(1)
    assert abs(z0 - (-1.0)) < 1e-8, f"<Z_0> = {z0}, expected -1"
    assert abs(z1 - 1.0) < 1e-8, f"<Z_1> = {z1}, expected +1"
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-10, f"Tr(rho) = {tr}"


def test_mpo_ry_pi_half():
    """Ry(pi/2)|0> = (|0>+|1>)/sqrt(2), so <Z> = 0."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(ry(np.pi / 2), 0)
    z0 = sim.expectation_z(0)
    assert abs(z0) < 1e-8, f"<Z_0> = {z0}, expected 0"


def test_mpo_rz_preserves_z():
    """Rz(theta)|0> = e^{-i*theta/2}|0>, so <Z> = +1 still."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(rz(np.pi / 3), 0)
    z0 = sim.expectation_z(0)
    assert abs(z0 - 1.0) < 1e-8, f"<Z_0> = {z0}, expected 1"


def test_mpo_hadamard():
    """H|0> = |+>, so <Z> = 0."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    z0 = sim.expectation_z(0)
    assert abs(z0) < 1e-8, f"<Z_0> = {z0}, expected 0"
    purity = sim.compute_purity()
    assert abs(purity - 1.0) < 1e-8, f"Purity = {purity}, expected 1"


def test_mpo_trace_preserved_after_gates():
    """Trace should remain 1 after multiple gate applications."""
    dev = get_device_profile("donor", 3)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(rx(0.5), 0)
    sim._apply_single_gate_to_mpo(ry(1.2), 1)
    sim._apply_single_gate_to_mpo(rz(0.8), 2)
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-8, f"Tr(rho) = {tr}"


test("MPO Rx(pi) flips qubit", test_mpo_rx_pi)
test("MPO Ry(pi/2) superposition", test_mpo_ry_pi_half)
test("MPO Rz preserves Z", test_mpo_rz_preserves_z)
test("MPO Hadamard", test_mpo_hadamard)
test("MPO trace preserved after gates", test_mpo_trace_preserved_after_gates)


# ═══════════════════════════════════════════════════════════════
# 3. TWO-QUBIT GATE TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("3. TWO-QUBIT GATE TESTS (MPO)")
print("=" * 60)


def test_mpo_cnot_on_zero():
    """CNOT|00> = |00>, so <Z> = +1 for both."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)
    z0 = sim.expectation_z(0)
    z1 = sim.expectation_z(1)
    assert abs(z0 - 1.0) < 1e-6, f"<Z_0> = {z0}"
    assert abs(z1 - 1.0) < 1e-6, f"<Z_1> = {z1}"


def test_mpo_bell_state():
    """H|0> then CNOT creates Bell state: <Z_0>=0, <Z_1>=0, <Z_0 Z_1>=+1."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)
    z0 = sim.expectation_z(0)
    z1 = sim.expectation_z(1)
    zz = sim.expectation_zz(0, 1)
    assert abs(z0) < 1e-6, f"<Z_0> = {z0}, expected 0"
    assert abs(z1) < 1e-6, f"<Z_1> = {z1}, expected 0"
    assert abs(zz - 1.0) < 1e-6, f"<Z_0 Z_1> = {zz}, expected 1"
    purity = sim.compute_purity()
    assert abs(purity - 1.0) < 1e-6, f"Purity = {purity}"


def test_mpo_cnot_flip():
    """CNOT|10> = |11>: control=1, target flips."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(rx(np.pi), 0)  # |0> -> |1>
    sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)
    z0 = sim.expectation_z(0)
    z1 = sim.expectation_z(1)
    assert abs(z0 - (-1.0)) < 1e-6, f"<Z_0> = {z0}, expected -1"
    assert abs(z1 - (-1.0)) < 1e-6, f"<Z_1> = {z1}, expected -1"


def test_mpo_trace_after_two_qubit():
    """Trace should be preserved after two-qubit gates."""
    dev = get_device_profile("donor", 3)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)
    sim._apply_two_qubit_gate_to_mpo(cz(), 1, 2)
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-6, f"Tr(rho) = {tr}"


test("MPO CNOT on |00>", test_mpo_cnot_on_zero)
test("MPO Bell state creation", test_mpo_bell_state)
test("MPO CNOT flip |10>->|11>", test_mpo_cnot_flip)
test("MPO trace after two-qubit gates", test_mpo_trace_after_two_qubit)


# ═══════════════════════════════════════════════════════════════
# 4. NOISE CHANNEL TESTS (EXACT)
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("4. NOISE CHANNEL TESTS (EXACT)")
print("=" * 60)


def test_mpo_depolarizing_channel():
    """Depolarizing channel should reduce purity."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    # Start with pure state
    purity_before = sim.compute_purity()
    assert abs(purity_before - 1.0) < 1e-8

    # Apply strong depolarizing noise
    sim._apply_depolarizing_channel(0, p=0.3)
    purity_after = sim.compute_purity()
    assert purity_after < purity_before, \
        f"Purity should decrease: {purity_before} -> {purity_after}"
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-8, f"Tr(rho) = {tr}"


def test_mpo_amplitude_damping():
    """Amplitude damping on |1> should drive toward |0>."""
    dev = get_device_profile("donor", 1)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(rx(np.pi), 0)  # |0> -> |1>
    z_before = sim.expectation_z(0)
    assert abs(z_before - (-1.0)) < 1e-8

    # Apply amplitude damping with gamma = 0.5
    kraus = amplitude_damping_kraus(0.5)
    sim._apply_kraus_channel(0, kraus)
    z_after = sim.expectation_z(0)
    # Should be closer to +1 (|0>) than before
    assert z_after > z_before, f"<Z> should increase: {z_before} -> {z_after}"
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-8, f"Tr(rho) = {tr}"


def test_mpo_phase_damping():
    """Phase damping on |+> should reduce off-diagonal elements."""
    dev = get_device_profile("donor", 1)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(hadamard(), 0)  # |0> -> |+>
    purity_before = sim.compute_purity()

    # Apply phase damping
    kraus = phase_damping_kraus(0.5)
    sim._apply_kraus_channel(0, kraus)
    purity_after = sim.compute_purity()
    assert purity_after < purity_before, \
        f"Purity should decrease: {purity_before} -> {purity_after}"
    # Z expectation should remain 0 (phase damping doesn't change populations)
    z = sim.expectation_z(0)
    assert abs(z) < 1e-6, f"<Z> = {z}, should be 0 (populations unchanged)"
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-8, f"Tr(rho) = {tr}"


def test_mpo_full_depolarization():
    """Full depolarization (p=0.75) should give maximally mixed state."""
    dev = get_device_profile("donor", 1)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    # Apply full depolarizing channel
    sim._apply_depolarizing_channel(0, p=0.75)
    z = sim.expectation_z(0)
    # Maximally mixed: <Z> = 0
    assert abs(z) < 0.1, f"<Z> = {z}, expected ~0 for maximally mixed"
    purity = sim.compute_purity()
    # Maximally mixed: purity = 1/d = 0.5
    assert abs(purity - 0.5) < 0.1, f"Purity = {purity}, expected ~0.5"


def test_mpo_noise_preserves_trace():
    """All noise channels should preserve Tr(rho) = 1."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)

    # Apply various noise channels
    sim._apply_depolarizing_channel(0, 0.1)
    sim._apply_t1_channel(1, 1e-6, 1e-3)
    sim._apply_t2_channel(0, 1e-6, 1e-4)

    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-6, f"Tr(rho) = {tr}"


test("MPO depolarizing reduces purity", test_mpo_depolarizing_channel)
test("MPO amplitude damping drives to |0>", test_mpo_amplitude_damping)
test("MPO phase damping preserves populations", test_mpo_phase_damping)
test("MPO full depolarization -> mixed", test_mpo_full_depolarization)
test("MPO noise preserves trace", test_mpo_noise_preserves_trace)


# ═══════════════════════════════════════════════════════════════
# 5. MPS vs MPO COMPARISON (PURE STATE LIMIT)
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("5. MPS vs MPO COMPARISON (PURE STATE LIMIT)")
print("=" * 60)


def test_mps_vs_mpo_single_gates():
    """MPS and MPO should give same <Z> for noiseless single-qubit gates."""
    dev = get_device_profile("donor", 2)
    mps_sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False))
    mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))

    # Apply same gates
    for theta in [0.3, 0.7, 1.5, np.pi]:
        mps_sim.reset()
        mpo_sim.reset()
        mps_sim.apply_single_gate(rx(theta), 0)
        mpo_sim._apply_single_gate_to_mpo(rx(theta), 0)
        z_mps = mps_sim.expectation_z(0)
        z_mpo = mpo_sim.expectation_z(0)
        assert abs(z_mps - z_mpo) < 1e-6, \
            f"Rx({theta:.2f}): MPS <Z>={z_mps:.6f}, MPO <Z>={z_mpo:.6f}"


def test_mps_vs_mpo_bell_state():
    """MPS and MPO should give same correlators for Bell state."""
    dev = get_device_profile("donor", 2)
    mps_sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False))
    mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))

    # Create Bell state in both
    mps_sim.apply_single_gate(hadamard(), 0)
    mps_sim.apply_two_qubit_gate(cnot(), 0, 1)

    mpo_sim._apply_single_gate_to_mpo(hadamard(), 0)
    mpo_sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)

    for q in range(2):
        z_mps = mps_sim.expectation_z(q)
        z_mpo = mpo_sim.expectation_z(q)
        assert abs(z_mps - z_mpo) < 1e-6, \
            f"<Z_{q}>: MPS={z_mps:.6f}, MPO={z_mpo:.6f}"

    zz_mps = mps_sim.expectation_zz(0, 1)
    zz_mpo = mpo_sim.expectation_zz(0, 1)
    assert abs(zz_mps - zz_mpo) < 1e-6, \
        f"<ZZ>: MPS={zz_mps:.6f}, MPO={zz_mpo:.6f}"


def test_mps_vs_mpo_fidelity():
    """Fidelity computation should match between MPS and MPO."""
    dev = get_device_profile("donor", 2)
    mps_sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False))
    mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))

    # Apply same circuit
    mps_sim.apply_single_gate(ry(np.pi / 3), 0)
    mps_sim.apply_two_qubit_gate(cnot(), 0, 1)

    mpo_sim._apply_single_gate_to_mpo(ry(np.pi / 3), 0)
    mpo_sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)

    # Compute fidelity against |00>
    target = MPS.computational_basis(2, state=0)
    fid_mps = mps_sim.compute_fidelity(target)
    fid_mpo = mpo_sim.compute_fidelity(target)
    assert abs(fid_mps - fid_mpo) < 1e-4, \
        f"Fidelity: MPS={fid_mps:.6f}, MPO={fid_mpo:.6f}"


def test_mps_vs_mpo_random_circuit():
    """Random circuit should give matching results in both simulators."""
    dev = get_device_profile("donor", 3)
    mps_sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False, max_bond_dim=32))
    mpo_sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False, max_bond_dim=32))

    np.random.seed(42)
    for _ in range(10):
        gate_type = np.random.choice(["rx", "ry", "rz", "cnot"])
        if gate_type in ["rx", "ry", "rz"]:
            theta = np.random.uniform(-np.pi, np.pi)
            qubit = np.random.randint(3)
            gate = {"rx": rx, "ry": ry, "rz": rz}[gate_type](theta)
            mps_sim.apply_single_gate(gate, qubit)
            mpo_sim._apply_single_gate_to_mpo(gate, qubit)
        else:
            q = np.random.randint(2)
            mps_sim.apply_two_qubit_gate(cnot(), q, q + 1)
            mpo_sim._apply_two_qubit_gate_to_mpo(cnot(), q, q + 1)

    for q in range(3):
        z_mps = mps_sim.expectation_z(q)
        z_mpo = mpo_sim.expectation_z(q)
        assert abs(z_mps - z_mpo) < 0.01, \
            f"<Z_{q}>: MPS={z_mps:.6f}, MPO={z_mpo:.6f}"


test("MPS vs MPO single-qubit gates", test_mps_vs_mpo_single_gates)
test("MPS vs MPO Bell state", test_mps_vs_mpo_bell_state)
test("MPS vs MPO fidelity", test_mps_vs_mpo_fidelity)
test("MPS vs MPO random circuit", test_mps_vs_mpo_random_circuit)


# ═══════════════════════════════════════════════════════════════
# 6. GYMNASIUM ENVIRONMENT WITH MPO MODE
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("6. GYMNASIUM ENVIRONMENT WITH MPO MODE")
print("=" * 60)

from siliqun.engine.gym_env import SiliQunEnv, make_siliqun_env


def test_env_mpo_creation():
    """Create environment in MPO mode."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=False),
    )
    assert env.sim_mode == "mpo"
    assert env.observation_space.shape[0] > 0
    assert env.action_space.shape[0] > 0


def test_env_mpo_reset():
    """Reset in MPO mode should return valid obs."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=False),
    )
    obs, info = env.reset(seed=42)
    assert len(obs) > 0
    assert info["sim_mode"] == "mpo"


def test_env_mpo_step():
    """Step in MPO mode should work correctly."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        max_steps=10, sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=False),
    )
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert len(obs) > 0
    assert isinstance(reward, float)
    assert "fidelity" in info
    assert info["sim_mode"] == "mpo"


def test_env_mpo_episode():
    """Full episode in MPO mode."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        max_steps=20, sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=False),
        reward_type="dense",
    )
    obs, _ = env.reset(seed=42)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert info["step"] <= 20


def test_env_mpo_with_purity():
    """MPO mode with purity in observations."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        max_steps=10, sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=False),
        include_purity=True,
    )
    # Observation space should be 1 larger than MPS mode
    env_mps = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        max_steps=10, sim_mode="mps",
        config=SimConfig(noise_enabled=False),
    )
    assert env.observation_space.shape[0] == env_mps.observation_space.shape[0] + 1

    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, r, t, tr, info = env.step(action)
    assert "purity" in info


def test_env_mpo_render():
    """Render in MPO mode should include Tr(rho)."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        render_mode="ansi", sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=False),
    )
    env.reset()
    env.step(env.action_space.sample())
    output = env.render()
    assert "[MPO]" in output
    assert "Tr(rho)" in output


def test_env_mpo_noisy():
    """MPO mode with noise enabled."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        max_steps=10, sim_mode="mpo",
        config=MPOSimConfig(noise_enabled=True, max_bond_dim=16),
    )
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        obs, r, t, tr, info = env.step(env.action_space.sample())
    assert info["fidelity"] >= 0
    assert "trace" in info


def test_make_siliqun_env_mpo():
    """Factory function for MPO mode."""
    env = make_siliqun_env(
        n_qubits=2, device="donor", target="bell",
        sim_mode="mpo", noise=False, seed=42,
    )
    assert env.sim_mode == "mpo"
    obs, info = env.reset()
    assert len(obs) > 0


def test_make_siliqun_env_mps():
    """Factory function for MPS mode (backward compatible)."""
    env = make_siliqun_env(
        n_qubits=2, device="donor", target="bell",
        sim_mode="mps", noise=False, seed=42,
    )
    assert env.sim_mode == "mps"
    obs, info = env.reset()
    assert len(obs) > 0


test("Env MPO creation", test_env_mpo_creation)
test("Env MPO reset", test_env_mpo_reset)
test("Env MPO step", test_env_mpo_step)
test("Env MPO full episode", test_env_mpo_episode)
test("Env MPO with purity", test_env_mpo_with_purity)
test("Env MPO render", test_env_mpo_render)
test("Env MPO noisy", test_env_mpo_noisy)
test("make_siliqun_env MPO", test_make_siliqun_env_mpo)
test("make_siliqun_env MPS", test_make_siliqun_env_mps)


# ═══════════════════════════════════════════════════════════════
# 7. MEASUREMENT TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("7. MEASUREMENT TESTS (MPO)")
print("=" * 60)


def test_mpo_measurement_probabilities():
    """Measurement probabilities should match state."""
    dev = get_device_profile("donor", 1)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False, seed=42))
    # |0> state: P(0) = 1, P(1) = 0
    p0 = sim._compute_outcome_probability(0, 0)
    p1 = sim._compute_outcome_probability(0, 1)
    assert abs(p0 - 1.0) < 1e-8, f"P(0) = {p0}"
    assert abs(p1) < 1e-8, f"P(1) = {p1}"


def test_mpo_measurement_superposition():
    """Measurement of |+> should give P(0) = P(1) = 0.5."""
    dev = get_device_profile("donor", 1)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False, seed=42))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    p0 = sim._compute_outcome_probability(0, 0)
    p1 = sim._compute_outcome_probability(0, 1)
    assert abs(p0 - 0.5) < 1e-6, f"P(0) = {p0}, expected 0.5"
    assert abs(p1 - 0.5) < 1e-6, f"P(1) = {p1}, expected 0.5"


def test_mpo_measurement_collapse():
    """Measurement should collapse the state."""
    dev = get_device_profile("donor", 1)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False, seed=42))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    outcome = sim.measure_qubit(0)
    assert outcome in [0, 1]
    # After measurement, state should be pure (|0> or |1>)
    z = sim.expectation_z(0)
    expected_z = 1.0 if outcome == 0 else -1.0
    assert abs(z - expected_z) < 0.1, f"After measuring {outcome}, <Z> = {z}"


test("MPO measurement probabilities", test_mpo_measurement_probabilities)
test("MPO measurement superposition", test_mpo_measurement_superposition)
test("MPO measurement collapse", test_mpo_measurement_collapse)


# ═══════════════════════════════════════════════════════════════
# 8. SCALABILITY TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("8. SCALABILITY TESTS (MPO)")
print("=" * 60)


def test_mpo_3_qubits():
    """3-qubit GHZ-like state in MPO mode."""
    dev = get_device_profile("donor", 3)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    sim._apply_two_qubit_gate_to_mpo(cnot(), 0, 1)
    sim._apply_two_qubit_gate_to_mpo(cnot(), 1, 2)
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-6, f"Tr(rho) = {tr}"
    # All qubits should have <Z> = 0 in GHZ state
    for q in range(3):
        z = sim.expectation_z(q)
        assert abs(z) < 1e-6, f"<Z_{q}> = {z}"


def test_mpo_4_qubits():
    """4-qubit system in MPO mode."""
    dev = get_device_profile("simos", 4)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False, max_bond_dim=32))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    for q in range(3):
        sim._apply_two_qubit_gate_to_mpo(cnot(), q, q + 1)
    tr = sim.trace()
    assert abs(tr - 1.0) < 1e-4, f"Tr(rho) = {tr}"


def test_mpo_compression():
    """MPO compression should reduce bond dimensions."""
    dev = get_device_profile("donor", 4)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
        noise_enabled=False, max_bond_dim=64, compress_every=0,
    ))
    # Build up entanglement
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    for q in range(3):
        sim._apply_two_qubit_gate_to_mpo(cnot(), q, q + 1)
    # Apply noise to increase bond dimension
    sim._apply_depolarizing_channel(0, 0.1)
    sim._apply_depolarizing_channel(1, 0.1)

    bd_before = sim.state.max_bond_dim
    sim.state.compress(max_bond=8, cutoff=1e-10)
    bd_after = sim.state.max_bond_dim
    assert bd_after <= 8, f"Bond dim after compression: {bd_after}"
    # Trace should still be approximately 1
    tr = sim.trace()
    assert abs(tr - 1.0) < 0.1, f"Tr(rho) after compression = {tr}"


test("MPO 3-qubit GHZ", test_mpo_3_qubits)
test("MPO 4-qubit system", test_mpo_4_qubits)
test("MPO compression", test_mpo_compression)


# ═══════════════════════════════════════════════════════════════
# 9. CIRCUIT EXECUTION
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("9. CIRCUIT EXECUTION (MPO)")
print("=" * 60)


def test_mpo_execute_circuit():
    """Execute a circuit on the MPO simulator."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(noise_enabled=False))
    circuit = [
        ("ry", {"theta": np.pi / 2}, [0]),
        ("cnot", {}, [0, 1]),
    ]
    result = sim.execute_circuit(circuit)
    assert "final_state" in result
    assert "trace" in result
    assert abs(result["trace"] - 1.0) < 1e-6


def test_mpo_snapshot():
    """Snapshot should capture current state metrics."""
    dev = get_device_profile("donor", 2)
    sim = MPODensityMatrixSimulator(dev, MPOSimConfig(
        noise_enabled=False, track_purity=True,
    ))
    sim._apply_single_gate_to_mpo(hadamard(), 0)
    snap = sim.snapshot()
    assert "trace" in snap
    assert "bond_dims" in snap
    assert "z_expectations" in snap
    assert "purity" in snap
    assert abs(snap["purity"] - 1.0) < 1e-6


test("MPO execute circuit", test_mpo_execute_circuit)
test("MPO snapshot with purity", test_mpo_snapshot)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL MPO TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
print("=" * 60)
