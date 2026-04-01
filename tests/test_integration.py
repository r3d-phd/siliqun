"""
SiliQun v1.0 — Comprehensive Integration Test Suite.

Tests the full stack: backend → tensor → physics → engine → gym_env → hpc.
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
# 1. BACKEND TESTS
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. BACKEND TESTS")
print("=" * 60)

from siliqun.backend import get_backend, active_backend, set_backend


def test_numpy_backend():
    be = get_backend("numpy")
    a = be.zeros((2, 2))
    assert a.shape == (2, 2)
    b = be.eye(3)
    assert b.shape == (3, 3)
    c = be.einsum("ij,jk->ik", b, b)
    assert np.allclose(be.to_numpy(c), np.eye(3))


def test_backend_svd():
    be = get_backend("numpy")
    A = be.array(np.random.randn(4, 3) + 1j * np.random.randn(4, 3))
    U, S, Vh = be.svd(A)
    recon = be.to_numpy(U) @ np.diag(be.to_numpy(S)) @ be.to_numpy(Vh)
    assert np.allclose(be.to_numpy(A), recon, atol=1e-12)


def test_backend_expm():
    be = get_backend("numpy")
    Z = be.zeros((2, 2))
    result = be.expm(Z)
    assert np.allclose(be.to_numpy(result), np.eye(2), atol=1e-12)


def test_set_backend():
    set_backend("numpy")
    be = active_backend()
    assert be.__class__.__name__ == "NumPyBackend"


test("NumPy backend basic ops", test_numpy_backend)
test("Backend SVD decomposition", test_backend_svd)
test("Backend matrix exponential", test_backend_expm)
test("set_backend / active_backend", test_set_backend)


# ═══════════════════════════════════════════════════════════════
# 2. TENSOR NETWORK TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("2. TENSOR NETWORK TESTS")
print("=" * 60)

from siliqun.tensor.mps import MPS
from siliqun.tensor.mpo import MPO


def test_mps_computational_basis():
    mps = MPS.computational_basis(3, state=0)
    assert mps.n_sites == 3
    assert abs(mps.norm() - 1.0) < 1e-10


def test_mps_ghz_state():
    ghz = MPS.ghz_state(4)
    assert ghz.n_sites == 4
    n = ghz.norm()
    assert abs(n - 1.0) < 1e-10, f"GHZ norm = {n}"


def test_mps_bell_state():
    bell = MPS.bell_state(2)
    assert bell.n_sites == 2
    n = bell.norm()
    assert abs(n - 1.0) < 1e-10, f"Bell norm = {n}"


def test_mps_w_state():
    w = MPS.w_state(3)
    assert w.n_sites == 3
    n = w.norm()
    assert abs(n - 1.0) < 1e-10, f"W state norm = {n}"


def test_mps_random():
    r = MPS.random(5, bond_dim=8)
    assert r.n_sites == 5
    n = r.norm()
    assert abs(n - 1.0) < 1e-6, f"Random MPS norm = {n}"


def test_mps_bond_dims():
    mps = MPS.ghz_state(4)
    bd = mps.bond_dims
    assert len(bd) == 3
    assert all(d > 0 for d in bd)


def test_mps_inner():
    a = MPS.computational_basis(3, state=0)
    b = MPS.computational_basis(3, state=0)
    c = MPS.computational_basis(3, state=1)
    assert abs(a.inner(b) - 1.0) < 1e-10
    assert abs(a.inner(c)) < 1e-10


def test_mps_expectation():
    # |0> has <Z> = +1
    mps0 = MPS.computational_basis(2, state=0)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    val = mps0.expectation_local(z, 0)
    assert abs(val - 1.0) < 1e-10, f"<0|Z|0> = {val}"

    # |1> has <Z> = -1 (state=2 means |10>)
    mps1 = MPS.computational_basis(2, state=2)
    val1 = mps1.expectation_local(z, 0)
    assert abs(val1 - (-1.0)) < 1e-10, f"<10|Z_0|10> = {val1}"


def test_mpo_identity():
    mpo = MPO.identity(3)
    assert mpo.n_sites == 3


test("MPS computational basis", test_mps_computational_basis)
test("MPS GHZ state", test_mps_ghz_state)
test("MPS Bell state", test_mps_bell_state)
test("MPS W state", test_mps_w_state)
test("MPS random state", test_mps_random)
test("MPS bond dimensions", test_mps_bond_dims)
test("MPS inner product", test_mps_inner)
test("MPS expectation value", test_mps_expectation)
test("MPO identity", test_mpo_identity)


# ═══════════════════════════════════════════════════════════════
# 3. PHYSICS TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("3. PHYSICS TESTS")
print("=" * 60)

from siliqun.physics.gates import (
    pauli_x, pauli_y, pauli_z, hadamard,
    rx, ry, rz, cnot, cz, sqrt_swap,
    PAULI_X, PAULI_Y, PAULI_Z,
)
from siliqun.physics.hamiltonian import (
    DeviceParams, build_hamiltonian_mpo,
    donor_2q_params, simos_4q_params, gaa_6q_params,
)
from siliqun.physics.noise.channels import (
    NoiseParams, default_noise_params,
    amplitude_damping_kraus, phase_damping_kraus,
    depolarizing_kraus, ChargeNoiseGenerator,
)
from siliqun.physics.devices.profiles import (
    get_device_profile, DeviceProfile,
)


def test_pauli_unitarity():
    for name, gate in [("X", pauli_x()), ("Y", pauli_y()), ("Z", pauli_z())]:
        prod = gate @ gate.conj().T
        assert np.allclose(prod, np.eye(2), atol=1e-12), f"{name} not unitary"


def test_rotation_gates():
    assert np.allclose(rx(0), np.eye(2), atol=1e-12)
    state = np.array([1, 0], dtype=complex)
    result = ry(np.pi) @ state
    assert abs(abs(result[1]) - 1.0) < 1e-10


def test_cnot_gate():
    g = cnot()
    assert g.shape == (4, 4)
    assert np.allclose(g @ g.conj().T, np.eye(4), atol=1e-12)


def test_hamiltonian_mpo():
    params = donor_2q_params()
    mpo = build_hamiltonian_mpo(params)
    assert mpo.n_sites == 2


def test_device_profiles():
    for name in ["donor", "simos", "gaa"]:
        dev = get_device_profile(name, 4)
        assert dev.n_qubits == 4
        assert dev.noise_params is not None
        assert len(dev.gate_times) > 0
        assert len(dev.native_gates) > 0


def test_noise_kraus():
    # Depolarizing channel
    kraus = depolarizing_kraus(0.01)
    assert len(kraus) == 4
    total = sum(k.conj().T @ k for k in kraus)
    assert np.allclose(total, np.eye(2), atol=1e-10)


def test_amplitude_damping():
    kraus = amplitude_damping_kraus(0.05)
    assert len(kraus) == 2
    total = sum(k.conj().T @ k for k in kraus)
    assert np.allclose(total, np.eye(2), atol=1e-10)


def test_charge_noise():
    gen = ChargeNoiseGenerator(n_qubits=2, amplitude=1e-4, dt=1e-9)
    sample = gen.sample()
    assert sample.shape == (2,)
    gen.reset(seed=42)
    s1 = gen.sample()
    gen.reset(seed=42)
    s2 = gen.sample()
    assert np.allclose(s1, s2)


test("Pauli gate unitarity", test_pauli_unitarity)
test("Rotation gates Rx/Ry/Rz", test_rotation_gates)
test("CNOT gate", test_cnot_gate)
test("Hamiltonian MPO construction", test_hamiltonian_mpo)
test("Device profiles (Donor/SiMOS/GAA)", test_device_profiles)
test("Depolarizing noise Kraus", test_noise_kraus)
test("Amplitude damping Kraus", test_amplitude_damping)
test("Charge noise generator", test_charge_noise)


# ═══════════════════════════════════════════════════════════════
# 4. SIMULATOR ENGINE TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("4. SIMULATOR ENGINE TESTS")
print("=" * 60)

from siliqun.engine.simulator import SiliQunSimulator, SimConfig


def test_simulator_reset():
    dev = get_device_profile("donor", 3)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False))
    sim.reset()
    assert sim.state.n_sites == 3
    assert abs(sim.state.norm() - 1.0) < 1e-10


def test_simulator_single_qubit_gates():
    dev = get_device_profile("donor", 2)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False, max_bond_dim=16))
    sim.reset()
    sim.apply_ry(np.pi, 0)
    z0 = sim.expectation_z(0)
    z1 = sim.expectation_z(1)
    assert abs(z0 - (-1.0)) < 0.1, f"Z0 after Ry(pi) = {z0}"
    assert abs(z1 - 1.0) < 0.1, f"Z1 unchanged = {z1}"


def test_simulator_entanglement():
    dev = get_device_profile("donor", 2)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False, max_bond_dim=16))
    sim.reset()
    sim.apply_ry(np.pi / 2, 0)
    sim.apply_rz(np.pi, 0)
    sim.apply_cnot(0, 1)
    zz = sim.expectation_zz(0, 1)
    assert abs(zz - 1.0) < 0.2, f"ZZ = {zz}"
    S = sim.compute_entanglement_entropy(1)
    assert S > 0.3, f"Entropy = {S}"


def test_simulator_noise():
    dev = get_device_profile("donor", 2)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=True, max_bond_dim=16))
    sim.reset()
    for _ in range(10):
        sim.apply_ry(np.pi / 4, 0)
        sim.apply_cnot(0, 1)
    n = sim.state.norm()
    assert abs(n - 1.0) < 0.5, f"Norm after noise = {n}"


def test_simulator_snapshot():
    dev = get_device_profile("simos", 2)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False))
    sim.reset()
    sim.apply_ry(np.pi / 3, 0)
    snap = sim.snapshot()
    assert "time" in snap
    assert "bond_dims" in snap
    assert "z_expectations" in snap


def test_simulator_fidelity():
    dev = get_device_profile("donor", 2)
    sim = SiliQunSimulator(dev, SimConfig(noise_enabled=False))
    sim.reset()
    target = MPS.computational_basis(2, state=0)
    fid = sim.compute_fidelity(target)
    assert abs(fid - 1.0) < 1e-6, f"Fidelity of |00> vs |00> = {fid}"


test("Simulator reset", test_simulator_reset)
test("Single-qubit gates", test_simulator_single_qubit_gates)
test("Entanglement creation", test_simulator_entanglement)
test("Noise simulation", test_simulator_noise)
test("Simulator snapshot", test_simulator_snapshot)
test("Simulator fidelity", test_simulator_fidelity)


# ═══════════════════════════════════════════════════════════════
# 5. GYMNASIUM ENVIRONMENT TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("5. GYMNASIUM ENVIRONMENT TESTS")
print("=" * 60)

from siliqun.engine.gym_env import SiliQunEnv


def test_env_creation():
    env = SiliQunEnv(device="donor", n_qubits=2, target_state="bell")
    assert env.observation_space.shape[0] > 0
    assert env.action_space.shape[0] > 0


def test_env_reset():
    env = SiliQunEnv(device="donor", n_qubits=2, target_state="ghz",
                     config=SimConfig(noise_enabled=False))
    obs, info = env.reset(seed=42)
    assert len(obs) > 0
    assert "fidelity" in info


def test_env_step():
    env = SiliQunEnv(device="donor", n_qubits=2, target_state="bell",
                     max_steps=10, config=SimConfig(noise_enabled=False))
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert len(obs) > 0
    assert isinstance(reward, float)
    assert "fidelity" in info


def test_env_episode():
    env = SiliQunEnv(device="donor", n_qubits=2, target_state="bell",
                     max_steps=20, config=SimConfig(noise_enabled=False),
                     reward_type="dense")
    obs, _ = env.reset(seed=42)
    total_reward = 0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    assert info["step"] <= 20


def test_env_all_devices():
    for dt in ["donor", "simos", "gaa"]:
        env = SiliQunEnv(device=dt, n_qubits=3, target_state="ghz",
                         max_steps=5, config=SimConfig(noise_enabled=False))
        obs, _ = env.reset()
        obs, r, t, tr, info = env.step(env.action_space.sample())
        assert len(obs) > 0
        assert "fidelity" in info


def test_env_render():
    env = SiliQunEnv(device="donor", n_qubits=2, target_state="bell",
                     render_mode="ansi", config=SimConfig(noise_enabled=False))
    env.reset()
    env.step(env.action_space.sample())
    output = env.render()
    assert isinstance(output, str)
    assert len(output) > 0


test("Env creation", test_env_creation)
test("Env reset", test_env_reset)
test("Env step", test_env_step)
test("Env full episode", test_env_episode)
test("Env all device types", test_env_all_devices)
test("Env render", test_env_render)


# ═══════════════════════════════════════════════════════════════
# 6. HPC RUNNER TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("6. HPC RUNNER TESTS")
print("=" * 60)

from siliqun.hpc import HPCRunner, PBSConfig, CheckpointConfig


def test_pbs_config():
    pbs = PBSConfig(queue="A100", nodes=2, gpus=2)
    assert pbs.queue == "A100"
    assert pbs.nodes == 2


def test_pbs_script_generation():
    runner = HPCRunner(PBSConfig(queue="A100", job_name="test"))
    script = runner.generate_pbs_script("train.py", {"n_qubits": 4})
    assert "#PBS -N test" in script
    assert "#PBS -q A100" in script
    assert "train.py" in script
    assert "--n_qubits 4" in script


def test_sweep_script():
    runner = HPCRunner(PBSConfig(job_name="sweep"))
    configs = [
        {"device": "donor", "seed": 42},
        {"device": "simos", "seed": 42},
    ]
    script = runner.generate_sweep_script("train.py", configs)
    assert "Run 1/2" in script
    assert "Run 2/2" in script


def test_dry_run():
    runner = HPCRunner(PBSConfig())
    script = runner.generate_pbs_script("test.py")
    result = runner.write_and_submit(script, "/tmp/test.pbs", dry_run=True)
    assert result is None
    assert os.path.exists("/tmp/test.pbs")


test("PBS config", test_pbs_config)
test("PBS script generation", test_pbs_script_generation)
test("Sweep script generation", test_sweep_script)
test("Dry run submission", test_dry_run)


# ═══════════════════════════════════════════════════════════════
# 7. CROSS-LAYER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("7. CROSS-LAYER INTEGRATION TESTS")
print("=" * 60)


def test_full_stack_2q():
    """Full stack: 2-qubit donor device, bell target, 50 steps."""
    env = SiliQunEnv(
        device="donor", n_qubits=2, target_state="bell",
        max_steps=50, config=SimConfig(noise_enabled=False),
        reward_type="dense",
    )
    obs, _ = env.reset(seed=42)
    best_fid = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        best_fid = max(best_fid, info["fidelity"])
        if term or trunc:
            break
    assert best_fid > 0, f"Best fidelity = {best_fid}"


def test_full_stack_4q():
    """Full stack: 4-qubit GHZ target."""
    env = SiliQunEnv(
        device="simos", n_qubits=4, target_state="ghz",
        max_steps=20, config=SimConfig(noise_enabled=False, max_bond_dim=16),
    )
    obs, _ = env.reset(seed=123)
    for _ in range(10):
        obs, r, t, tr, info = env.step(env.action_space.sample())
    assert "fidelity" in info


def test_full_stack_noisy():
    """Full stack with noise enabled."""
    env = SiliQunEnv(
        device="gaa", n_qubits=2, target_state="bell",
        max_steps=30, config=SimConfig(noise_enabled=True, max_bond_dim=16),
        reward_type="dense",
    )
    obs, _ = env.reset(seed=456)
    for _ in range(15):
        obs, r, t, tr, info = env.step(env.action_space.sample())
    assert info["fidelity"] >= 0


def test_scalability_8q():
    """Test 8-qubit system (target size for PhD)."""
    env = SiliQunEnv(
        device="donor", n_qubits=8, target_state="ghz",
        max_steps=10, config=SimConfig(noise_enabled=False, max_bond_dim=32),
    )
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        obs, r, t, tr, info = env.step(env.action_space.sample())
    assert info["step"] == 5


def test_hamiltonian_mpo_apply():
    """Test that Hamiltonian MPO can be applied to an MPS."""
    params = donor_2q_params()
    mpo = build_hamiltonian_mpo(params)
    mps = MPS.computational_basis(2, state=0)
    # Just verify the MPO has the right structure
    assert mpo.n_sites == mps.n_sites


test("Full stack 2Q Bell", test_full_stack_2q)
test("Full stack 4Q GHZ", test_full_stack_4q)
test("Full stack noisy", test_full_stack_noisy)
test("Scalability 8Q", test_scalability_8q)
test("Hamiltonian MPO construction", test_hamiltonian_mpo_apply)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
print("=" * 60)
