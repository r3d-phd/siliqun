"""
Test suite for Nature paper integration:
    - DFS encoding (3 spins → 1 logical qubit)
    - Sequential pulsing constraint
    - SLEDGE device profile
    - Calibrated noise parameters
    - Updated gym environment

Validates against experimental data from:
    Weinstein et al., Nature 615, 817-822 (2023)
"""

import sys
import os
import numpy as np

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ══════════════════════════════════════════════════════════════════════
# 1. DFS Encoding Tests
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("1. DFS ENCODING TESTS")
print("=" * 60)

from siliqun.physics.dfs_encoding import (
    encoded_zero, encoded_one, gauge_zero,
    encoded_subspace_projector, leakage_subspace_projector,
    compute_leakage, compute_gauge_population,
    exchange_12, exchange_23,
    logical_z_rotation, logical_x_rotation,
    logical_arbitrary_rotation,
    partial_swap, exchange_quality_factor,
    DFSEncoder,
)

# Test 1.1: Encoded basis states are normalised
z = encoded_zero()
o = encoded_one()
g = gauge_zero()

check("encoded_zero is normalised", abs(np.linalg.norm(z) - 1.0) < 1e-10,
      f"norm = {np.linalg.norm(z)}")
check("encoded_one is normalised", abs(np.linalg.norm(o) - 1.0) < 1e-10,
      f"norm = {np.linalg.norm(o)}")
check("gauge_zero is normalised", abs(np.linalg.norm(g) - 1.0) < 1e-10,
      f"norm = {np.linalg.norm(g)}")

# Test 1.2: Orthogonality
check("⟨0_L|1_L⟩ = 0", abs(z.conj() @ o) < 1e-10,
      f"overlap = {abs(z.conj() @ o)}")
check("⟨0_L|g0⟩ = 0", abs(z.conj() @ g) < 1e-10,
      f"overlap = {abs(z.conj() @ g)}")
check("⟨1_L|g0⟩ = 0", abs(o.conj() @ g) < 1e-10,
      f"overlap = {abs(o.conj() @ g)}")

# Test 1.3: Same total m_s = -1/2
# For |0_L⟩ and |1_L⟩, check that total Sz = -1/2
Sz_total = np.zeros((8, 8), dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
for i in range(3):
    ops = [I2, I2, I2]
    ops[i] = Sz
    term = ops[0]
    for op in ops[1:]:
        term = np.kron(term, op)
    Sz_total += term

mz_zero = float(np.real(z.conj() @ Sz_total @ z))
mz_one = float(np.real(o.conj() @ Sz_total @ o))
check("|0_L⟩ has m_s = -1/2", abs(mz_zero - (-0.5)) < 1e-10,
      f"m_s = {mz_zero}")
check("|1_L⟩ has m_s = -1/2", abs(mz_one - (-0.5)) < 1e-10,
      f"m_s = {mz_one}")

# Test 1.4: Projectors
P_enc = encoded_subspace_projector()
P_leak = leakage_subspace_projector()

check("P_enc is Hermitian", np.allclose(P_enc, P_enc.conj().T))
check("P_enc is idempotent", np.allclose(P_enc @ P_enc, P_enc))
check("P_enc rank = 2", abs(np.trace(P_enc) - 2.0) < 1e-10,
      f"trace = {np.trace(P_enc)}")

# Test 1.5: Leakage computation
check("encoded state has 0 leakage", compute_leakage(z) < 1e-10)
check("encoded state has 0 leakage", compute_leakage(o) < 1e-10)
# A random state should have non-zero leakage
random_state = np.random.randn(8) + 1j * np.random.randn(8)
random_state /= np.linalg.norm(random_state)
leak = compute_leakage(random_state)
check("random state has leakage > 0", leak > 0.01,
      f"leakage = {leak}")

# Test 1.6: Exchange gates preserve encoded subspace
# J12 (logical Z) should keep |0_L⟩ and |1_L⟩ in the encoded subspace
for theta in [0.1, 0.5, np.pi / 4, np.pi / 2, np.pi]:
    U12 = exchange_12(theta)
    z_evolved = U12 @ z
    o_evolved = U12 @ o
    check(f"J12(θ={theta:.2f}) preserves |0_L⟩ encoding",
          compute_leakage(z_evolved) < 1e-8,
          f"leakage = {compute_leakage(z_evolved)}")
    check(f"J12(θ={theta:.2f}) preserves |1_L⟩ encoding",
          compute_leakage(o_evolved) < 1e-8,
          f"leakage = {compute_leakage(o_evolved)}")

# Test 1.7: Partial SWAP gate
ps = partial_swap(np.pi)  # Full SWAP
check("partial_swap(π) is unitary",
      np.allclose(ps @ ps.conj().T, np.eye(4), atol=1e-10))
# SWAP|01⟩ = |10⟩
state_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
state_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
swapped = ps @ state_01
check("SWAP|01⟩ = |10⟩ (up to phase)",
      abs(abs(swapped @ state_10.conj()) - 1.0) < 1e-8,
      f"overlap = {abs(swapped @ state_10.conj())}")

# Test 1.8: Exchange quality factor
N_osc = exchange_quality_factor(100e6, 3.5e-6)
check("N_osc ≈ 350 at J=100MHz, T2*=3.5μs",
      abs(N_osc - 350.0) < 1.0,
      f"N_osc = {N_osc}")

# Test 1.9: DFSEncoder multi-qubit
print("\n  --- DFSEncoder (multi-qubit) ---")
for n_L in [1, 2, 3]:
    enc = DFSEncoder(n_L)
    check(f"DFSEncoder({n_L}L): V shape = ({2**(3*n_L)}, {2**n_L})",
          enc.encoding_isometry.shape == (2**(3*n_L), 2**n_L),
          f"shape = {enc.encoding_isometry.shape}")

    # V should be an isometry: V†V = I
    VtV = enc.encoding_isometry.conj().T @ enc.encoding_isometry
    check(f"DFSEncoder({n_L}L): V†V = I",
          np.allclose(VtV, np.eye(2**n_L), atol=1e-8),
          f"max error = {np.max(np.abs(VtV - np.eye(2**n_L)))}")

    # Encode |0...0⟩ and check leakage
    logical_zero = np.zeros(2**n_L, dtype=np.complex128)
    logical_zero[0] = 1.0
    physical = enc.encode(logical_zero)
    check(f"DFSEncoder({n_L}L): encoded |0⟩ has 0 leakage",
          enc.compute_total_leakage(physical) < 1e-8)

    # Round-trip: encode then decode
    decoded = enc.decode(physical)
    check(f"DFSEncoder({n_L}L): round-trip fidelity = 1",
          abs(np.abs(decoded.conj() @ logical_zero)**2 - 1.0) < 1e-8)


# ══════════════════════════════════════════════════════════════════════
# 2. Sequential Pulsing Tests
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. SEQUENTIAL PULSING TESTS")
print("=" * 60)

from siliqun.physics.sequential_pulsing import (
    ExchangePulse, PulseLayer, PulseScheduler, SequentialActionSpace,
)

# Test 2.1: Conflict detection
p1 = ExchangePulse(0, 1, np.pi / 4)
p2 = ExchangePulse(1, 2, np.pi / 4)
p3 = ExchangePulse(2, 3, np.pi / 4)

check("Pulses (0,1) and (1,2) conflict", p1.conflicts_with(p2))
check("Pulses (0,1) and (2,3) don't conflict", not p1.conflicts_with(p3))

# Test 2.2: Sequential scheduling (linear 4q chain)
linear_4q = [(0, 1), (1, 2), (2, 3)]
scheduler_seq = PulseScheduler(4, linear_4q, sequential=True)

pulses = [
    ExchangePulse(0, 1, 0.1, label="J01"),
    ExchangePulse(1, 2, 0.2, label="J12"),
    ExchangePulse(2, 3, 0.3, label="J23"),
]
layers = scheduler_seq.schedule(pulses)
check("Sequential: 3 pulses → 3 layers", len(layers) == 3,
      f"got {len(layers)} layers")
for i, layer in enumerate(layers):
    check(f"Sequential layer {i}: exactly 1 pulse",
          len(layer.pulses) == 1)

# Test 2.3: Parallel scheduling (linear 4q chain)
scheduler_par = PulseScheduler(4, linear_4q, sequential=False)
layers_par = scheduler_par.schedule(pulses)
check("Parallel: 3 pulses → 2 layers (J01∥J23, then J12)",
      len(layers_par) == 2,
      f"got {len(layers_par)} layers")

# Test 2.4: 2D grid scheduling (2×2)
grid_2x2 = [(0, 1), (2, 3), (0, 2), (1, 3)]
scheduler_2d = PulseScheduler(4, grid_2x2, sequential=True)
pulses_2d = [ExchangePulse(e[0], e[1], 0.1) for e in grid_2x2]
layers_2d = scheduler_2d.schedule(pulses_2d)
check("2×2 grid sequential: 4 pulses → 4 layers", len(layers_2d) == 4)

# Parallel 2×2
scheduler_2d_par = PulseScheduler(4, grid_2x2, sequential=False)
layers_2d_par = scheduler_2d_par.schedule(pulses_2d)
check("2×2 grid parallel: 4 pulses → 2 layers",
      len(layers_2d_par) == 2,
      f"got {len(layers_2d_par)} layers")

# Test 2.5: Edge colouring
colouring = scheduler_2d_par.get_edge_colouring()
check("2×2 grid chromatic index ≤ 3",
      max(colouring.values()) + 1 <= 3,
      f"chromatic index = {max(colouring.values()) + 1}")

# Test 2.6: 3×3 grid scheduling
grid_3x3 = [
    (0, 1), (1, 2),           # row 0
    (3, 4), (4, 5),           # row 1
    (6, 7), (7, 8),           # row 2
    (0, 3), (1, 4), (2, 5),   # col 0,1,2
    (3, 6), (4, 7), (5, 8),   # col 0,1,2
]
scheduler_3x3 = PulseScheduler(9, grid_3x3, sequential=True)
pulses_3x3 = [ExchangePulse(e[0], e[1], 0.1) for e in grid_3x3]
layers_3x3 = scheduler_3x3.schedule(pulses_3x3)
check("3×3 grid sequential: 12 pulses → 12 layers",
      len(layers_3x3) == 12)

# Parallel 3×3
scheduler_3x3_par = PulseScheduler(9, grid_3x3, sequential=False)
layers_3x3_par = scheduler_3x3_par.schedule(pulses_3x3)
check(f"3×3 grid parallel: 12 pulses → {len(layers_3x3_par)} layers (< 12)",
      len(layers_3x3_par) < 12)

# Test 2.7: SequentialActionSpace
seq_action = SequentialActionSpace(4, grid_2x2, sequential=True)
check("SequentialActionSpace action_dim = 4 (2×2 grid)",
      seq_action.action_dim == 4)

action = np.array([0.5, -0.3, 0.0, 0.8])
schedule = seq_action.action_to_schedule(action)
# Only non-zero angles should produce pulses
check("Zero-angle edge skipped", len(schedule) == 3,
      f"got {len(schedule)} layers (expected 3, skipping edge with angle 0)")

# Test 2.8: Validation
valid, errors = scheduler_2d.validate_pulse_sequence([
    ExchangePulse(0, 1, 0.1),
    ExchangePulse(0, 2, 0.1),  # vertical edge in 2x2 grid
])
check("Valid pulses on 2×2 grid", valid, str(errors))

invalid_pulse = ExchangePulse(0, 3, 0.1)
valid2, errors2 = scheduler_seq.validate_pulse_sequence([invalid_pulse])
check("Invalid pulse (0,3) on linear 4q detected", not valid2)


# ══════════════════════════════════════════════════════════════════════
# 3. SLEDGE Device Profile Tests
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. SLEDGE DEVICE PROFILE TESTS")
print("=" * 60)

from siliqun.physics.devices.profiles import (
    sledge_device, sledge_2x2, sledge_3x3, sledge_4x2,
    get_device_profile, DeviceProfile,
)

# Test 3.1: Basic SLEDGE device
dev = sledge_device(n_qubits=3)
check("SLEDGE-3Q name", "SLEDGE" in dev.name)
check("SLEDGE-3Q is DFS encoded", dev.dfs_encoded)
check("SLEDGE-3Q has 9 physical qubits", dev.n_physical_qubits == 9)
check("SLEDGE-3Q sequential pulsing", dev.sequential_pulsing)
check("SLEDGE-3Q linear connectivity (2 edges)",
      len(dev.connectivity) == 2)

# Test 3.2: SLEDGE 2×2
dev_2x2 = sledge_2x2()
check("SLEDGE-2×2 has 4 logical qubits", dev_2x2.n_qubits == 4)
check("SLEDGE-2×2 has 12 physical qubits", dev_2x2.n_physical_qubits == 12)
check("SLEDGE-2×2 grid_shape = (2,2)", dev_2x2.grid_shape == (2, 2))
check("SLEDGE-2×2 has 4 edges", len(dev_2x2.connectivity) == 4)
check("SLEDGE-2×2 is 2D", dev_2x2.is_2d)

# Test 3.3: SLEDGE 3×3
dev_3x3 = sledge_3x3()
check("SLEDGE-3×3 has 9 logical qubits", dev_3x3.n_qubits == 9)
check("SLEDGE-3×3 has 27 physical qubits", dev_3x3.n_physical_qubits == 27)
check("SLEDGE-3×3 grid_shape = (3,3)", dev_3x3.grid_shape == (3, 3))
check("SLEDGE-3×3 has 12 edges", len(dev_3x3.connectivity) == 12)

# Test 3.4: SLEDGE 4×2
dev_4x2 = sledge_4x2()
check("SLEDGE-4×2 has 8 logical qubits", dev_4x2.n_qubits == 8)
check("SLEDGE-4×2 has 24 physical qubits", dev_4x2.n_physical_qubits == 24)
check("SLEDGE-4×2 grid_shape = (4,2)", dev_4x2.grid_shape == (4, 2))
# 4×2 grid: 3 horizontal + 4 vertical = 7... wait
# rows=4, cols=2: horizontal = 4*1=4, vertical = 3*2=6 → 10 edges
n_edges_4x2 = len(dev_4x2.connectivity)
expected_edges = 4 * (2 - 1) + (4 - 1) * 2  # 4 + 6 = 10
check(f"SLEDGE-4×2 has {expected_edges} edges",
      n_edges_4x2 == expected_edges,
      f"got {n_edges_4x2}")

# Test 3.5: Registry
dev_reg = get_device_profile("sledge", n_qubits=6, grid_shape=(3, 2))
check("Registry: sledge 3×2 works", dev_reg.n_qubits == 6)
check("Registry: sledge 3×2 grid_shape", dev_reg.grid_shape == (3, 2))

# Test 3.6: Exchange couplings match Nature paper (~100 MHz)
J_values = dev.hamiltonian_params.exchange_couplings
for i, J in enumerate(J_values):
    check(f"J[{i}] ≈ 100 MHz (±5 MHz)",
          95e6 <= J <= 105e6,
          f"J = {J/1e6:.1f} MHz")

# Test 3.7: Gate times
check("SLEDGE single gate ≈ 10 ns",
      dev.gate_times["single"] == 10e-9)
check("SLEDGE two-qubit gate ≈ 100 ns",
      dev.gate_times["two"] == 100e-9)
check("SLEDGE readout ≈ 10 μs",
      dev.gate_times["readout"] == 10e-6)

# Test 3.8: Native gates
check("SLEDGE has Exchange_partial_SWAP",
      "Exchange_partial_SWAP" in dev.native_gates)
check("SLEDGE has FW_CNOT",
      "FW_CNOT" in dev.native_gates)
check("SLEDGE has LCCZ",
      "LCCZ" in dev.native_gates)


# ══════════════════════════════════════════════════════════════════════
# 4. Noise Parameters Tests
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("4. CALIBRATED NOISE PARAMETERS TESTS")
print("=" * 60)

from siliqun.physics.noise.channels import (
    NoiseParams, default_noise_params,
)

# Test 4.1: SLEDGE noise params
noise_sledge = default_noise_params(3, "sledge")
check("SLEDGE T2* = 3.5 μs", abs(noise_sledge.t2_star_times[0] - 3.5e-6) < 1e-10)
check("SLEDGE dephasing = gaussian",
      noise_sledge.dephasing_model == "gaussian")
check("SLEDGE has crosstalk",
      noise_sledge.crosstalk_amplitude > 0)
check("SLEDGE DFS encoded",
      noise_sledge.dfs_encoded is True)

# Test 4.2: Donor noise params (should be unchanged)
noise_donor = default_noise_params(2, "donor")
check("Donor T1 = 30 s", abs(noise_donor.t1_times[0] - 30.0) < 1e-10)
check("Donor T2* = 0.5 ms", abs(noise_donor.t2_star_times[0] - 0.5e-3) < 1e-10)

# Test 4.3: SiMOS noise params
noise_simos = default_noise_params(4, "simos")
check("SiMOS T2* = 20 μs", abs(noise_simos.t2_star_times[0] - 20e-6) < 1e-10)

# Test 4.4: GAA noise params
noise_gaa = default_noise_params(6, "gaa")
check("GAA T2* = 10 μs", abs(noise_gaa.t2_star_times[0] - 10e-6) < 1e-10)


# ══════════════════════════════════════════════════════════════════════
# 5. Gym Environment Integration Tests
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("5. GYM ENVIRONMENT INTEGRATION TESTS")
print("=" * 60)

from siliqun.engine.gym_env import SiliQunEnv, make_siliqun_env

# Test 5.1: Standard donor env (backward compatibility)
try:
    env_donor = make_siliqun_env(n_qubits=2, device="donor", target="bell")
    obs, info = env_donor.reset()
    check("Donor 2q env creates successfully", True)
    check("Donor 2q obs shape = (7,)",
          obs.shape == (7,),
          f"shape = {obs.shape}")
    check("Donor 2q info has fidelity", "fidelity" in info)

    # Step
    action = env_donor.action_space.sample()
    obs2, reward, term, trunc, info2 = env_donor.step(action)
    check("Donor 2q step works", True)
    check("Donor 2q step obs shape matches", obs2.shape == obs.shape)
except Exception as e:
    check("Donor 2q env", False, str(e))

# Test 5.2: Standard SiMOS env with 2D grid info
try:
    env_simos = make_siliqun_env(n_qubits=4, device="simos", target="ghz")
    obs_s, info_s = env_simos.reset()
    check("SiMOS 4q env creates successfully", True)
    # SiMOS is linear: obs = 3*4 + 1 = 13
    check("SiMOS 4q obs shape = (13,)",
          obs_s.shape == (13,),
          f"shape = {obs_s.shape}")
except Exception as e:
    check("SiMOS 4q env", False, str(e))

# Test 5.3: SLEDGE env (DFS encoded)
try:
    env_sledge = make_siliqun_env(
        n_qubits=3, device="sledge", target="ghz",
        max_steps=100,
    )
    obs_sl, info_sl = env_sledge.reset()
    check("SLEDGE 3q env creates successfully", True)
    check("SLEDGE 3q info has dfs_encoded", info_sl.get("dfs_encoded") is True)

    # Obs dim: n(3) + edges(2) + cuts(2) + scalars(3) + leakage(1) + gauge(3) = 14
    expected_obs_dim = 3 + 2 + 2 + 3 + 1 + 3
    check(f"SLEDGE 3q obs dim = {expected_obs_dim}",
          obs_sl.shape[0] == expected_obs_dim,
          f"shape = {obs_sl.shape}")

    # Action dim = n_edges = 2
    check("SLEDGE 3q action dim = 2",
          env_sledge.action_space.shape[0] == 2,
          f"action shape = {env_sledge.action_space.shape}")

    # Step
    action_sl = env_sledge.action_space.sample()
    obs_sl2, rew_sl, term_sl, trunc_sl, info_sl2 = env_sledge.step(action_sl)
    check("SLEDGE 3q step works", True)
    check("SLEDGE step info has leakage", "leakage" in info_sl2)
except Exception as e:
    check("SLEDGE 3q env", False, str(e))

# Test 5.4: SLEDGE 2×2 grid env
try:
    env_2x2 = make_siliqun_env(
        n_qubits=4, device="sledge", target="ghz",
        grid_shape=(2, 2), max_steps=100,
    )
    obs_2x2, info_2x2 = env_2x2.reset()
    check("SLEDGE 2×2 env creates successfully", True)
    check("SLEDGE 2×2 has grid_shape in info",
          info_2x2.get("grid_shape") == (2, 2))

    # Obs dim: n(4) + edges(4) + cuts(?) + scalars(3) + leakage(1) + gauge(4)
    check(f"SLEDGE 2×2 obs dim > 0", obs_2x2.shape[0] > 0,
          f"obs dim = {obs_2x2.shape[0]}")

    # Action dim = 4 edges
    check("SLEDGE 2×2 action dim = 4",
          env_2x2.action_space.shape[0] == 4,
          f"action shape = {env_2x2.action_space.shape}")
except Exception as e:
    check("SLEDGE 2×2 env", False, str(e))

# Test 5.5: SLEDGE 3×3 grid env
try:
    env_3x3 = make_siliqun_env(
        n_qubits=9, device="sledge", target="ghz",
        grid_shape=(3, 3), max_steps=500,
    )
    obs_3x3, info_3x3 = env_3x3.reset()
    check("SLEDGE 3×3 env creates successfully", True)
    check("SLEDGE 3×3 grid_shape = (3,3)",
          info_3x3.get("grid_shape") == (3, 3))

    # Action dim = 12 edges
    check("SLEDGE 3×3 action dim = 12",
          env_3x3.action_space.shape[0] == 12,
          f"action shape = {env_3x3.action_space.shape}")
except Exception as e:
    check("SLEDGE 3×3 env", False, str(e))


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"FAILURES: {FAIL}")
print("=" * 60)
