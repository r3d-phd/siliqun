"""
Validation test suite for the generalised SiliQun modules.
Tests: tomography, compiler, Lindblad solver, REST API.
"""
import sys
sys.path.insert(0, '/home/ubuntu/siliqun')

import numpy as np
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results.append((name, False, str(e)))

# ── Tomography ──────────────────────────────────────────────────────────────
print("\n=== Tomography ===")

def test_qst_bell():
    from siliqun.tomography import StateTomography, fidelity, purity
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    qst = StateTomography(n_qubits=2)
    exp = qst.simulate_measurements(rho, n_shots=50000, add_shot_noise=False)
    res = qst.reconstruct(exp, target_state=rho)
    assert abs(res.purity - 1.0) < 1e-6, f"Purity {res.purity} != 1.0"
    assert abs(res.fidelity_to_target - 1.0) < 1e-6, f"Fidelity {res.fidelity_to_target} != 1.0"

def test_qst_mixed():
    from siliqun.tomography import StateTomography, purity
    # Mixed state: maximally mixed
    rho = np.eye(4) / 4
    qst = StateTomography(n_qubits=2)
    exp = qst.simulate_measurements(rho, n_shots=50000, add_shot_noise=False)
    res = qst.reconstruct(exp)
    assert abs(res.purity - 0.25) < 1e-4, f"Purity {res.purity} != 0.25"

def test_fidelity_pure():
    from siliqun.tomography import fidelity
    psi = np.array([1, 0]) / 1.0
    rho = np.outer(psi, psi.conj())
    f = fidelity(rho, rho)
    assert abs(f - 1.0) < 1e-10, f"Self-fidelity {f} != 1.0"

def test_fidelity_orthogonal():
    from siliqun.tomography import fidelity
    rho0 = np.array([[1,0],[0,0]], dtype=complex)
    rho1 = np.array([[0,0],[0,1]], dtype=complex)
    f = fidelity(rho0, rho1)
    assert abs(f) < 1e-10, f"Orthogonal fidelity {f} != 0.0"

def test_purity_pure():
    from siliqun.tomography import purity
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    p = purity(rho)
    assert abs(p - 1.0) < 1e-10, f"Pure state purity {p} != 1.0"

def test_reconstruct_density_matrix():
    from siliqun.tomography import reconstruct_density_matrix
    # |0><0| should give expectations: Z=+1, X=0, Y=0
    expectations = {"I": 1.0, "Z": 1.0, "X": 0.0, "Y": 0.0}
    rho = reconstruct_density_matrix(expectations, n_qubits=1)
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(rho, expected, atol=1e-10), f"Density matrix mismatch"

test("QST Bell state (linear, no noise)", test_qst_bell)
test("QST maximally mixed state", test_qst_mixed)
test("Fidelity: pure state self-fidelity", test_fidelity_pure)
test("Fidelity: orthogonal states", test_fidelity_orthogonal)
test("Purity: pure state", test_purity_pure)
test("Reconstruct |0><0| from Pauli expectations", test_reconstruct_density_matrix)

# ── Compiler ─────────────────────────────────────────────────────────────────
print("\n=== Gate-to-Pulse Compiler ===")

def test_compiler_import():
    from siliqun.compiler.gate_compiler import GateToPulseCompiler, QFORGE_FIDELITY_TABLE
    assert "cnot" in QFORGE_FIDELITY_TABLE
    assert "cz" in QFORGE_FIDELITY_TABLE
    assert "sqrt_swap" in QFORGE_FIDELITY_TABLE

def test_compiler_single_qubit():
    from siliqun.compiler.gate_compiler import GateToPulseCompiler
    circuit = [("rx", {"theta": np.pi/2}, [0]), ("rz", {"theta": np.pi}, [1])]
    compiler = GateToPulseCompiler()
    result = compiler.compile(circuit)
    assert result.total_fidelity > 0.99, f"Fidelity {result.total_fidelity} too low"
    assert result.n_drive_pulses >= 1

def test_compiler_cnot():
    from siliqun.compiler.gate_compiler import GateToPulseCompiler
    circuit = [("h", {}, [0]), ("cnot", {}, [0, 1])]
    compiler = GateToPulseCompiler()
    result = compiler.compile(circuit)
    assert result.n_exchange_pulses >= 1, "CNOT should produce exchange pulses"
    assert result.total_fidelity > 0.99

def test_compiler_qasm():
    from siliqun.compiler.gate_compiler import GateToPulseCompiler
    qasm = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];'''
    compiler = GateToPulseCompiler()
    result = compiler.compile_qasm(qasm)
    assert result.n_exchange_pulses >= 1

def test_compiler_fidelity_breakdown():
    from siliqun.compiler.gate_compiler import GateToPulseCompiler
    circuit = [("cnot", {}, [0, 1]), ("cz", {}, [1, 2])]
    compiler = GateToPulseCompiler()
    result = compiler.compile(circuit)
    breakdown = result.fidelity_breakdown()
    # Keys are in the form "gate@q[qubits]" e.g. "cnot@q[0, 1]"
    assert any("cnot" in k for k in breakdown) or any("cz" in k for k in breakdown), \
        f"Expected cnot or cz in breakdown keys: {list(breakdown.keys())}"

test("Compiler imports and fidelity table", test_compiler_import)
test("Single-qubit gate compilation", test_compiler_single_qubit)
test("CNOT compilation with exchange pulses", test_compiler_cnot)
test("QASM string compilation", test_compiler_qasm)
test("Per-gate fidelity breakdown", test_compiler_fidelity_breakdown)

# ── REST API ──────────────────────────────────────────────────────────────────
print("\n=== REST API ===")

def test_api_import():
    from siliqun.api.server import app
    routes = {r.path for r in app.routes if hasattr(r, 'path')}
    required = {"/health", "/devices", "/simulate/circuit", "/compile",
                "/tomography/state", "/fidelity", "/jobs/{job_id}"}
    missing = required - routes
    assert not missing, f"Missing routes: {missing}"

def test_api_pydantic_models():
    from siliqun.api.server import (
        CircuitSimRequest, DeviceConfig, GateInstruction,
        CompileRequest, TomographyRequest
    )
    req = CircuitSimRequest(
        circuit=[GateInstruction(gate="h", qubits=[0], params={})],
        device=DeviceConfig(device="simos", n_qubits=2),
    )
    assert req.device.device == "simos"
    assert req.sim_mode == "auto"

def test_api_device_validator():
    from siliqun.api.server import DeviceConfig
    from pydantic import ValidationError
    try:
        DeviceConfig(device="invalid_device")
        assert False, "Should have raised ValidationError"
    except (ValidationError, ValueError):
        pass  # Expected

test("API routes registered correctly", test_api_import)
test("Pydantic request models", test_api_pydantic_models)
test("Device profile validator", test_api_device_validator)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"Results: {passed}/{total} passed")
if passed < total:
    print("\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("All tests passed.")
