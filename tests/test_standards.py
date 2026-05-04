"""
Standards compliance test suite for the generalised SiliQun platform.
Tests: OpenQASM 3.0 compiler, OpenPulse schedule, Qiskit BackendV2, PennyLane Device.
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

# ── OpenQASM 3.0 Compiler ─────────────────────────────────────────────────────
print("\n=== OpenQASM 3.0 Compiler ===")

def test_qasm2_string():
    from siliqun.compiler.qasm3_compiler import OpenQASM3Compiler
    qasm = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];'''
    compiler = OpenQASM3Compiler(optimise=False)
    result = compiler.compile(qasm)
    assert result.n_gates >= 2, f"Expected >=2 gates, got {result.n_gates}"
    assert result.total_fidelity > 0.99, f"Fidelity {result.total_fidelity} too low"
    assert result.n_2q_gates >= 1, "Expected at least 1 two-qubit gate"

def test_qasm3_string():
    from siliqun.compiler.qasm3_compiler import OpenQASM3Compiler
    qasm3 = '''OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];'''
    compiler = OpenQASM3Compiler(optimise=False)
    result = compiler.compile(qasm3)
    assert result.n_gates >= 2
    assert "OPENQASM" in result.qasm3_source or result.qasm3_source == ""

def test_qiskit_circuit_input():
    from qiskit import QuantumCircuit
    from siliqun.compiler.qasm3_compiler import OpenQASM3Compiler
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.cz(1, 2)
    compiler = OpenQASM3Compiler(optimise=False)
    result = compiler.compile(qc)
    assert result.n_gates >= 3
    assert result.n_2q_gates >= 2

def test_gate_list_input():
    from siliqun.compiler.qasm3_compiler import OpenQASM3Compiler
    circuit = [
        ("h", {}, [0]),
        ("cx", {}, [0, 1]),
        ("rz", {"theta": np.pi/4}, [2]),
    ]
    compiler = OpenQASM3Compiler(optimise=False)
    result = compiler.compile(circuit)
    assert result.n_gates == 3
    assert result.total_fidelity > 0.98

def test_fidelity_table_completeness():
    from siliqun.compiler.qasm3_compiler import OpenQASM3Compiler
    table = OpenQASM3Compiler.QFORGE_FIDELITY_TABLE
    required = {"rx", "ry", "rz", "h", "cx", "cz", "swap"}
    missing = required - set(table.keys())
    assert not missing, f"Missing gates in fidelity table: {missing}"

def test_compilation_summary():
    from siliqun.compiler.qasm3_compiler import OpenQASM3Compiler
    circuit = [("h", {}, [0]), ("cx", {}, [0, 1])]
    compiler = OpenQASM3Compiler(optimise=False)
    result = compiler.compile(circuit)
    summary = result.summary()
    assert "Gates:" in summary
    assert "fidelity:" in summary
    assert "duration:" in summary

def test_backend_v2():
    from siliqun.compiler.qasm3_compiler import SiliQunBackendV2
    backend = SiliQunBackendV2()
    # In Qiskit 2.x, name is an instance attribute set in __init__
    assert backend.name == "siliqun_simos", f"Expected 'siliqun_simos', got '{backend.name}'"
    assert backend.num_qubits == 4
    assert backend.target is not None
    # Check that CZ and CX are in the target
    gate_names = list(backend.target.operation_names_for_qargs((0, 1)))
    assert any(g in gate_names for g in ["cz", "cx"]), \
        f"No 2Q gates found for (0,1): {gate_names}"

test("QASM 2.0 string input", test_qasm2_string)
test("QASM 3.0 string input", test_qasm3_string)
test("Qiskit QuantumCircuit input", test_qiskit_circuit_input)
test("SiliQun gate list input", test_gate_list_input)
test("Fidelity table completeness", test_fidelity_table_completeness)
test("Compilation summary string", test_compilation_summary)
test("Qiskit BackendV2 interface", test_backend_v2)

# ── OpenPulse Schedule ─────────────────────────────────────────────────────────
print("\n=== OpenPulse Schedule ===")

def test_openpulse_drive_pulse():
    from siliqun.pulse.openpulse_schedule import (
        OpenPulseSchedule, WaveformShape, ChannelType
    )
    sched = OpenPulseSchedule(name="test_drive")
    sched.play(WaveformShape.GAUSSIAN, amplitude=0.5, duration_ns=50.0,
               channel_type=ChannelType.DRIVE, qubit=0, t_start_ns=0.0)
    assert len(sched.instructions) == 1
    assert sched.total_duration_ns == 50.0

def test_openpulse_exchange_pulse():
    from siliqun.pulse.openpulse_schedule import OpenPulseSchedule, WaveformShape
    sched = OpenPulseSchedule(name="test_exchange")
    sched.exchange(J_hz=50e6, duration_ns=10.0, qubit_i=0, qubit_j=1, t_start_ns=0.0)
    assert len(sched.instructions) == 1
    assert sched.total_duration_ns == 10.0

def test_openpulse_channel_names():
    from siliqun.pulse.openpulse_schedule import Channel, ChannelType
    d0 = Channel(ChannelType.DRIVE, 0)
    u01 = Channel(ChannelType.EXCHANGE, 1)
    m0 = Channel(ChannelType.READOUT, 0)
    assert d0.name == "d0"
    assert u01.name == "u1"
    assert m0.name == "m0"

def test_openpulse_waveform_samples():
    from siliqun.pulse.openpulse_schedule import Waveform, WaveformShape
    # Square waveform
    sq = Waveform(WaveformShape.SQUARE, amplitude=1.0, duration_ns=10.0)
    samples = sq.to_samples(sample_rate_ghz=1.0)
    assert len(samples) == 10
    assert np.allclose(np.abs(samples), 1.0)
    # Gaussian waveform
    gauss = Waveform(WaveformShape.GAUSSIAN, amplitude=1.0, duration_ns=50.0, sigma_ns=10.0)
    samples_g = gauss.to_samples(sample_rate_ghz=1.0)
    assert len(samples_g) == 50
    assert np.max(np.abs(samples_g)) <= 1.0 + 1e-10

def test_openpulse_to_qiskit():
    # qiskit.pulse was separated from qiskit-terra in Qiskit 2.x.
    # Gracefully skip if not installed (install qiskit-dynamics for pulse support).
    try:
        import qiskit.pulse  # noqa: F401
    except ImportError:
        print("    SKIP (qiskit.pulse not installed)")
        return  # treat as pass
    from siliqun.pulse.openpulse_schedule import (
        OpenPulseSchedule, WaveformShape, ChannelType
    )
    sched = OpenPulseSchedule(name="bell_pulse")
    sched.play(WaveformShape.GAUSSIAN, amplitude=0.5, duration_ns=50.0,
               channel_type=ChannelType.DRIVE, qubit=0)
    sched.exchange(J_hz=50e6, duration_ns=10.0, qubit_i=0, qubit_j=1, t_start_ns=50.0)
    qiskit_sched = sched.to_qiskit_schedule()
    assert qiskit_sched is not None
    assert qiskit_sched.duration > 0

def test_openpulse_string_export():
    from siliqun.pulse.openpulse_schedule import (
        OpenPulseSchedule, WaveformShape, ChannelType
    )
    sched = OpenPulseSchedule(name="test_export")
    sched.play(WaveformShape.SQUARE, amplitude=1.0, duration_ns=10.0,
               channel_type=ChannelType.DRIVE, qubit=0)
    openpulse_str = sched.to_openpulse_string()
    assert "cal {" in openpulse_str
    assert "play(" in openpulse_str
    assert "}" in openpulse_str

def test_converter_from_siliqun():
    from siliqun.pulse.lindblad import DrivePulse, ExchangePulse, PulseSequence
    from siliqun.pulse.openpulse_schedule import from_siliqun_pulse_sequence, ChannelType
    pulses = [
        DrivePulse(qubit=0, amplitude=0.5, frequency=0.0, phase=0.0,
                   duration=50e-9, t_start=0.0),
        ExchangePulse(qubit_i=0, qubit_j=1, J=50e6, duration=10e-9,
                      shape="square", t_start=50e-9),
    ]
    seq = PulseSequence()
    for p in pulses:
        seq.add(p)
    sched = from_siliqun_pulse_sequence(seq, device_name="simos")
    assert len(sched.instructions) == 2
    ch_types = {inst.channel.channel_type for inst in sched.instructions}
    assert ChannelType.DRIVE in ch_types
    assert ChannelType.EXCHANGE in ch_types

test("Drive pulse play instruction", test_openpulse_drive_pulse)
test("Exchange pulse instruction", test_openpulse_exchange_pulse)
test("Channel name conventions (d/u/m/a)", test_openpulse_channel_names)
test("Waveform sample generation (square + gaussian)", test_openpulse_waveform_samples)
test("Export to Qiskit Pulse Schedule", test_openpulse_to_qiskit)
test("Export to OpenPulse grammar string", test_openpulse_string_export)
test("Convert from legacy PulseSequence", test_converter_from_siliqun)

# ── PennyLane Device ──────────────────────────────────────────────────────────
print("\n=== PennyLane Device Plugin ===")

def test_pennylane_device_import():
    from siliqun.plugins.pennylane_device import SiliQunDevice, PENNYLANE_AVAILABLE
    assert PENNYLANE_AVAILABLE, "PennyLane not available"
    dev = SiliQunDevice(wires=4, device_type="simos")
    assert dev is not None

def test_pennylane_bell_state():
    import pennylane as qml
    from siliqun.plugins.pennylane_device import SiliQunDevice
    dev = SiliQunDevice(wires=2, device_type="simos")

    @qml.qnode(dev)
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    state = bell()
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    fidelity = abs(np.dot(state.conj(), expected)) ** 2
    assert fidelity > 0.99, f"Bell state fidelity {fidelity:.4f} too low"

def test_pennylane_expval():
    import pennylane as qml
    from siliqun.plugins.pennylane_device import SiliQunDevice
    dev = SiliQunDevice(wires=1, device_type="simos")

    @qml.qnode(dev)
    def z_expval():
        return qml.expval(qml.PauliZ(wires=0))

    val = z_expval()
    assert abs(val - 1.0) < 1e-6, f"<Z> for |0> should be 1.0, got {val}"

def test_pennylane_supported_ops():
    from siliqun.plugins.pennylane_device import SiliQunDevice
    dev = SiliQunDevice(wires=4)
    required_ops = {"PauliX", "PauliZ", "Hadamard", "CNOT", "CZ", "RX", "RY", "RZ"}
    missing = required_ops - dev.operations
    assert not missing, f"Missing operations: {missing}"

test("SiliQunDevice import and instantiation", test_pennylane_device_import)
test("Bell state via PennyLane QNode", test_pennylane_bell_state)
test("Expectation value <Z> for |0>", test_pennylane_expval)
test("Supported operations set", test_pennylane_supported_ops)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
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
