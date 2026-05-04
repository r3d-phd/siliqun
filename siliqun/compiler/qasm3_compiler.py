"""
SiliQun OpenQASM 3.0 Compiler
==============================
Parses OpenQASM 3.0 circuits natively (via Qiskit's qasm3 module) and
compiles them into SiliQun pulse schedules for silicon spin qubit devices.

This module implements the standards-based circuit entry point for the
generalised SiliQun platform. It replaces the custom tuple-based circuit
representation with the OpenQASM 3.0 industry standard.

Standards:
    - Circuit format: OpenQASM 3.0 (https://openqasm.com/versions/3.0/)
    - Compilation: Qiskit transpiler (BackendV2 interface)
    - Pulse output: OpenPulse-compatible PulseSchedule

References:
    [1] Cross et al., "OpenQASM 3: A broader and deeper quantum assembly language",
        ACM Transactions on Quantum Computing, 2022.
    [2] Qiskit SDK v2.0 documentation, https://docs.quantum.ibm.com/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Qiskit imports ─────────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit, qasm3, transpile
    from qiskit.circuit import Gate as QiskitGate
    from qiskit.providers import BackendV2
    from qiskit.transpiler import Target, InstructionProperties
    from qiskit.circuit.library import (
        RXGate, RYGate, RZGate, CXGate, CZGate, HGate, SGate, TGate,
        XGate, YGate, ZGate, SwapGate, IGate,
    )
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. OpenQASM 3.0 parsing will be limited.")

# ── SiliQun imports ────────────────────────────────────────────────────────────
from ..physics.devices.profiles import DeviceProfile, simos_device
from ..pulse.lindblad import DrivePulse, ExchangePulse, PulseSequence


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CompiledPulseSchedule:
    """Result of compiling an OpenQASM 3.0 circuit to a SiliQun pulse schedule.

    Attributes
    ----------
    pulse_sequence : PulseSequence
        The compiled pulse schedule ready for simulation.
    total_fidelity : float
        Estimated total gate fidelity (product of per-gate fidelities).
    n_gates : int
        Total number of gates in the compiled circuit.
    n_1q_gates : int
        Number of single-qubit gates.
    n_2q_gates : int
        Number of two-qubit gates.
    total_duration_ns : float
        Total pulse schedule duration in nanoseconds.
    gate_fidelities : dict
        Per-gate fidelity estimates keyed by gate label.
    qasm3_source : str
        The original OpenQASM 3.0 source string.
    qiskit_circuit : QuantumCircuit, optional
        The parsed Qiskit QuantumCircuit object.
    """
    pulse_sequence: PulseSequence
    total_fidelity: float
    n_gates: int
    n_1q_gates: int
    n_2q_gates: int
    total_duration_ns: float
    gate_fidelities: Dict[str, float] = field(default_factory=dict)
    qasm3_source: str = ""
    qiskit_circuit: Optional[object] = None  # QuantumCircuit

    def summary(self) -> str:
        """Return a human-readable compilation summary."""
        return (
            f"CompiledPulseSchedule:\n"
            f"  Gates: {self.n_gates} ({self.n_1q_gates} 1Q + {self.n_2q_gates} 2Q)\n"
            f"  Total fidelity: {self.total_fidelity:.4f}\n"
            f"  Total duration: {self.total_duration_ns:.1f} ns\n"
            f"  Pulses: {len(self.pulse_sequence.pulses)}"
        )


# ── SiliQun BackendV2 (Qiskit standard interface) ──────────────────────────────

class SiliQunBackendV2(BackendV2):
    """A Qiskit BackendV2-compatible backend for silicon spin qubit simulation.

    This class exposes SiliQun as a standard Qiskit backend, allowing users
    to run any Qiskit QuantumCircuit on SiliQun's physics-accurate simulator
    without modifying their existing Qiskit code.

    Parameters
    ----------
    device : DeviceProfile, optional
        SiliQun device profile. Defaults to SiMOS 4-qubit.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> from siliqun.compiler.qasm3_compiler import SiliQunBackendV2
    >>> backend = SiliQunBackendV2()
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0); qc.cx(0, 1)
    >>> job = backend.run(qc)
    >>> result = job.result()
    """
    # Note: do NOT define 'name' as a class attribute here.
    # In Qiskit 2.x, BackendV2.name is set via __init__ as an instance attribute.

    def __init__(self, device: Optional[DeviceProfile] = None):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for SiliQunBackendV2. "
                              "Install with: pip install qiskit")
        super().__init__(name="siliqun_simos", description="SiliQun silicon spin qubit simulator")
        self._device = device or simos_device(n_qubits=4)
        self._target = self._build_target()

    def _build_target(self) -> "Target":
        """Build the Qiskit Target from the SiliQun device profile."""
        target = Target(
            description=f"SiliQun {self._device.name} device",
            num_qubits=self._device.n_qubits,
        )
        n = self._device.n_qubits
        gt = self._device.gate_times

        # Single-qubit gates on all qubits
        for qubit in range(n):
            t1q = gt.get("rx", gt.get("single_qubit", 50e-9))
            props = InstructionProperties(duration=t1q, error=1e-3)
            for gate_cls in [RXGate(0.0), RYGate(0.0), RZGate(0.0), HGate(), XGate(), YGate(), ZGate()]:
                try:
                    target.add_instruction(gate_cls, {(qubit,): props})
                except Exception:
                    pass

        # Two-qubit gates on connected pairs
        t2q = gt.get("cz", gt.get("two_qubit", 200e-9))
        for (i, j) in self._device.connectivity:
            props2q = InstructionProperties(duration=t2q, error=5e-3)
            for gate_cls in [CZGate(), CXGate()]:
                try:
                    target.add_instruction(gate_cls, {(i, j): props2q})
                except Exception:
                    pass

        return target

    @property
    def target(self) -> "Target":
        """Return the Qiskit Target for this backend (required by BackendV2)."""
        return self._target

    @property
    def max_circuits(self) -> int:
        return 100

    @classmethod
    def _default_options(cls):
        from qiskit.providers import Options
        return Options(shots=1024, sim_mode="auto")

    def run(self, circuits, **kwargs):
        """Execute circuits on the SiliQun simulator.

        Parameters
        ----------
        circuits : QuantumCircuit or list of QuantumCircuit
            Circuits to execute.

        Returns
        -------
        SiliQunJob
            A job object with a `.result()` method.
        """
        if not isinstance(circuits, list):
            circuits = [circuits]
        return SiliQunJob(backend=self, circuits=circuits,
                          device=self._device, **kwargs)


# ── Minimal Job class ──────────────────────────────────────────────────────────

class SiliQunJob:
    """Minimal Qiskit-compatible job object for SiliQun circuit execution."""

    def __init__(self, backend, circuits, device, shots=1024, sim_mode="auto", **kwargs):
        self._backend = backend
        self._circuits = circuits
        self._device = device
        self._shots = shots
        self._sim_mode = sim_mode
        self._results = None

    def result(self):
        """Execute and return results."""
        if self._results is None:
            self._results = self._execute()
        return self._results

    def _execute(self):
        """Run circuits through SiliQun's simulation engine."""
        from ..engine.simulator import SiliQunSimulator
        results = []
        sim = SiliQunSimulator(device=self._device, sim_mode=self._sim_mode)
        for qc in self._circuits:
            # Convert Qiskit circuit to SiliQun gate list
            gates = _qiskit_circuit_to_gate_list(qc)
            state = sim.run_circuit(gates)
            results.append({
                "statevector": state,
                "n_qubits": qc.num_qubits,
                "circuit_name": qc.name,
            })
        return SiliQunResult(results)


class SiliQunResult:
    """Minimal result container compatible with Qiskit's Result interface."""

    def __init__(self, results: list):
        self._results = results

    def get_statevector(self, idx: int = 0) -> np.ndarray:
        return self._results[idx]["statevector"]

    def __repr__(self):
        return f"SiliQunResult({len(self._results)} circuits)"


# ── OpenQASM 3.0 Compiler ─────────────────────────────────────────────────────

class OpenQASM3Compiler:
    """Compiles OpenQASM 3.0 circuits to SiliQun pulse schedules.

    This is the primary standards-based entry point for the generalised
    SiliQun platform. It accepts OpenQASM 3.0 strings, Qiskit QuantumCircuits,
    or the custom SiliQun gate list format, and produces a pulse schedule
    optimised for the target silicon spin qubit device.

    Parameters
    ----------
    device : DeviceProfile, optional
        Target device profile. Defaults to SiMOS 4-qubit.
    optimise : bool
        Whether to apply Qiskit transpiler optimisations before compilation.
        Default True.
    optimisation_level : int
        Qiskit transpiler optimisation level (0–3). Default 2.

    Examples
    --------
    >>> compiler = OpenQASM3Compiler()
    >>> qasm = '''
    ... OPENQASM 3.0;
    ... qubit[2] q;
    ... h q[0];
    ... cx q[0], q[1];
    ... '''
    >>> schedule = compiler.compile(qasm)
    >>> print(schedule.summary())
    """

    # Gate fidelity table from Q-Forge calibration data (SiMOS device, T=20 mK)
    # Source: Q-Forge forecast_fidelity results, May 2026
    QFORGE_FIDELITY_TABLE: Dict[str, float] = {
        "rx": 0.9992, "ry": 0.9992, "rz": 0.9998,
        "h": 0.9990, "x": 0.9991, "y": 0.9991, "z": 0.9998,
        "s": 0.9995, "t": 0.9993, "sdg": 0.9995, "tdg": 0.9993,
        "cx": 0.9964, "cnot": 0.9964, "cz": 0.9976,
        "swap": 0.9921, "iswap": 0.9918,
        "sqrt_swap": 0.9961, "sqrt_iswap": 0.9958,
        "id": 1.0000, "barrier": 1.0000, "measure": 0.9900,
    }

    def __init__(
        self,
        device: Optional[DeviceProfile] = None,
        optimise: bool = True,
        optimisation_level: int = 2,
    ):
        self._device = device or simos_device(n_qubits=4)
        self._optimise = optimise and QISKIT_AVAILABLE
        self._opt_level = optimisation_level
        self._backend = SiliQunBackendV2(device=self._device) if QISKIT_AVAILABLE else None

    # ── Public API ─────────────────────────────────────────────────────────────

    def compile(
        self,
        circuit: Union[str, "QuantumCircuit", List],
        *,
        shots: int = 1024,
    ) -> CompiledPulseSchedule:
        """Compile a circuit to a SiliQun pulse schedule.

        Parameters
        ----------
        circuit : str, QuantumCircuit, or list
            - str: OpenQASM 3.0 or OpenQASM 2.0 source string
            - QuantumCircuit: Qiskit circuit object
            - list: SiliQun gate list [(gate_name, params, qubits), ...]
        shots : int
            Number of shots for simulation (used by the REST API).

        Returns
        -------
        CompiledPulseSchedule
            Compiled pulse schedule with fidelity estimates.
        """
        qasm3_source = ""

        if isinstance(circuit, str):
            qasm3_source = circuit
            qc = self._parse_qasm(circuit)
        elif QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
            qc = circuit
            try:
                qasm3_source = qasm3.dumps(qc)
            except Exception:
                qasm3_source = qc.qasm() if hasattr(qc, 'qasm') else ""
        elif isinstance(circuit, list):
            # Legacy SiliQun gate list format
            qc = self._gate_list_to_qiskit(circuit) if QISKIT_AVAILABLE else None
            if qc is not None:
                try:
                    qasm3_source = qasm3.dumps(qc)
                except Exception:
                    pass
        else:
            raise TypeError(f"Unsupported circuit type: {type(circuit)}")

        # Optionally transpile to device-native gates
        if self._optimise and qc is not None and QISKIT_AVAILABLE:
            try:
                qc = transpile(qc, backend=self._backend,
                               optimization_level=self._opt_level)
            except Exception as e:
                logger.warning("Transpilation failed: %s. Using original circuit.", e)

        # Convert to gate list and build pulse schedule
        if qc is not None and QISKIT_AVAILABLE:
            gate_list = _qiskit_circuit_to_gate_list(qc)
        elif isinstance(circuit, list):
            gate_list = circuit
        else:
            gate_list = []

        return self._build_pulse_schedule(gate_list, qasm3_source=qasm3_source, qc=qc)

    def compile_qasm2(self, qasm2_str: str) -> CompiledPulseSchedule:
        """Compile an OpenQASM 2.0 string (backward compatibility)."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for QASM 2.0 parsing.")
        qc = QuantumCircuit.from_qasm_str(qasm2_str)
        return self.compile(qc)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _parse_qasm(self, qasm_str: str) -> Optional["QuantumCircuit"]:
        """Parse an OpenQASM 2.0 or 3.0 string into a Qiskit QuantumCircuit."""
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available — skipping QASM parse.")
            return None
        qasm_str = qasm_str.strip()
        try:
            if qasm_str.startswith("OPENQASM 3") or qasm_str.startswith("OPENQASM 3.0"):
                # Auto-inject stdgates.inc if not already present
                # (required for standard gate names like h, cx, rz in QASM 3.0)
                if 'stdgates.inc' not in qasm_str and 'include' not in qasm_str:
                    lines = qasm_str.split('\n')
                    insert_idx = next(
                        (i + 1 for i, l in enumerate(lines) if l.strip().startswith('OPENQASM')),
                        1
                    )
                    lines.insert(insert_idx, 'include "stdgates.inc";')
                    qasm_str = '\n'.join(lines)
                return qasm3.loads(qasm_str)
            else:
                return QuantumCircuit.from_qasm_str(qasm_str)
        except Exception as e:
            logger.error("QASM parse error: %s", e)
            raise ValueError(f"Failed to parse QASM circuit: {e}") from e

    def _gate_list_to_qiskit(self, gate_list: List) -> "QuantumCircuit":
        """Convert a SiliQun gate list to a Qiskit QuantumCircuit."""
        n_qubits = max(
            max(q for q in qubits) for _, _, qubits in gate_list
            if qubits
        ) + 1
        qc = QuantumCircuit(n_qubits)
        for gate_name, params, qubits in gate_list:
            gate_name = gate_name.lower()
            try:
                if gate_name in ("rx",):
                    qc.rx(params.get("theta", np.pi / 2), qubits[0])
                elif gate_name in ("ry",):
                    qc.ry(params.get("theta", np.pi / 2), qubits[0])
                elif gate_name in ("rz",):
                    qc.rz(params.get("theta", np.pi / 2), qubits[0])
                elif gate_name in ("h",):
                    qc.h(qubits[0])
                elif gate_name in ("x",):
                    qc.x(qubits[0])
                elif gate_name in ("y",):
                    qc.y(qubits[0])
                elif gate_name in ("z",):
                    qc.z(qubits[0])
                elif gate_name in ("s",):
                    qc.s(qubits[0])
                elif gate_name in ("t",):
                    qc.t(qubits[0])
                elif gate_name in ("cx", "cnot"):
                    qc.cx(qubits[0], qubits[1])
                elif gate_name in ("cz",):
                    qc.cz(qubits[0], qubits[1])
                elif gate_name in ("swap",):
                    qc.swap(qubits[0], qubits[1])
                else:
                    logger.warning("Unknown gate '%s' — skipping.", gate_name)
            except Exception as e:
                logger.warning("Gate '%s' on qubits %s failed: %s", gate_name, qubits, e)
        return qc

    def _build_pulse_schedule(
        self,
        gate_list: List,
        qasm3_source: str = "",
        qc: Optional["QuantumCircuit"] = None,
    ) -> CompiledPulseSchedule:
        """Build a SiliQun PulseSequence from a gate list."""
        pulses: List[Union[DrivePulse, ExchangePulse]] = []
        gate_fidelities: Dict[str, float] = {}
        total_fidelity = 1.0
        n_1q = 0
        n_2q = 0
        t_cursor = 0.0  # seconds

        gt = self._device.gate_times
        J_max = getattr(self._device, "J_max_hz", 50e6)

        for gate_name, params, qubits in gate_list:
            gname = gate_name.lower()
            fidelity = self.QFORGE_FIDELITY_TABLE.get(gname, 0.995)
            label = f"{gname}@q{qubits}"

            if len(qubits) == 1:
                # Single-qubit gate → drive pulse
                duration = gt.get(gname, gt.get("rx", 50e-9))
                theta = params.get("theta", np.pi / 2) if params else np.pi / 2
                phase = params.get("phi", 0.0) if params else 0.0
                pulses.append(DrivePulse(
                    qubit=qubits[0],
                    amplitude=theta / (2 * np.pi * duration) if duration > 0 else 0,
                    frequency=0.0,
                    phase=phase,
                    duration=duration,
                    t_start=t_cursor,
                ))
                t_cursor += duration
                n_1q += 1

            elif len(qubits) == 2:
                # Two-qubit gate → exchange pulse
                duration = gt.get(gname, gt.get("cz", 200e-9))
                # CZ: θ = π → J·t = π/2 → t = π/(2J)
                if gname in ("cz",):
                    J = J_max
                    t_ex = np.pi / (2 * J) if J > 0 else duration
                elif gname in ("cx", "cnot"):
                    # CNOT = Ry(π/2) · CZ · Ry(-π/2) on target
                    t_ex = np.pi / (2 * J_max) if J_max > 0 else duration
                    # Pre-rotation on target qubit
                    pulses.append(DrivePulse(
                        qubit=qubits[1], amplitude=0.5 / duration,
                        frequency=0.0, phase=np.pi / 2,
                        duration=gt.get("ry", 50e-9), t_start=t_cursor,
                    ))
                    t_cursor += gt.get("ry", 50e-9)
                else:
                    J = J_max
                    t_ex = duration

                pulses.append(ExchangePulse(
                    qubit_i=qubits[0], qubit_j=qubits[1],
                    J=J_max, duration=t_ex, shape="square",
                    t_start=t_cursor,
                ))
                t_cursor += t_ex

                if gname in ("cx", "cnot"):
                    # Post-rotation on target qubit
                    pulses.append(DrivePulse(
                        qubit=qubits[1], amplitude=0.5 / duration,
                        frequency=0.0, phase=-np.pi / 2,
                        duration=gt.get("ry", 50e-9), t_start=t_cursor,
                    ))
                    t_cursor += gt.get("ry", 50e-9)
                n_2q += 1

            gate_fidelities[label] = fidelity
            total_fidelity *= fidelity

        pulse_seq = PulseSequence()
        for p in pulses:
            pulse_seq.add(p)

        return CompiledPulseSchedule(
            pulse_sequence=pulse_seq,
            total_fidelity=total_fidelity,
            n_gates=n_1q + n_2q,
            n_1q_gates=n_1q,
            n_2q_gates=n_2q,
            total_duration_ns=t_cursor * 1e9,
            gate_fidelities=gate_fidelities,
            qasm3_source=qasm3_source,
            qiskit_circuit=qc,
        )


# ── Utility: Qiskit circuit → SiliQun gate list ────────────────────────────────

def _qiskit_circuit_to_gate_list(qc: "QuantumCircuit") -> List[Tuple]:
    """Convert a Qiskit QuantumCircuit to a SiliQun gate list.

    Parameters
    ----------
    qc : QuantumCircuit
        Qiskit circuit to convert.

    Returns
    -------
    list of (gate_name, params, qubits)
    """
    gate_list = []
    for instruction in qc.data:
        op = instruction.operation
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        params = {}
        if op.params:
            param_names = ["theta", "phi", "lam"]
            for i, p in enumerate(op.params):
                if i < len(param_names):
                    try:
                        params[param_names[i]] = float(p)
                    except (TypeError, ValueError):
                        pass
        gate_list.append((op.name.lower(), params, qubits))
    return gate_list
