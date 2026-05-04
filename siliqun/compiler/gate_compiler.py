"""
Gate-to-Pulse Compiler for Silicon Spin Qubits.

Translates abstract quantum gate sequences (OpenQASM-style tuples) into
physically realisable pulse sequences for silicon spin qubit devices.

Gate decompositions are based on the exchange interaction as the native
two-qubit primitive, following the Loss-DiVincenzo (1998) model:

    H_exchange = (2*pi*J/4) * (XX + YY + ZZ)

where J is the exchange coupling in Hz.

Fidelity estimates for each compiled gate are provided via Q-Forge's
analytical error accumulation model, using device-calibrated noise levels.

Supported gates:
    Single-qubit: rx, ry, rz, x, y, z, h, s, t
    Two-qubit:    cnot, cx, cz, swap, sqrt_swap, exchange, iswap

References:
    Loss & DiVincenzo, Phys. Rev. A 57, 120 (1998)
    Vatan & Williams, Phys. Rev. A 69, 032315 (2004)
    Burkard et al., Rev. Mod. Phys. 95, 025003 (2023)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..physics.devices.profiles import DeviceProfile, simos_device
from ..pulse.lindblad import DrivePulse, ExchangePulse, PulseSequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate fidelity table (Q-Forge validated, SiMOS noise level 0.3%)
# ---------------------------------------------------------------------------

# Predicted fidelities from Q-Forge forecast_fidelity at backend_noise=0.003
# (SiMOS two-qubit gate error rate ~0.3%, consistent with Tanttu et al. 2024)
QFORGE_FIDELITY_TABLE: Dict[str, float] = {
    "rx": 0.9997,    # Single-qubit: ~0.03% error
    "ry": 0.9997,
    "rz": 0.9999,    # Virtual Z: near-perfect (software gate)
    "x": 0.9997,
    "y": 0.9997,
    "z": 0.9999,
    "h": 0.9994,     # H = Rz(pi/2) * Ry(pi/2)
    "s": 0.9999,
    "t": 0.9999,
    "cnot": 0.9964,  # Q-Forge forecast: 0.996406 at noise=0.003
    "cx": 0.9964,
    "cz": 0.9976,    # Q-Forge forecast: 0.997603 at noise=0.003
    "swap": 0.9893,  # 3 CNOTs: 0.9964^3
    "sqrt_swap": 0.9982,
    "exchange": 0.9982,
    "iswap": 0.9929,  # 2 sqrt_SWAPs
}


# ---------------------------------------------------------------------------
# Compiled gate result
# ---------------------------------------------------------------------------

@dataclass
class CompiledGate:
    """Result of compiling a single gate into pulses.

    Attributes
    ----------
    gate_name : str
        Original gate name.
    qubits : list of int
        Target qubit indices.
    pulses : list
        List of ExchangePulse and/or DrivePulse objects.
    predicted_fidelity : float
        Q-Forge predicted fidelity for this gate at the device noise level.
    t_start : float
        Start time of this gate in the full sequence (seconds).
    duration : float
        Total duration of this gate's pulses (seconds).
    """
    gate_name: str
    qubits: List[int]
    pulses: List
    predicted_fidelity: float
    t_start: float = 0.0
    duration: float = 0.0


@dataclass
class CompilationResult:
    """Result of compiling a full circuit into a pulse sequence.

    Attributes
    ----------
    sequence : PulseSequence
        The compiled pulse sequence ready for LindbladSimulator.
    compiled_gates : list of CompiledGate
        Per-gate compilation results with fidelity estimates.
    total_fidelity : float
        Product of all per-gate fidelities (upper bound on circuit fidelity).
    total_duration : float
        Total circuit duration in seconds.
    n_exchange_pulses : int
        Total number of exchange pulses in the sequence.
    n_drive_pulses : int
        Total number of drive pulses in the sequence.
    warnings : list of str
        Any compilation warnings (e.g., non-native gates, low fidelity).
    """
    sequence: PulseSequence
    compiled_gates: List[CompiledGate]
    total_fidelity: float
    total_duration: float
    n_exchange_pulses: int
    n_drive_pulses: int
    warnings: List[str] = field(default_factory=list)

    def fidelity_breakdown(self) -> Dict[str, float]:
        """Return per-gate fidelity breakdown."""
        return {
            f"{g.gate_name}@q{g.qubits}": g.predicted_fidelity
            for g in self.compiled_gates
        }


# ---------------------------------------------------------------------------
# Gate-to-Pulse Compiler
# ---------------------------------------------------------------------------

class GateToPulseCompiler:
    """Compiles abstract gate sequences into silicon spin qubit pulse sequences.

    Uses the exchange interaction as the native two-qubit primitive and
    microwave drives (ESR/EDSR) for single-qubit rotations.

    Parameters
    ----------
    device : DeviceProfile
        Silicon spin qubit device profile. Provides:
            - J_max_hz : Maximum exchange coupling (Hz)
            - gate_time_1q : Single-qubit gate time (seconds)
            - gate_time_2q : Two-qubit gate time (seconds)
            - noise_params : T1, T2*, charge noise parameters
    schedule_mode : str
        "sequential" (default): gates execute one after another.
        "parallel": gates on non-overlapping qubits execute simultaneously.

    Examples
    --------
    >>> from siliqun.physics.devices.profiles import get_device
    >>> from siliqun.compiler import GateToPulseCompiler
    >>>
    >>> device = get_device("simos", n_qubits=4)
    >>> compiler = GateToPulseCompiler(device)
    >>>
    >>> circuit = [
    ...     ("h", {}, [0]),
    ...     ("cnot", {}, [0, 1]),
    ...     ("cnot", {}, [1, 2]),
    ...     ("cnot", {}, [2, 3]),
    ... ]
    >>> result = compiler.compile(circuit)
    >>> print(f"Total fidelity: {result.total_fidelity:.4f}")
    >>> print(f"Total duration: {result.total_duration*1e9:.1f} ns")
    """

    def __init__(
        self,
        device: Optional[DeviceProfile] = None,
        schedule_mode: str = "sequential",
    ):
        if device is None:
            # Default: SiMOS device profile (4 qubits)
            device = simos_device(n_qubits=4)
        self.device = device
        self.schedule_mode = schedule_mode

        # Extract device parameters
        self._J_max = getattr(device, "J_max_hz", 50e6)       # Hz, default 50 MHz
        self._t1q = getattr(device, "gate_time_1q", 50e-9)    # s, default 50 ns
        self._t2q = getattr(device, "gate_time_2q", 200e-9)   # s, default 200 ns
        self._noise_level = self._estimate_noise_level()

        logger.info(
            "GateToPulseCompiler: device=%s, J_max=%.1f MHz, "
            "t1q=%.0f ns, t2q=%.0f ns, noise=%.4f",
            getattr(device, "name", "unknown"),
            self._J_max / 1e6,
            self._t1q * 1e9,
            self._t2q * 1e9,
            self._noise_level,
        )

    def _estimate_noise_level(self) -> float:
        """Estimate the effective per-gate noise level from device T2* and gate time."""
        try:
            T2 = self.device.noise_params.T2_star
            t2q = self._t2q
            # Rough estimate: error ~ t_gate / T2*
            return min(0.05, t2q / T2)
        except AttributeError:
            return 0.003  # Default SiMOS noise level

    # ------------------------------------------------------------------
    # Main compile method
    # ------------------------------------------------------------------

    def compile(
        self,
        circuit: List[Tuple],
        initial_time: float = 0.0,
    ) -> CompilationResult:
        """Compile a gate circuit into a pulse sequence.

        Parameters
        ----------
        circuit : list of tuple
            Each tuple is (gate_name, params, qubits).
            e.g., [("rx", {"theta": 1.57}, [0]),
                   ("cnot", {}, [0, 1]),
                   ("h", {}, [2])]
        initial_time : float
            Start time offset for the entire sequence (seconds).

        Returns
        -------
        CompilationResult
            Compiled pulse sequence with per-gate fidelity estimates.
        """
        sequence = PulseSequence()
        compiled_gates = []
        warnings = []
        t_current = initial_time

        for gate_name, params, qubits in circuit:
            gate_name_lower = gate_name.lower()

            # Compile this gate
            try:
                pulses, duration = self._compile_gate(
                    gate_name_lower, params, qubits, t_current
                )
            except ValueError as e:
                warnings.append(f"Gate '{gate_name}' on q{qubits}: {e}")
                logger.warning("Skipping unsupported gate: %s on %s", gate_name, qubits)
                continue

            # Get predicted fidelity
            fidelity = QFORGE_FIDELITY_TABLE.get(gate_name_lower, 0.99)
            # Scale fidelity by device noise relative to SiMOS baseline
            noise_scale = self._noise_level / 0.003
            if gate_name_lower in ("rz", "z", "s", "t"):
                pass  # Virtual Z gates are noise-free
            else:
                fidelity = max(0.5, fidelity ** noise_scale)

            if fidelity < 0.95:
                warnings.append(
                    f"Low fidelity ({fidelity:.3f}) for gate '{gate_name}' "
                    f"on q{qubits} — device noise level {self._noise_level:.4f}"
                )

            # Add pulses to sequence
            for p in pulses:
                sequence.add(p)

            compiled_gates.append(CompiledGate(
                gate_name=gate_name_lower,
                qubits=qubits,
                pulses=pulses,
                predicted_fidelity=fidelity,
                t_start=t_current,
                duration=duration,
            ))

            t_current += duration

        # Compute total fidelity (product of per-gate fidelities)
        total_fidelity = 1.0
        for g in compiled_gates:
            total_fidelity *= g.predicted_fidelity

        n_exchange = sum(
            1 for p in sequence.pulses if isinstance(p, ExchangePulse)
        )
        n_drive = sum(
            1 for p in sequence.pulses if isinstance(p, DrivePulse)
        )

        logger.info(
            "Compiled %d gates: total_fidelity=%.4f, duration=%.1f ns, "
            "exchange=%d, drive=%d",
            len(compiled_gates), total_fidelity,
            (t_current - initial_time) * 1e9,
            n_exchange, n_drive,
        )

        return CompilationResult(
            sequence=sequence,
            compiled_gates=compiled_gates,
            total_fidelity=total_fidelity,
            total_duration=t_current - initial_time,
            n_exchange_pulses=n_exchange,
            n_drive_pulses=n_drive,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Individual gate compilers
    # ------------------------------------------------------------------

    def _compile_gate(
        self,
        gate_name: str,
        params: Dict,
        qubits: List[int],
        t_start: float,
    ) -> Tuple[List, float]:
        """Compile a single gate. Returns (pulses, duration)."""

        # --- Single-qubit rotations ---
        if gate_name == "rx":
            return self._rx(params["theta"], qubits[0], t_start)
        elif gate_name == "ry":
            return self._ry(params["theta"], qubits[0], t_start)
        elif gate_name == "rz":
            return self._rz(params["theta"], qubits[0], t_start)
        elif gate_name == "x":
            return self._rx(math.pi, qubits[0], t_start)
        elif gate_name == "y":
            return self._ry(math.pi, qubits[0], t_start)
        elif gate_name == "z":
            return self._rz(math.pi, qubits[0], t_start)
        elif gate_name == "h":
            return self._hadamard(qubits[0], t_start)
        elif gate_name == "s":
            return self._rz(math.pi / 2, qubits[0], t_start)
        elif gate_name == "t":
            return self._rz(math.pi / 4, qubits[0], t_start)
        elif gate_name == "sdg":
            return self._rz(-math.pi / 2, qubits[0], t_start)
        elif gate_name == "tdg":
            return self._rz(-math.pi / 4, qubits[0], t_start)

        # --- Two-qubit gates ---
        elif gate_name in ("cnot", "cx"):
            return self._cnot(qubits[0], qubits[1], t_start)
        elif gate_name == "cz":
            return self._cz(qubits[0], qubits[1], t_start)
        elif gate_name == "swap":
            return self._swap(qubits[0], qubits[1], t_start)
        elif gate_name == "sqrt_swap":
            return self._sqrt_swap(qubits[0], qubits[1], t_start)
        elif gate_name == "exchange":
            J = params.get("J", self._J_max)
            t = params.get("t", self._t2q)
            return self._exchange(qubits[0], qubits[1], J, t, t_start)
        elif gate_name == "iswap":
            return self._iswap(qubits[0], qubits[1], t_start)

        else:
            raise ValueError(f"Unsupported gate: {gate_name}")

    # ------------------------------------------------------------------
    # Single-qubit gate implementations
    # ------------------------------------------------------------------

    def _rx(self, theta: float, qubit: int, t_start: float):
        """Rx(theta) via ESR/EDSR microwave drive on X-axis."""
        p = DrivePulse(
            qubit=qubit,
            amplitude=abs(theta) / (2 * math.pi * self._t1q),
            frequency=0.0,  # Rotating frame
            phase=0.0,      # X-axis rotation
            duration=self._t1q,
            t_start=t_start,
        )
        return [p], self._t1q

    def _ry(self, theta: float, qubit: int, t_start: float):
        """Ry(theta) via ESR/EDSR microwave drive on Y-axis."""
        p = DrivePulse(
            qubit=qubit,
            amplitude=abs(theta) / (2 * math.pi * self._t1q),
            frequency=0.0,
            phase=math.pi / 2,  # Y-axis rotation
            duration=self._t1q,
            t_start=t_start,
        )
        return [p], self._t1q

    def _rz(self, theta: float, qubit: int, t_start: float):
        """Rz(theta) as a virtual Z gate (zero duration, no physical pulse).

        Virtual Z gates are implemented by updating the reference frame of
        subsequent pulses. They are noise-free and instantaneous.
        Ref: McKay et al., Phys. Rev. Applied 6, 064007 (2016)

        Note: In the current implementation, we use a zero-duration DrivePulse
        as a placeholder. A full implementation would track the frame offset.
        """
        # Virtual Z: zero-duration placeholder
        p = DrivePulse(
            qubit=qubit,
            amplitude=0.0,
            frequency=0.0,
            phase=theta,
            duration=1e-12,  # Effectively zero
            t_start=t_start,
        )
        return [p], 1e-12

    def _hadamard(self, qubit: int, t_start: float):
        """H = Rz(pi/2) * Ry(pi/2) decomposition."""
        pulses = []
        t = t_start

        # Rz(pi/2) — virtual Z
        rz_pulses, rz_dur = self._rz(math.pi / 2, qubit, t)
        pulses.extend(rz_pulses)
        t += rz_dur

        # Ry(pi/2) — physical drive
        ry_pulses, ry_dur = self._ry(math.pi / 2, qubit, t)
        pulses.extend(ry_pulses)
        t += ry_dur

        return pulses, t - t_start

    # ------------------------------------------------------------------
    # Two-qubit gate implementations
    # ------------------------------------------------------------------

    def _exchange(
        self,
        qubit_i: int,
        qubit_j: int,
        J: float,
        duration: float,
        t_start: float,
    ):
        """Direct exchange pulse with specified J and duration."""
        p = ExchangePulse(
            qubit_i=qubit_i,
            qubit_j=qubit_j,
            J=J,
            duration=duration,
            shape="square",
            t_start=t_start,
        )
        return [p], duration

    def _sqrt_swap(self, qubit_i: int, qubit_j: int, t_start: float):
        """sqrt-SWAP via exchange at theta = pi/2 (J*t = 1/4).

        Exchange angle: theta = 2*pi*J*t = pi/2
        => J*t = 1/4
        => t = 1/(4*J)
        """
        duration = 1.0 / (4.0 * self._J_max)
        p = ExchangePulse(
            qubit_i=qubit_i,
            qubit_j=qubit_j,
            J=self._J_max,
            duration=duration,
            shape="square",
            t_start=t_start,
        )
        return [p], duration

    def _cz(self, qubit_i: int, qubit_j: int, t_start: float):
        """CZ gate via exchange at theta = pi (J*t = 1/2), plus Z corrections.

        Decomposition:
            CZ = Rz(-pi/2) ⊗ Rz(-pi/2) · Exchange(theta=pi) · Rz(pi/2) ⊗ Rz(pi/2)

        Ref: Vatan & Williams (2004), Phys. Rev. A 69, 032315
        """
        pulses = []
        t = t_start

        # Pre-rotation: Rz(pi/2) on both qubits (virtual Z, zero duration)
        for q in [qubit_i, qubit_j]:
            rz_p, rz_d = self._rz(math.pi / 2, q, t)
            pulses.extend(rz_p)

        # Exchange pulse at theta=pi: duration = 1/(2*J_max)
        duration_ex = 1.0 / (2.0 * self._J_max)
        ex_p = ExchangePulse(
            qubit_i=qubit_i,
            qubit_j=qubit_j,
            J=self._J_max,
            duration=duration_ex,
            shape="square",
            t_start=t,
        )
        pulses.append(ex_p)
        t += duration_ex

        # Post-rotation: Rz(-pi/2) on both qubits (virtual Z)
        for q in [qubit_i, qubit_j]:
            rz_p, rz_d = self._rz(-math.pi / 2, q, t)
            pulses.extend(rz_p)

        return pulses, t - t_start

    def _cnot(self, control: int, target: int, t_start: float):
        """CNOT via CZ + Hadamard decomposition.

        CNOT(c,t) = (I ⊗ H) · CZ(c,t) · (I ⊗ H)

        Ref: Nielsen & Chuang, Quantum Computation and Quantum Information (2000)
        """
        pulses = []
        t = t_start

        # H on target qubit
        h_pulses, h_dur = self._hadamard(target, t)
        pulses.extend(h_pulses)
        t += h_dur

        # CZ
        cz_pulses, cz_dur = self._cz(control, target, t)
        pulses.extend(cz_pulses)
        t += cz_dur

        # H on target qubit
        h_pulses, h_dur = self._hadamard(target, t)
        pulses.extend(h_pulses)
        t += h_dur

        return pulses, t - t_start

    def _swap(self, qubit_i: int, qubit_j: int, t_start: float):
        """SWAP = 3 × CNOT decomposition."""
        pulses = []
        t = t_start

        for _ in range(3):
            cnot_p, cnot_d = self._cnot(qubit_i, qubit_j, t)
            pulses.extend(cnot_p)
            t += cnot_d
            # Alternate control/target for second CNOT
            qubit_i, qubit_j = qubit_j, qubit_i

        return pulses, t - t_start

    def _iswap(self, qubit_i: int, qubit_j: int, t_start: float):
        """iSWAP = 2 × sqrt-SWAP + Z corrections.

        iSWAP = Rz(pi/2) ⊗ Rz(pi/2) · sqrt-SWAP · sqrt-SWAP · Rz(pi/2) ⊗ Rz(pi/2)
        """
        pulses = []
        t = t_start

        # Pre-Z rotations
        for q in [qubit_i, qubit_j]:
            rz_p, rz_d = self._rz(math.pi / 2, q, t)
            pulses.extend(rz_p)

        # Two sqrt-SWAPs
        for _ in range(2):
            ss_p, ss_d = self._sqrt_swap(qubit_i, qubit_j, t)
            pulses.extend(ss_p)
            t += ss_d

        # Post-Z rotations
        for q in [qubit_i, qubit_j]:
            rz_p, rz_d = self._rz(math.pi / 2, q, t)
            pulses.extend(rz_p)

        return pulses, t - t_start

    # ------------------------------------------------------------------
    # Utility: compile from OpenQASM string
    # ------------------------------------------------------------------

    def compile_qasm(self, qasm_str: str) -> CompilationResult:
        """Compile an OpenQASM 2.0 circuit string into a pulse sequence.

        Parameters
        ----------
        qasm_str : str
            OpenQASM 2.0 circuit string.

        Returns
        -------
        CompilationResult
            Compiled pulse sequence.
        """
        circuit = _parse_qasm(qasm_str)
        return self.compile(circuit)


# ---------------------------------------------------------------------------
# Minimal OpenQASM 2.0 parser
# ---------------------------------------------------------------------------

def _parse_qasm(qasm_str: str) -> List[Tuple]:
    """Parse a minimal subset of OpenQASM 2.0 into gate tuples.

    Supports: rx, ry, rz, x, y, z, h, s, t, sdg, tdg, cx, cz, swap, measure.
    Does not support: custom gate definitions, if statements, barriers.

    Parameters
    ----------
    qasm_str : str
        OpenQASM 2.0 string.

    Returns
    -------
    list of (gate_name, params, qubits)
    """
    import re

    circuit = []
    qubit_map = {}  # Maps "q[i]" -> integer index

    for line in qasm_str.splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith("//") or line.startswith("OPENQASM"):
            continue
        if line.startswith("include"):
            continue
        if line.startswith("qreg"):
            # e.g., "qreg q[4]"
            m = re.match(r"qreg\s+(\w+)\[(\d+)\]", line)
            if m:
                name, size = m.group(1), int(m.group(2))
                for i in range(size):
                    qubit_map[f"{name}[{i}]"] = i
            continue
        if line.startswith("creg") or line.startswith("measure"):
            continue
        if line.startswith("barrier"):
            continue

        # Parse gate: name(params) q0, q1, ...
        # e.g., "rx(1.5707963267948966) q[0]"
        # e.g., "cx q[0],q[1]"
        m = re.match(r"(\w+)(?:\(([^)]*)\))?\s+(.*)", line)
        if not m:
            continue

        gate_name = m.group(1).lower()
        param_str = m.group(2) or ""
        qubit_str = m.group(3)

        # Parse parameters
        params = {}
        if param_str:
            param_values = [float(p.strip()) for p in param_str.split(",")]
            if gate_name in ("rx", "ry", "rz", "r", "u1", "u2", "u3"):
                param_names = ["theta", "phi", "lam"]
                for i, v in enumerate(param_values):
                    if i < len(param_names):
                        params[param_names[i]] = v

        # Parse qubits
        qubit_tokens = [q.strip() for q in qubit_str.split(",")]
        qubits = []
        for qt in qubit_tokens:
            if qt in qubit_map:
                qubits.append(qubit_map[qt])
            else:
                # Try to parse "q[i]" directly
                m2 = re.match(r"\w+\[(\d+)\]", qt)
                if m2:
                    qubits.append(int(m2.group(1)))

        if qubits:
            circuit.append((gate_name, params, qubits))

    return circuit
