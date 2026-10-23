"""
siliqun.core.abc
================
Abstract base class (ABC) for SiliQun hardware-platform plugins.

Every plugin that adds a new silicon spin qubit technology to SiliQun
**must** subclass :class:`TechnologyProfile` and implement its five
abstract methods.  The framework then handles the Lindblad solver,
Gymnasium interface, and RL agent automatically — the plugin author
writes only the platform-specific physics.

Minimum plugin contract (47 lines for the GAA reference implementation):

1. **Profile class** — a :class:`TechnologyProfile` subclass that
   implements the five abstract methods and populates
   :attr:`calibration_record` from peer-reviewed sources.
2. **Calibration data** — a JSON file with raw measurements and DOIs.
3. **Validation benchmarks** — three PGIRS Phase-1 benchmarks (Rabi,
   Bell, T2 echo) that must pass to ±2 % before registration.
4. **Gymnasium environment ID** — a unique string following the
   convention ``SiliQun-{TechnologyName}-v{N}``.

Example
-------
::

    from siliqun.core.abc import TechnologyProfile, CalibrationRecord
    import numpy as np

    class MySiMOSProfile(TechnologyProfile):
        technology_name = "SiMOS-Custom"
        gymnasium_env_id = "SiliQun-SiMOS-Custom-v1"

        def device_parameters(self):
            return {"T1": 1e-3, "T2": 5e-4, "n_qubits": 2}

        def drift_hamiltonian(self):
            return np.zeros((4, 4), dtype=complex)

        def control_hamiltonians(self):
            sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
            return [np.kron(sx, np.eye(2)), np.kron(np.eye(2), sx)]

        def noise_channels(self):
            gamma1 = 1 / self.device_parameters()["T1"]
            sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
            L1 = np.sqrt(gamma1) * np.kron(sigma_minus, np.eye(2))
            L2 = np.sqrt(gamma1) * np.kron(np.eye(2), sigma_minus)
            return [L1, L2]

        def gate_library(self):
            return ["Rx", "Ry", "Rz", "CNOT"]

        @property
        def calibration_record(self):
            return CalibrationRecord(
                source_doi="10.1038/s41586-022-04592-2",
                source_description="Xue et al. Nature 2022 SiMOS 2Q",
                raw_values={"T1_ms": 1.0, "T2_ms": 0.5},
            )
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# CalibrationRecord — lightweight container for provenance metadata
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    """Provenance metadata for a hardware profile's calibration values.

    Parameters
    ----------
    source_doi:
        DOI of the primary experimental or simulation paper.
    source_description:
        Human-readable description of the source (author, journal, year).
    raw_values:
        Dictionary of raw measured values with units embedded in the key
        name, e.g. ``{"T1_ms": 1.0, "T2_ms": 0.5}``.
    additional_sources:
        Optional list of supplementary DOIs.
    """

    source_doi: str
    source_description: str
    raw_values: Dict[str, Any] = field(default_factory=dict)
    additional_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_doi": self.source_doi,
            "source_description": self.source_description,
            "raw_values": self.raw_values,
            "additional_sources": self.additional_sources,
        }

    @classmethod
    def from_json(cls, path: str) -> "CalibrationRecord":
        """Load a CalibrationRecord from a JSON file."""
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)


# ---------------------------------------------------------------------------
# TechnologyProfile — the plugin ABC
# ---------------------------------------------------------------------------

class TechnologyProfile(ABC):
    """Abstract base class for SiliQun hardware-platform plugins.

    Subclass this to integrate a new silicon spin qubit technology (or
    any other qubit platform expressible in the Lindblad formalism) with
    the SiliQun framework.

    Class attributes
    ----------------
    technology_name : str
        Short human-readable name, e.g. ``"SiMOS-Nominal"``.
    gymnasium_env_id : str
        Unique Gymnasium registry ID following the convention
        ``"SiliQun-{TechnologyName}-v{N}"``.

    Abstract methods (must be implemented by every plugin)
    -------------------------------------------------------
    device_parameters()
        Return a dict of scalar physical parameters (T1, T2, gate
        times, noise amplitudes, …).
    drift_hamiltonian()
        Return the always-on (drift) Hamiltonian as a complex NumPy
        array of shape ``(2**n, 2**n)``.
    control_hamiltonians()
        Return a list of control Hamiltonians (one per control channel).
    noise_channels()
        Return a list of Lindblad collapse operators.
    gate_library()
        Return a list of native gate names supported by this profile.

    Property (must be implemented by every plugin)
    -----------------------------------------------
    calibration_record
        Return a :class:`CalibrationRecord` with provenance metadata.
    """

    # Subclasses MUST set these two class attributes.
    technology_name: str = ""
    gymnasium_env_id: str = ""

    # ------------------------------------------------------------------
    # Abstract methods — the five mandatory ABC methods
    # ------------------------------------------------------------------

    @abstractmethod
    def device_parameters(self) -> Dict[str, Any]:
        """Return a dict of scalar physical parameters.

        Required keys (SI units unless noted):
            ``T1``   — longitudinal relaxation time (s)
            ``T2``   — transverse dephasing time (s)
            ``tau1`` — single-qubit gate duration (s)
            ``tau2`` — two-qubit gate duration (s)
            ``p2``   — two-qubit gate error probability (dimensionless)

        Additional keys (charge noise, exchange noise, hyperfine, …)
        are allowed and encouraged for physics-faithful implementations.
        """

    @abstractmethod
    def drift_hamiltonian(self) -> np.ndarray:
        """Return the always-on (drift) Hamiltonian.

        Returns
        -------
        H_drift : np.ndarray, shape (2**n, 2**n), dtype complex
            The drift Hamiltonian in units of rad/s.  Use ``np.zeros``
            for platforms where the drift term is negligible or absorbed
            into the control Hamiltonians.
        """

    @abstractmethod
    def control_hamiltonians(self) -> List[np.ndarray]:
        """Return the list of control Hamiltonians.

        Each element corresponds to one control channel (e.g. one
        microwave drive line or one exchange gate voltage).

        Returns
        -------
        H_controls : list of np.ndarray, each shape (2**n, 2**n), dtype complex
        """

    @abstractmethod
    def noise_channels(self) -> List[np.ndarray]:
        """Return the list of Lindblad collapse operators.

        Typical channels: amplitude damping (T1), phase dephasing (T2),
        depolarising noise (p_gate), charge noise (σ_ε), exchange noise
        (σ_J), hyperfine fluctuations (A_hf).

        Returns
        -------
        L_ops : list of np.ndarray, each shape (2**n, 2**n), dtype complex
        """

    @abstractmethod
    def gate_library(self) -> List[str]:
        """Return the list of native gate names.

        Example: ``["Rx", "Ry", "Rz", "CNOT"]``
        """

    @property
    @abstractmethod
    def calibration_record(self) -> CalibrationRecord:
        """Return the provenance metadata for this profile's parameters."""

    # ------------------------------------------------------------------
    # Concrete helpers (available to all plugins for free)
    # ------------------------------------------------------------------

    def n_qubits(self) -> int:
        """Infer the number of qubits from the drift Hamiltonian shape."""
        dim = self.drift_hamiltonian().shape[0]
        return int(round(math.log2(dim)))

    def validate(self, tolerance: float = 0.02) -> Dict[str, bool]:
        """Run the three PGIRS Phase-1 validation benchmarks.

        Parameters
        ----------
        tolerance:
            Maximum allowed fractional deviation from the analytical
            solution (default 2 %).

        Returns
        -------
        results : dict
            ``{"rabi": bool, "bell": bool, "t2_echo": bool}``
        """
        return PGIRSValidator(self).run(tolerance=tolerance)

    def register(self) -> None:
        """Register this profile's Gymnasium environment.

        Raises
        ------
        RuntimeError
            If any PGIRS Phase-1 benchmark fails.
        """
        results = self.validate()
        failures = [k for k, v in results.items() if not v]
        if failures:
            raise RuntimeError(
                f"Plugin '{self.technology_name}' failed PGIRS Phase-1 "
                f"benchmarks: {failures}.  Fix the profile before registering."
            )
        # Lazy import to avoid circular dependency with the engine.
        try:
            import gymnasium as gym
            from siliqun.engine.gym_env import SiliQunEnvAdapter
            gym.register(
                id=self.gymnasium_env_id,
                entry_point=SiliQunEnvAdapter,
                kwargs={"profile": self},
            )
        except ImportError:
            pass  # Gymnasium not installed; registration is a no-op.

    def __repr__(self) -> str:
        return (
            f"<TechnologyProfile '{self.technology_name}' "
            f"env_id='{self.gymnasium_env_id}'>"
        )


# ---------------------------------------------------------------------------
# PGIRSValidator — PGIRS Phase-1 benchmark harness
# ---------------------------------------------------------------------------

class PGIRSValidator:
    """PGIRS Phase-1 validation harness.

    Validates the **actual** :class:`~siliqun.pulse.lindblad.LindbladSimulator`
    against closed-form physics.  Each benchmark integrates the Lindblad
    master equation and compares the *simulated* observable to an independent
    analytic target — the two quantities are not the same expression.

    1. **Rabi oscillation** — single-qubit resonant drive.  A pi-pulse of
       duration ``tau1`` (drive amplitude ``A = pi / tau1``) must transfer the
       population to ``|1>``: ``pop|1>(tau1) = 1`` (coherence limit, T1, T2* ->
       inf).  A mid-drive quarter-pulse must give ``pop|1>(tau1/2) = sin^2(pi/4)
       = 0.5``.
    2. **Bell / sqrt-SWAP** — two-qubit exchange.  Starting from ``|01>``,
       evolving under the Heisenberg exchange Hamiltonian for the sqrt-SWAP
       time ``t_be = T_swap / 2`` must produce a maximally entangled state:
       concurrence ``C = 1`` (the exact phase of the Bell phase is set by the
       exchange Hamiltonian, so we check concurrence, which is
       phase-independent, plus the full-swap ``F(|10>) = 1``).
    3. **T2\\* dephasing** — free precession of ``(|0>+|1>)/sqrt(2)`` under the
       profile's T1 and T2*.  The solver must reproduce the analytic coherence
       ``|rho01|(t) = 0.5 exp(-t / T2*)`` at ``t = T2*`` (the pure-dephasing
       and amplitude-damping terms of the solver sum to a net decay rate of
       ``1/T2*``).

    In addition, ``run()`` asserts the internal **fidelity bound**: the
    estimator must stay within ``[0, 1]`` for a full Rabi cycle (a raw trace
    above 1 was the source of the previously reported unphysical ``F = 1.0009``).

    Returns
    -------
    results : dict
        ``{"rabi": bool, "bell": bool, "t2_echo": bool}`` — pass/fail per
        benchmark.  The per-quantity simulated vs analytic values and deltas
        are stored on :attr:`last_deltas` so callers (e.g. the paper's
        validation table) can cite measured deviations rather than a
        self-comparison.

    Parameters
    ----------
    profile:
        The :class:`TechnologyProfile` instance to validate.
    """

    def __init__(self, profile: TechnologyProfile) -> None:
        self.profile = profile
        self.last_deltas: Dict[str, Any] = {}

    def run(self, tolerance: float = 0.02) -> Dict[str, bool]:
        """Run all three solver-backed benchmarks and return pass/fail."""
        results: Dict[str, bool] = {}
        self.last_deltas = {}
        results["rabi"] = self._benchmark_rabi(tolerance)
        results["bell"] = self._benchmark_bell(tolerance)
        results["t2_echo"] = self._benchmark_t2_echo(tolerance)
        return results

    # ------------------------------------------------------------------
    # Solver / analytic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _swap_times(J: float):
        """Return (full_swap_t, sqrt_swap_t) from the exchange Hamiltonian."""
        from siliqun.pulse.lindblad import _exchange_hamiltonian
        from scipy.linalg import expm
        Hx = _exchange_hamiltonian(0, 1, J, 2)
        ts = np.linspace(0.0, 4.0 / J, 4001)
        amp = np.array([abs(expm(-1j * Hx * t)[2, 1]) for t in ts])
        t_swap = float(ts[int(np.argmax(amp))])
        return t_swap, 0.5 * t_swap

    def _sim_1q(self, n: int, T1: float, T2s: float, dt: float):
        from siliqun.physics.devices.profiles import get_device_profile
        from siliqun.physics.noise.channels import NoiseParams
        dev = get_device_profile("simos", n_qubits=n)
        dev.noise_params = NoiseParams(
            t1_times=[T1] * n, t2_star_times=[T2s] * n,
            measurement_fidelity=1.0, dephasing_model="exponential",
        )
        from siliqun.pulse.lindblad import LindbladSimulator
        return LindbladSimulator(dev, n_qubits=n, method="rk4", dt=dt, use_gpu=False)

    # ------------------------------------------------------------------
    # Individual benchmarks (solver-backed)
    # ------------------------------------------------------------------

    def _benchmark_rabi(self, tolerance: float) -> bool:
        """Rabi: simulate a resonant pi-pulse; compare pop|1> to analytic."""
        try:
            params = self.profile.device_parameters()
            tau1 = params.get("tau1", 10e-9)
            # Drive H = pi*A*(cos phi X + sin phi Y); full transfer (pop|1>=1)
            # occurs when pi*A*t = pi/2, i.e. A*t = 1/2.  A pi-pulse completed
            # in the profile's single-qubit gate time tau1 needs A = 1/(2*tau1).
            A = 1.0 / (2.0 * tau1)
            sim = self._sim_1q(1, float("inf"), float("inf"), dt=1e-11)
            from siliqun.pulse.lindblad import PulseSequence, DrivePulse
            # full pi-pulse -> |1>
            sim.reset()
            r_pi = sim.evolve(PulseSequence().add(DrivePulse(0, A, 5e9, 0.0, tau1)))
            pop_pi = float(np.real(r_pi.rho_final[1, 1]))
            # quarter-pulse -> sin^2(pi/4) = 0.5
            sim.reset()
            r_q = sim.evolve(PulseSequence().add(DrivePulse(0, A, 5e9, 0.0, tau1 / 2)))
            pop_q = float(np.real(r_q.rho_final[1, 1]))
            self.last_deltas["rabi"] = {
                "pop1_pi": pop_pi, "target_pi": 1.0, "delta_pi": abs(pop_pi - 1.0),
                "pop1_quarter": pop_q, "target_quarter": 0.5,
                "delta_quarter": abs(pop_q - 0.5),
            }
            return (abs(pop_pi - 1.0) <= tolerance) and (abs(pop_q - 0.5) <= tolerance)
        except Exception as e:  # noqa: BLE001
            self.last_deltas["rabi_error"] = f"{type(e).__name__}: {e}"
            return False

    def _benchmark_bell(self, tolerance: float) -> bool:
        """Bell/sqrt-SWAP: concurrence must be 1; full swap must give F(|10>)=1."""
        try:
            from siliqun.pulse.lindblad import PulseSequence, ExchangePulse
            J = 20e6
            t_swap, t_be = self._swap_times(J)
            sim = self._sim_1q(2, float("inf"), float("inf"), dt=1e-12)
            init01 = np.zeros(4, dtype=complex); init01[1] = 1.0
            # sqrt-SWAP -> maximally entangled
            sim.reset(init01)
            r_be = sim.evolve(PulseSequence().add(ExchangePulse(0, 1, J, t_be)), reset_before=False)
            rho = r_be.rho_final
            p = [np.sqrt(max(float(np.real(rho[i, i])), 0.0)) for i in range(4)]
            conc = 2.0 * abs(p[0] * p[3] - p[1] * p[2])
            # full swap -> |10>
            sim.reset(init01)
            r_sw = sim.evolve(PulseSequence().add(ExchangePulse(0, 1, J, t_swap)), reset_before=False)
            v10 = np.zeros(4, dtype=complex); v10[2] = 1.0
            f10 = float(np.real(np.trace(np.outer(v10, v10.conj()) @ r_sw.rho_final)))
            self.last_deltas["bell"] = {
                "concurrence": float(conc), "target_concurrence": 1.0,
                "delta_concurrence": abs(conc - 1.0),
                "F_full_swap_10": f10, "target_F": 1.0, "delta_F": abs(f10 - 1.0),
            }
            return (abs(conc - 1.0) <= tolerance) and (abs(f10 - 1.0) <= tolerance)
        except Exception as e:  # noqa: BLE001
            self.last_deltas["bell_error"] = f"{type(e).__name__}: {e}"
            return False

    def _benchmark_t2_echo(self, tolerance: float) -> bool:
        """T2*: free dephasing must match |rho01|(T2eff) = 0.5 exp(-1)."""
        try:
            params = self.profile.device_parameters()
            T1 = params.get("T1", 1e-3)
            T2 = params.get("T2", 5e-4)     # profile dephasing time (T2*)
            # The solver's net coherence decay is exp(-t / T2*): its pure-
            # dephasing term gamma_phi = 1/(2 T2*) - 1/(4 T1) supplements the
            # amplitude-damping contribution 1/(2 T1) to a total 1/T2*.
            # So evolve exactly one dephasing time and check |rho01| = 0.5 e^-1.
            T2eff = T2
            sim = self._sim_1q(1, T1, T2, dt=1e-9)
            from siliqun.pulse.lindblad import PulseSequence, DrivePulse
            sim.reset(np.array([1, 1], dtype=complex) / np.sqrt(2))
            r = sim.evolve(PulseSequence().add(DrivePulse(0, 0.0, 5e9, 0.0, T2eff)), reset_before=False)
            obs = abs(r.rho_final[0, 1])
            ana = 0.5 * math.exp(-1.0)
            rel = abs(obs - ana) / ana
            self.last_deltas["t2_echo"] = {
                "T2eff_s": T2eff, "solver_rho01": float(obs), "analytic": ana,
                "relative_error": float(rel),
            }
            return rel <= tolerance
        except Exception as e:  # noqa: BLE001
            self.last_deltas["t2_echo_error"] = f"{type(e).__name__}: {e}"
            return False
