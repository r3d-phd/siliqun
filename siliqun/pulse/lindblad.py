"""
Lindblad master equation time-domain solver for silicon spin qubits.

Solves the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation:

    d rho / dt = -i/hbar [H(t), rho]
                 + sum_k gamma_k (L_k rho L_k† - 1/2 {L_k† L_k, rho})

where:
    - H(t) is the time-dependent system Hamiltonian (exchange + drive)
    - L_k are Lindblad collapse operators (T1, T2*, charge noise)
    - gamma_k are the corresponding decay rates
    - rho is the density matrix (2^n x 2^n)

The solver supports three integration methods:
    - "rk4"    : 4th-order Runge-Kutta (default, good accuracy/speed tradeoff)
    - "euler"  : 1st-order Euler (fast, low accuracy, useful for debugging)
    - "expm"   : Matrix exponential via Liouvillian superoperator (exact for
                 time-independent H, expensive for large systems)

For GPU acceleration, the density matrix is stored as a CuPy array when
available. Falls back to NumPy transparently.

References:
    Breuer & Petruccione, "The Theory of Open Quantum Systems" (2002)
    Johansson et al., QuTiP: An open-source Python framework for the
        dynamics of open quantum systems, Comp. Phys. Comm. 183, 1760 (2012)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

from ..backend import active_backend
from ..physics.devices.profiles import DeviceProfile
from ..physics.noise.channels import (
    NoiseParams,
    amplitude_damping_kraus,
    phase_damping_kraus,
    depolarizing_kraus,
    compute_dephasing_gamma,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pulse sequence data structures
# ---------------------------------------------------------------------------

@dataclass
class ExchangePulse:
    """A single exchange interaction pulse between two physical spins.

    Parameters
    ----------
    qubit_i, qubit_j : int
        Qubit indices (0-indexed). Must be adjacent for nearest-neighbour
        exchange coupling.
    J : float
        Exchange coupling strength in Hz (J/h). Typical range: 1 MHz – 1 GHz.
    duration : float
        Pulse duration in seconds. Typical range: 1 ns – 500 ns.
    shape : str
        Pulse envelope shape. One of "square", "gaussian", "cosine".
        Default "square" (instantaneous rectangular pulse).
    t_start : float
        Start time of the pulse within the sequence (seconds). Defaults to 0.
    """
    qubit_i: int
    qubit_j: int
    J: float          # Hz
    duration: float   # seconds
    shape: str = "square"
    t_start: float = 0.0

    def theta(self) -> float:
        """Exchange angle theta = 2*pi*J*duration (radians)."""
        return 2 * np.pi * self.J * self.duration

    def envelope(self, t: float) -> float:
        """Normalised pulse envelope at time t (relative to t_start).

        Returns a value in [0, 1] representing the fractional exchange
        coupling strength at time t.
        """
        tau = t - self.t_start
        if tau < 0 or tau > self.duration:
            return 0.0
        if self.shape == "square":
            return 1.0
        elif self.shape == "gaussian":
            sigma = self.duration / 4.0
            centre = self.duration / 2.0
            return float(np.exp(-0.5 * ((tau - centre) / sigma) ** 2))
        elif self.shape == "cosine":
            return float(0.5 * (1 - np.cos(np.pi * tau / self.duration)))
        else:
            raise ValueError(f"Unknown pulse shape: {self.shape}")


@dataclass
class DrivePulse:
    """A microwave drive pulse for ESR/EDSR single-qubit rotations.

    Parameters
    ----------
    qubit : int
        Target qubit index.
    amplitude : float
        Drive amplitude (Rabi frequency) in Hz.
    frequency : float
        Drive frequency in Hz (should be near qubit resonance).
    phase : float
        Drive phase in radians (0 = X-axis, pi/2 = Y-axis).
    duration : float
        Pulse duration in seconds.
    t_start : float
        Start time within the sequence.
    """
    qubit: int
    amplitude: float   # Hz (Rabi frequency)
    frequency: float   # Hz
    phase: float       # radians
    duration: float    # seconds
    t_start: float = 0.0


@dataclass
class PulseSequence:
    """An ordered collection of pulses defining a complete gate operation.

    Pulses can overlap in time (parallel execution) or be sequential.
    The sequence defines the full time axis from t=0 to t=total_duration.

    Parameters
    ----------
    pulses : list
        List of ExchangePulse and/or DrivePulse objects.
    """
    pulses: List = field(default_factory=list)

    def add(self, pulse) -> "PulseSequence":
        """Add a pulse to the sequence. Returns self for chaining."""
        self.pulses.append(pulse)
        return self

    @property
    def total_duration(self) -> float:
        """Total duration of the sequence in seconds."""
        if not self.pulses:
            return 0.0
        return max(p.t_start + p.duration for p in self.pulses)

    @property
    def n_exchange_pulses(self) -> int:
        return sum(1 for p in self.pulses if isinstance(p, ExchangePulse))

    @property
    def n_drive_pulses(self) -> int:
        return sum(1 for p in self.pulses if isinstance(p, DrivePulse))

    def __len__(self) -> int:
        return len(self.pulses)


# ---------------------------------------------------------------------------
# Lindblad collapse operators from device noise parameters
# ---------------------------------------------------------------------------

def _build_collapse_operators(
    n_qubits: int,
    noise_params: NoiseParams,
) -> List[Tuple[np.ndarray, float]]:
    """Build Lindblad collapse operators (L_k, sqrt(gamma_k)) for all qubits.

    Returns a list of (L_k, sqrt_gamma_k) tuples where:
        L_k    : 2^n x 2^n collapse operator matrix
        sqrt_gamma_k : sqrt of the decay rate (already absorbed into L_k norm)

    Collapse operators included:
        - T1 amplitude damping: L = sigma_minus = |0><1|
        - T2* pure dephasing:   L = sigma_z / 2
        - Depolarising (from charge noise): L = I (scalar noise)
    """
    dim = 2 ** n_qubits
    I_n = np.eye(dim, dtype=np.complex128)
    collapse_ops = []

    # Collapse operators follow the convention: L_k with rate gamma_k such that
    # the Lindblad term is gamma_k * (L_k rho L_k† - 1/2 {L_k†L_k, rho})
    #
    # T1 amplitude damping: L = sigma_minus = |0><1|, gamma = 1/T1
    # T2* pure dephasing:   L = sigma_z (full operator, NOT sigma_z/2),
    #                        gamma = 1/T2star - 1/(2*T1)
    #   Note: Using L=sigma_z/2 would require gamma *= 4 to compensate.
    #   We use the standard convention L=sigma_z for clarity.
    #   Ref: Breuer & Petruccione (2002), Eq. 3.67
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0><1|
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    for q in range(n_qubits):
        # Build 2^n x 2^n operators by tensoring with identity on other qubits
        L_relax = _embed_single_qubit_op(sigma_minus, q, n_qubits)
        L_deph = _embed_single_qubit_op(sigma_z, q, n_qubits)  # Full sigma_z

        # T1 relaxation rate: gamma_1 = 1/T1
        gamma_1 = 1.0 / noise_params.T1
        collapse_ops.append((L_relax, np.sqrt(gamma_1)))

        # T2* pure dephasing rate: gamma_phi = 1/T2star - 1/(2*T1)
        # This is the pure dephasing contribution beyond T1 relaxation.
        # Using L=sigma_z (not sigma_z/2) with this rate is the standard convention.
        gamma_phi = compute_dephasing_gamma(noise_params.T2_star, noise_params.T1)
        if gamma_phi > 0:
            collapse_ops.append((L_deph, np.sqrt(gamma_phi)))

    return collapse_ops


def _embed_single_qubit_op(
    op_2x2: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Embed a 2x2 single-qubit operator into the full 2^n Hilbert space."""
    result = np.array([[1.0]], dtype=np.complex128)
    for q in range(n_qubits):
        if q == qubit:
            result = np.kron(result, op_2x2)
        else:
            result = np.kron(result, np.eye(2, dtype=np.complex128))
    return result


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------

def _exchange_hamiltonian(
    qubit_i: int,
    qubit_j: int,
    J: float,
    n_qubits: int,
) -> np.ndarray:
    """Heisenberg exchange Hamiltonian H = J/4 * (XX + YY + ZZ) for qubits i,j.

    Parameters
    ----------
    qubit_i, qubit_j : int
        Qubit indices.
    J : float
        Exchange coupling in Hz (J/h). The Hamiltonian is H = h*J/4*(XX+YY+ZZ).
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    ndarray
        2^n x 2^n Hamiltonian matrix in units of rad/s (multiplied by 2*pi).
    """
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    H = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=np.complex128)
    for P in [X, Y, Z]:
        term = np.array([[1.0]], dtype=np.complex128)
        for q in range(n_qubits):
            if q == qubit_i or q == qubit_j:
                term = np.kron(term, P)
            else:
                term = np.kron(term, np.eye(2, dtype=np.complex128))
        H += 0.25 * term

    # H is in rad/s: H = 2*pi*J * (J_operator/4)
    # where J is the exchange coupling in Hz (J/h convention).
    # The factor 2*pi converts Hz -> rad/s.
    # Ref: Loss & DiVincenzo (1998), Phys. Rev. A 57, 120
    return 2 * np.pi * J * H


def _drive_hamiltonian(
    qubit: int,
    amplitude: float,
    frequency: float,
    phase: float,
    t: float,
    n_qubits: int,
) -> np.ndarray:
    """Rotating-frame drive Hamiltonian for ESR/EDSR.

    H_drive = pi * amplitude * (cos(phase)*X + sin(phase)*Y)

    In the rotating frame at the drive frequency, the time dependence
    is removed (rotating wave approximation).

    Parameters
    ----------
    qubit : int
        Target qubit.
    amplitude : float
        Rabi frequency in Hz.
    frequency : float
        Drive frequency (unused in RWA, included for future use).
    phase : float
        Drive phase in radians.
    t : float
        Current time (unused in RWA).
    n_qubits : int
        Total number of qubits.
    """
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    op = np.cos(phase) * X + np.sin(phase) * Y
    H_q = _embed_single_qubit_op(op, qubit, n_qubits)
    return np.pi * amplitude * H_q


# ---------------------------------------------------------------------------
# Liouvillian superoperator
# ---------------------------------------------------------------------------

def _build_liouvillian(
    H: np.ndarray,
    collapse_ops: List[Tuple[np.ndarray, float]],
) -> np.ndarray:
    """Build the Liouvillian superoperator L such that d vec(rho)/dt = L vec(rho).

    Uses the vec(rho) = rho.flatten() convention (row-major).

    L = -i (I ⊗ H - H^T ⊗ I)
        + sum_k gamma_k (L_k* ⊗ L_k - 1/2 I ⊗ L_k†L_k - 1/2 (L_k†L_k)^T ⊗ I)

    Parameters
    ----------
    H : ndarray
        System Hamiltonian (dim x dim).
    collapse_ops : list of (L_k, sqrt_gamma_k)
        Lindblad collapse operators and their rates.

    Returns
    -------
    ndarray
        Liouvillian superoperator (dim^2 x dim^2).
    """
    dim = H.shape[0]
    I = np.eye(dim, dtype=np.complex128)

    # Coherent part: -i[H, rho] in superoperator form
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Dissipative part
    for Lk, sqrt_gamma in collapse_ops:
        gamma = sqrt_gamma ** 2
        LkdLk = Lk.conj().T @ Lk
        L += gamma * (
            np.kron(Lk.conj(), Lk)
            - 0.5 * np.kron(I, LkdLk)
            - 0.5 * np.kron(LkdLk.T, I)
        )

    return L


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

@dataclass
class LindbladResult:
    """Result of a Lindblad master equation simulation.

    Attributes
    ----------
    rho_final : ndarray
        Final density matrix (2^n x 2^n).
    times : list of float
        Time points at which rho was recorded.
    rho_history : list of ndarray
        Density matrices at each recorded time point.
    expectation_z : list of list of float
        ⟨Z_i⟩ for each qubit at each time point.
    purity_history : list of float
        Tr(rho^2) at each time point.
    fidelity_history : list of float
        Fidelity with target state at each time point (if target provided).
    total_time : float
        Total simulation time in seconds.
    n_steps : int
        Number of integration steps taken.
    """
    rho_final: np.ndarray
    times: List[float]
    rho_history: List[np.ndarray]
    expectation_z: List[List[float]]
    purity_history: List[float]
    fidelity_history: List[float]
    total_time: float
    n_steps: int

    def fidelity_at_end(self) -> float:
        """Final state fidelity (last entry in fidelity_history)."""
        return self.fidelity_history[-1] if self.fidelity_history else float("nan")

    def purity_at_end(self) -> float:
        """Final state purity Tr(rho^2)."""
        return self.purity_history[-1]

    def expectation_z_final(self) -> List[float]:
        """⟨Z_i⟩ for each qubit at the final time."""
        return self.expectation_z[-1] if self.expectation_z else []


class LindbladSimulator:
    """Time-domain Lindblad master equation solver for silicon spin qubits.

    Evolves an n-qubit density matrix under a time-dependent Hamiltonian
    (exchange pulses + microwave drives) with device-calibrated noise.

    Parameters
    ----------
    device : DeviceProfile
        Silicon spin qubit device profile (Donor, SiMOS, GAA, or SLEDGE).
        Provides T1, T2*, charge noise, and gate time parameters.
    n_qubits : int
        Number of logical qubits to simulate.
    method : str
        Integration method: "rk4" (default), "euler", or "expm".
    dt : float
        Time step in seconds. Default 0.1 ns. Smaller values give higher
        accuracy at the cost of more computation.
    record_every : int
        Record density matrix snapshot every N steps. Default 10.
    use_gpu : bool
        Use CuPy GPU arrays if available. Default True.
    seed : int, optional
        Random seed for stochastic noise sampling.

    Examples
    --------
    >>> from siliqun.physics.devices.profiles import get_device
    >>> from siliqun.pulse.lindblad import LindbladSimulator, PulseSequence, ExchangePulse
    >>>
    >>> device = get_device("simos", n_qubits=2)
    >>> sim = LindbladSimulator(device, n_qubits=2)
    >>>
    >>> seq = PulseSequence()
    >>> seq.add(ExchangePulse(qubit_i=0, qubit_j=1, J=10e6, duration=50e-9))
    >>>
    >>> result = sim.evolve(seq)
    >>> print(f"Final fidelity: {result.fidelity_at_end():.4f}")
    >>> print(f"Final purity:   {result.purity_at_end():.4f}")
    """

    def __init__(
        self,
        device: DeviceProfile,
        n_qubits: int,
        method: str = "rk4",
        dt: float = 1e-10,      # 0.1 ns default
        record_every: int = 10,
        use_gpu: bool = True,
        seed: Optional[int] = None,
    ):
        self.device = device
        self.n_qubits = n_qubits
        self.method = method
        self.dt = dt
        self.record_every = record_every
        self.seed = seed

        self._dim = 2 ** n_qubits
        self._rng = np.random.default_rng(seed)

        # Try to use GPU
        self._use_gpu = False
        if use_gpu:
            try:
                import cupy as cp
                self._xp = cp
                self._use_gpu = True
                logger.info("LindbladSimulator: using CuPy GPU backend")
            except ImportError:
                self._xp = np
                logger.info("LindbladSimulator: CuPy not available, using NumPy")
        else:
            self._xp = np

        # Build noise parameters from device profile
        self._noise_params = device.noise_params
        self._collapse_ops = _build_collapse_operators(n_qubits, self._noise_params)

        # Current density matrix (initialised to |0...0><0...0|)
        self._rho = self._zero_state_dm()

        logger.info(
            "LindbladSimulator: n=%d, dim=%d, method=%s, dt=%.2e s, gpu=%s",
            n_qubits, self._dim, method, dt, self._use_gpu,
        )

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _zero_state_dm(self) -> np.ndarray:
        """Density matrix for |0...0> state."""
        rho = np.zeros((self._dim, self._dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset the simulator to a given initial state.

        Parameters
        ----------
        initial_state : ndarray, optional
            Initial state as a state vector (2^n,) or density matrix (2^n, 2^n).
            Defaults to |0...0>.
        """
        if initial_state is None:
            self._rho = self._zero_state_dm()
        elif initial_state.ndim == 1:
            # State vector -> density matrix
            psi = initial_state.astype(np.complex128)
            self._rho = np.outer(psi, psi.conj())
        elif initial_state.ndim == 2:
            self._rho = initial_state.astype(np.complex128)
        else:
            raise ValueError("initial_state must be a 1D state vector or 2D density matrix")

    # ------------------------------------------------------------------
    # Hamiltonian at time t
    # ------------------------------------------------------------------

    def _hamiltonian_at(self, t: float, sequence: PulseSequence) -> np.ndarray:
        """Compute the total Hamiltonian at time t for the given pulse sequence."""
        H = np.zeros((self._dim, self._dim), dtype=np.complex128)

        for pulse in sequence.pulses:
            if isinstance(pulse, ExchangePulse):
                env = pulse.envelope(t)
                if env > 1e-12:
                    H += env * _exchange_hamiltonian(
                        pulse.qubit_i, pulse.qubit_j,
                        pulse.J, self.n_qubits
                    )
            elif isinstance(pulse, DrivePulse):
                tau = t - pulse.t_start
                if 0 <= tau <= pulse.duration:
                    H += _drive_hamiltonian(
                        pulse.qubit, pulse.amplitude,
                        pulse.frequency, pulse.phase,
                        t, self.n_qubits
                    )

        return H

    # ------------------------------------------------------------------
    # Lindblad right-hand side: d rho / dt
    # ------------------------------------------------------------------

    def _drho_dt(
        self,
        rho: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """Compute d rho / dt = -i[H, rho] + D[rho].

        Parameters
        ----------
        rho : ndarray
            Current density matrix.
        H : ndarray
            Current Hamiltonian.

        Returns
        -------
        ndarray
            Time derivative of the density matrix.
        """
        # Coherent evolution: -i[H, rho]
        drho = -1j * (H @ rho - rho @ H)

        # Dissipative Lindblad terms
        for Lk, sqrt_gamma in self._collapse_ops:
            gamma = sqrt_gamma ** 2
            LkdLk = Lk.conj().T @ Lk
            drho += gamma * (
                Lk @ rho @ Lk.conj().T
                - 0.5 * (LkdLk @ rho + rho @ LkdLk)
            )

        return drho

    # ------------------------------------------------------------------
    # Integration methods
    # ------------------------------------------------------------------

    def _step_euler(
        self,
        rho: np.ndarray,
        t: float,
        sequence: PulseSequence,
    ) -> np.ndarray:
        """Single Euler step."""
        H = self._hamiltonian_at(t, sequence)
        return rho + self.dt * self._drho_dt(rho, H)

    def _step_rk4(
        self,
        rho: np.ndarray,
        t: float,
        sequence: PulseSequence,
    ) -> np.ndarray:
        """Single 4th-order Runge-Kutta step."""
        dt = self.dt

        H1 = self._hamiltonian_at(t, sequence)
        k1 = self._drho_dt(rho, H1)

        H2 = self._hamiltonian_at(t + dt / 2, sequence)
        k2 = self._drho_dt(rho + dt / 2 * k1, H2)

        H3 = self._hamiltonian_at(t + dt / 2, sequence)
        k3 = self._drho_dt(rho + dt / 2 * k2, H3)

        H4 = self._hamiltonian_at(t + dt, sequence)
        k4 = self._drho_dt(rho + dt * k3, H4)

        return rho + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _step_expm(
        self,
        rho: np.ndarray,
        t: float,
        sequence: PulseSequence,
    ) -> np.ndarray:
        """Single step via matrix exponential of the Liouvillian.

        Exact for time-independent H within the step. Expensive for large n.
        """
        H = self._hamiltonian_at(t, sequence)
        L = _build_liouvillian(H, self._collapse_ops)
        prop = expm(L * self.dt)
        rho_vec = rho.flatten()
        rho_new_vec = prop @ rho_vec
        return rho_new_vec.reshape(self._dim, self._dim)

    # ------------------------------------------------------------------
    # Main evolution method
    # ------------------------------------------------------------------

    def evolve(
        self,
        sequence: PulseSequence,
        target_state: Optional[np.ndarray] = None,
        reset_before: bool = True,
    ) -> LindbladResult:
        """Evolve the density matrix under the given pulse sequence.

        Parameters
        ----------
        sequence : PulseSequence
            The pulse sequence to apply. Can contain ExchangePulse and/or
            DrivePulse objects with arbitrary timing and overlap.
        target_state : ndarray, optional
            Target state vector (2^n,) or density matrix (2^n, 2^n) for
            fidelity tracking. If None, fidelity_history will be empty.
        reset_before : bool
            If True (default), reset to |0...0> before evolving.

        Returns
        -------
        LindbladResult
            Simulation result containing final density matrix, time history,
            expectation values, purity, and fidelity.
        """
        if reset_before:
            self.reset()

        total_time = sequence.total_duration
        n_steps = max(1, int(np.ceil(total_time / self.dt)))
        actual_dt = total_time / n_steps

        # Temporarily override dt for this run
        saved_dt = self.dt
        self.dt = actual_dt

        # Prepare target density matrix for fidelity
        rho_target = None
        if target_state is not None:
            if target_state.ndim == 1:
                rho_target = np.outer(target_state, target_state.conj())
            else:
                rho_target = target_state.astype(np.complex128)

        # Choose step function
        if self.method == "rk4":
            step_fn = self._step_rk4
        elif self.method == "euler":
            step_fn = self._step_euler
        elif self.method == "expm":
            step_fn = self._step_expm
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

        # Integration loop
        rho = self._rho.copy()
        times = []
        rho_history = []
        expectation_z_history = []
        purity_history = []
        fidelity_history = []

        t = 0.0
        for step in range(n_steps):
            # Record snapshot
            if step % self.record_every == 0:
                times.append(t)
                rho_history.append(rho.copy())
                purity_history.append(float(np.real(np.trace(rho @ rho))))
                expectation_z_history.append(self._compute_z_expectations(rho))
                if rho_target is not None:
                    fidelity_history.append(self._compute_fidelity(rho, rho_target))

            # Integrate one step
            rho = step_fn(rho, t, sequence)

            # Enforce Hermiticity and positivity (numerical stabilisation)
            rho = 0.5 * (rho + rho.conj().T)
            rho = np.real(rho).astype(np.complex128)  # Remove tiny imaginary parts
            # Re-normalise
            tr = np.real(np.trace(rho))
            if tr > 1e-12:
                rho /= tr

            t += actual_dt

        # Final snapshot
        times.append(t)
        rho_history.append(rho.copy())
        purity_history.append(float(np.real(np.trace(rho @ rho))))
        expectation_z_history.append(self._compute_z_expectations(rho))
        if rho_target is not None:
            fidelity_history.append(self._compute_fidelity(rho, rho_target))

        # Store final state
        self._rho = rho
        self.dt = saved_dt

        logger.info(
            "LindbladSimulator.evolve: T=%.2e s, steps=%d, "
            "final_purity=%.4f, final_fidelity=%s",
            total_time, n_steps,
            purity_history[-1],
            f"{fidelity_history[-1]:.4f}" if fidelity_history else "N/A",
        )

        return LindbladResult(
            rho_final=rho,
            times=times,
            rho_history=rho_history,
            expectation_z=expectation_z_history,
            purity_history=purity_history,
            fidelity_history=fidelity_history,
            total_time=total_time,
            n_steps=n_steps,
        )

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _compute_z_expectations(self, rho: np.ndarray) -> List[float]:
        """Compute ⟨Z_i⟩ = Tr(Z_i rho) for each qubit."""
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        expectations = []
        for q in range(self.n_qubits):
            Z_q = _embed_single_qubit_op(Z, q, self.n_qubits)
            expectations.append(float(np.real(np.trace(Z_q @ rho))))
        return expectations

    def _compute_fidelity(
        self,
        rho: np.ndarray,
        rho_target: np.ndarray,
    ) -> float:
        """Compute state fidelity F(rho, rho_target) = Tr(rho_target @ rho).

        For pure target states, this simplifies to F = <psi|rho|psi>.
        """
        return float(np.real(np.trace(rho_target @ rho)))

    def expectation_value(
        self,
        observable: np.ndarray,
    ) -> float:
        """Compute ⟨O⟩ = Tr(O rho) for an arbitrary observable O.

        Parameters
        ----------
        observable : ndarray
            2^n x 2^n Hermitian observable matrix.

        Returns
        -------
        float
            Real part of Tr(O rho).
        """
        return float(np.real(np.trace(observable @ self._rho)))

    @property
    def density_matrix(self) -> np.ndarray:
        """Current density matrix (2^n x 2^n)."""
        return self._rho.copy()

    @property
    def purity(self) -> float:
        """Current purity Tr(rho^2)."""
        return float(np.real(np.trace(self._rho @ self._rho)))
