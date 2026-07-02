"""
Noise channels for silicon spin qubit systems.

Implements realistic decoherence models as Kraus operators and
MPO-based quantum channels:

    - T1 relaxation (amplitude damping)
    - T2 dephasing (phase damping) - both exponential and Gaussian
    - 1/f charge noise (correlated, non-Markovian)
    - Johnson-Nyquist thermal noise
    - Leakage to non-computational states
    - Crosstalk-induced errors (1/r^3 capacitive coupling)
    - DFS-encoded qubit noise (quadratic error suppression)

Calibrated to experimental data from:
    Weinstein et al., Nature 615, 817-822 (2023) - HRL SLEDGE device

Each channel can be applied to an MPS (pure state) or MPO (density
matrix) representation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from ...backend import active_backend
from ...tensor.mps import MPS
from ...tensor.mpo import MPO
from .. import gates


# ======================================================================
# Noise parameter dataclass
# ======================================================================

@dataclass
class NoiseParams:
    """Noise parameters for a silicon spin qubit device.

    Parameters
    ----------
    t1_times : list of float
        T1 relaxation times for each qubit (seconds).
    t2_star_times : list of float
        T2* dephasing times for each qubit (seconds).
    t2_echo_times : list of float
        T2 (Hahn echo) dephasing times for each qubit (seconds).
    charge_noise_amplitude : float
        1/f charge noise amplitude (V/sqrtHz at 1 Hz).
    charge_noise_correlation_length : int
        Spatial correlation length of charge noise (in qubit spacings).
    thermal_photon_number : float
        Mean thermal photon number n_th = 1/(exp(hbaromega/kT) - 1).
    leakage_rate : float
        Rate of leakage to non-computational states (Hz).
    measurement_fidelity : float
        Single-shot readout fidelity.
    gate_error_rates : dict
        Error rates for different gate types.
    dephasing_model : str
        Dephasing decay model: "exponential" or "gaussian".
        Gaussian is more physically accurate for nuclear-spin-induced
        dephasing (Weinstein et al., Nature 2023).
    crosstalk_decay_exponent : float
        Spatial decay exponent for capacitive crosstalk (default 3.0
        for 1/r^3 decay from classical cross-capacitance).
    crosstalk_amplitude : float
        Crosstalk coupling strength at unit distance (Hz).
    exchange_frequency : float
        Typical exchange frequency J/h (Hz) for the device.
    pulse_duration : float
        Typical gate pulse duration (seconds).
    idle_duration : float
        Typical inter-pulse idle time (seconds).
    dfs_encoded : bool
        Whether qubits use decoherence-free subspace encoding
        (3 physical spins per logical qubit). Enables quadratic
        error suppression for uniform magnetic noise.
    dfs_leakage_rate : float
        Rate of leakage from DFS-encoded subspace to gauge states (Hz).
        Caused by magnetic field gradients.
    n_exchange_oscillations : float
        Number of coherent exchange oscillations N_osc before decay.
        Characterises the exchange gate quality factor.
    """
    t1_times: Optional[List[float]] = None
    t2_star_times: Optional[List[float]] = None
    t2_echo_times: Optional[List[float]] = None
    charge_noise_amplitude: float = 1e-6  # V/sqrtHz
    charge_noise_correlation_length: int = 2
    thermal_photon_number: float = 0.01
    leakage_rate: float = 0.0
    measurement_fidelity: float = 0.99
    gate_error_rates: Optional[dict] = None
    dephasing_model: str = "exponential"
    crosstalk_decay_exponent: float = 3.0
    crosstalk_amplitude: float = 0.0
    exchange_frequency: float = 100e6  # 100 MHz default
    pulse_duration: float = 10e-9  # 10 ns
    idle_duration: float = 10e-9  # 10 ns
    dfs_encoded: bool = False
    dfs_leakage_rate: float = 0.0
    n_exchange_oscillations: float = 50.0

    def __post_init__(self):
        if self.gate_error_rates is None:
            self.gate_error_rates = {
                "single": 1e-4,
                "two": 1e-3,
                "readout": 1 - self.measurement_fidelity,
            }

    def compute_exchange_gate_error(self) -> float:
        """Compute exchange gate error from N_osc and pulse duration.

        For DFS-encoded qubits, error scales as (t_gate/T2*)^2
        (quadratic suppression). For bare qubits, error scales as
        t_gate/T2* (linear).

        Based on Weinstein et al., Nature 615, 817-822 (2023):
        IRB error eps = (120/129) x (t/T2*)^2 for idle operations.
        """
        if self.t2_star_times is None or len(self.t2_star_times) == 0:
            return self.gate_error_rates.get("two", 1e-3)

        t2_star = np.mean(self.t2_star_times)
        t_gate = self.pulse_duration

        if self.dfs_encoded:
            # Quadratic suppression: eps ~ (t_gate/T2*)^2
            epsilon = (120.0 / 129.0) * (t_gate / t2_star) ** 2
        else:
            # Linear scaling: eps ~ t_gate/T2*
            epsilon = t_gate / t2_star

        return float(epsilon)

    def compute_idle_error(self, idle_time: float) -> float:
        """Compute error accumulated during idle time.

        For DFS-encoded qubits, idle error also scales quadratically.
        """
        if self.t2_star_times is None or len(self.t2_star_times) == 0:
            return 0.0

        t2_star = np.mean(self.t2_star_times)

        if self.dfs_encoded:
            return (idle_time / t2_star) ** 2
        else:
            return idle_time / t2_star


# ======================================================================
# Default noise parameter presets
# ======================================================================

def default_noise_params(n_qubits: int, device_type: str = "donor") -> NoiseParams:
    """Create default noise parameters for a given device type."""
    if device_type == "donor":
        return NoiseParams(
            t1_times=[30.0] * n_qubits,        # 30 s (P:Si)
            t2_star_times=[0.5e-3] * n_qubits,  # 0.5 ms
            t2_echo_times=[1.2e-3] * n_qubits,  # 1.2 ms
            charge_noise_amplitude=0.5e-6,
            measurement_fidelity=0.994,
            dephasing_model="exponential",
            exchange_frequency=18e6,
            pulse_duration=1e-6,
            idle_duration=100e-9,
            n_exchange_oscillations=30.0,
        )
    elif device_type == "simos":
        return NoiseParams(
            t1_times=[10.0] * n_qubits,
            t2_star_times=[20e-6] * n_qubits,
            t2_echo_times=[100e-6] * n_qubits,
            charge_noise_amplitude=2e-6,
            measurement_fidelity=0.985,
            dephasing_model="exponential",
            exchange_frequency=12e6,
            pulse_duration=200e-9,
            idle_duration=50e-9,
            n_exchange_oscillations=20.0,
        )
    elif device_type == "gaa":
        return NoiseParams(
            t1_times=[5.0] * n_qubits,
            t2_star_times=[10e-6] * n_qubits,
            t2_echo_times=[50e-6] * n_qubits,
            charge_noise_amplitude=3e-6,
            measurement_fidelity=0.975,
            dephasing_model="exponential",
            exchange_frequency=20e6,
            pulse_duration=50e-9,
            idle_duration=30e-9,
            n_exchange_oscillations=15.0,
        )
    elif device_type == "sledge":
        # -- HRL SLEDGE device (Weinstein et al., Nature 2023) --
        # Exchange-only encoded qubits in Si/SiGe quantum dots
        # DFS encoding with 3 physical spins per logical qubit
        return NoiseParams(
            t1_times=[100.0] * n_qubits,         # Very long T1 in Si/SiGe
            t2_star_times=[3.5e-6] * n_qubits,   # 3.5 us (Gaussian decay)
            t2_echo_times=[30e-6] * n_qubits,    # ~30 us with echo
            charge_noise_amplitude=1e-6,          # Moderate charge noise
            charge_noise_correlation_length=2,
            measurement_fidelity=0.960,           # 96% SPAM fidelity
            dephasing_model="gaussian",           # Gaussian from nuclear spins
            crosstalk_decay_exponent=3.0,         # 1/r^3 capacitive crosstalk
            crosstalk_amplitude=1e3,              # ~1 kHz at unit distance
            exchange_frequency=100e6,             # J/h ~ 100 MHz
            pulse_duration=10e-9,                 # 10 ns exchange pulses
            idle_duration=10e-9,                  # 5-30 ns idle (use 10 ns)
            dfs_encoded=True,                     # DFS encoding active
            dfs_leakage_rate=1e4,                 # Leakage from gradients
            n_exchange_oscillations=57.6,         # N_osc ~ 57.6 at 100 MHz
            gate_error_rates={
                "single": 1.1e-3,                 # (1.1+/-0.1)x10-^3 (RB)
                "two": 3.7e-2,                    # FW-CNOT: 96.3% fidelity
                "readout": 0.04,                  # ~96% SPAM
                "leakage": 3e-4,                  # (3+/-1)x10-^4
                "encoded_swap": 7e-3,             # 99.3% fidelity
                "lccz": 6.2e-2,                   # 93.8% fidelity
                "clifford_2q": 2.9e-2,            # 97.1% fidelity (RB)
            },
        )
    else:
        raise ValueError(f"Unknown device type: {device_type}")


# ======================================================================
# Kraus operators
# ======================================================================

def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """Amplitude damping (T1 relaxation) Kraus operators.

    gamma = 1 - exp(-dt/T1)
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def phase_damping_kraus(gamma: float) -> List[np.ndarray]:
    """Phase damping (T2 dephasing) Kraus operators.

    gamma = 1 - exp(-dt/T2) for exponential decay, or
    gamma = 1 - exp(-(dt/T2*)^2) for Gaussian decay.
    """
    K0 = np.array(
        [[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128
    )
    K1 = np.array(
        [[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128
    )
    return [K0, K1]


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """Depolarizing channel Kraus operators.

    rho -> (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ)
    """
    K0 = np.sqrt(1 - p) * gates.PAULI_I
    K1 = np.sqrt(p / 3) * gates.PAULI_X
    K2 = np.sqrt(p / 3) * gates.PAULI_Y
    K3 = np.sqrt(p / 3) * gates.PAULI_Z
    return [K0, K1, K2, K3]


def leakage_kraus(p_leak: float) -> List[np.ndarray]:
    """Leakage channel Kraus operators for DFS-encoded qubits.

    Models leakage from the computational subspace {|0>, |1>} to a
    leaked state. In the 2-level approximation, leakage maps the
    qubit state to a mixed state with probability p_leak.

    For DFS encoding, leakage is caused by magnetic field gradients
    that break the degeneracy of the encoded subspace.

    Parameters
    ----------
    p_leak : float
        Probability of leakage per gate/time step.
    """
    # K0: stay in computational subspace
    K0 = np.sqrt(1 - p_leak) * gates.PAULI_I
    # K1: leak to |0> (depolarize - effective model of leakage)
    K1 = np.sqrt(p_leak / 2) * np.array(
        [[1, 0], [0, 0]], dtype=np.complex128
    )
    # K2: leak to |1>
    K2 = np.sqrt(p_leak / 2) * np.array(
        [[0, 0], [0, 1]], dtype=np.complex128
    )
    return [K0, K1, K2]


# ======================================================================
# Kraus-to-MPO conversion
# ======================================================================

def kraus_to_mpo_tensor(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Convert Kraus operators to a single-site MPO tensor.

    The channel eps(rho) = Sum_k K_k rho K_kdag is represented as an MPO
    with bond dimension equal to the number of Kraus operators.

    Returns a rank-4 tensor of shape (1, d, d, 1) that implements
    the channel as a superoperator.
    """
    be = active_backend()
    d = kraus_ops[0].shape[0]

    # Build the superoperator: S[s,t] = Sum_k K_k[s,s'] K_k*[t,t']
    superop = be.zeros((d, d, d, d))
    for K in kraus_ops:
        K = be.array(K)
        K_conj = be.conj(K)
        superop = superop + be.einsum("ac,bd->abcd", K, K_conj)

    return be.to_numpy(be.reshape(superop, (1, d**2, d**2, 1)))


# ======================================================================
# Dephasing gamma computation (exponential vs Gaussian)
# ======================================================================

def compute_dephasing_gamma(
    dt: float,
    t2_star: float,
    model: str = "exponential",
) -> float:
    """Compute the dephasing parameter gamma for a given time step.

    Parameters
    ----------
    dt : float
        Time step (seconds).
    t2_star : float
        T2* dephasing time (seconds).
    model : str
        "exponential": gamma = 1 - exp(-dt/T2*)
            Standard Markovian dephasing.
        "gaussian": gamma = 1 - exp(-(dt/T2*)^2)
            Non-Markovian dephasing from quasi-static nuclear spin bath.
            More physically accurate for silicon spin qubits
            (Weinstein et al., Nature 2023).

    Returns
    -------
    float
        Dephasing parameter gamma in [0, 1].
    """
    if model == "gaussian":
        return 1.0 - np.exp(-(dt / t2_star) ** 2)
    else:
        return 1.0 - np.exp(-dt / t2_star)


# ======================================================================
# Crosstalk noise model
# ======================================================================

class CrosstalkModel:
    """Capacitive crosstalk noise model for silicon spin qubit arrays.

    Models the classical cross-capacitance between metal gates that
    causes spurious exchange coupling between non-adjacent qubits.
    The crosstalk falls off as 1/r^alpha where r is the physical distance
    between qubits and alpha is typically 3 (Weinstein et al., Nature 2023).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    qubit_positions : list of tuple
        Physical (x, y) coordinates of each qubit (nm).
    amplitude : float
        Crosstalk coupling strength at unit distance (Hz).
    decay_exponent : float
        Spatial decay exponent (default 3.0 for 1/r^3).
    """

    def __init__(
        self,
        n_qubits: int,
        qubit_positions: Optional[List[Tuple[float, float]]] = None,
        amplitude: float = 1e3,
        decay_exponent: float = 3.0,
    ):
        self.n_qubits = n_qubits
        self.amplitude = amplitude
        self.decay_exponent = decay_exponent

        if qubit_positions is None:
            # Default: linear chain with 80 nm spacing
            qubit_positions = [(i * 80.0, 0.0) for i in range(n_qubits)]
        self.positions = qubit_positions

        # Precompute the crosstalk matrix
        self._crosstalk_matrix = self._build_crosstalk_matrix()

    def _build_crosstalk_matrix(self) -> np.ndarray:
        """Build the NxN crosstalk coupling matrix.

        C[i,j] = amplitude / r_{ij}^alpha for i != j, 0 for i = j.
        """
        n = self.n_qubits
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.positions[i][0] - self.positions[j][0]
                    dy = self.positions[i][1] - self.positions[j][1]
                    r = np.sqrt(dx**2 + dy**2)
                    if r > 0:
                        # Normalise to unit spacing (80 nm)
                        r_norm = r / 80.0
                        C[i, j] = self.amplitude / (r_norm ** self.decay_exponent)
        return C

    @property
    def crosstalk_matrix(self) -> np.ndarray:
        """The NxN crosstalk coupling matrix (Hz)."""
        return self._crosstalk_matrix

    def get_crosstalk_dephasing(
        self, active_qubit: int, dt: float
    ) -> np.ndarray:
        """Compute crosstalk-induced dephasing angles for all qubits
        when a gate is applied to active_qubit.

        Parameters
        ----------
        active_qubit : int
            Index of the qubit being actively pulsed.
        dt : float
            Duration of the active pulse (seconds).

        Returns
        -------
        ndarray of shape (n_qubits,)
            Phase errors (radians) on each qubit due to crosstalk.
        """
        phases = np.zeros(self.n_qubits)
        for j in range(self.n_qubits):
            if j != active_qubit:
                # Crosstalk-induced frequency shift -> phase error
                phases[j] = 2 * np.pi * self._crosstalk_matrix[active_qubit, j] * dt
        return phases


# ======================================================================
# 1/f charge noise generator
# ======================================================================

class ChargeNoiseGenerator:
    """Generates correlated 1/f charge noise for silicon spin qubits.

    The noise has a power spectral density S(f) ~ 1/f^alpha with
    spatial correlations between nearby qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    amplitude : float
        Noise amplitude (V/sqrtHz at 1 Hz).
    alpha : float
        Spectral exponent (1.0 for pure 1/f noise).
    correlation_length : int
        Spatial correlation length in qubit spacings.
    dt : float
        Time step for noise generation (seconds).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int,
        amplitude: float = 1e-6,
        alpha: float = 1.0,
        correlation_length: int = 2,
        dt: float = 1e-9,
        seed: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.amplitude = amplitude
        self.alpha = alpha
        self.correlation_length = correlation_length
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        self._buffer_size = 1024
        self._buffer = None
        self._buffer_idx = 0

    def _generate_buffer(self):
        """Generate a buffer of correlated 1/f noise samples."""
        n = self._buffer_size
        nq = self.n_qubits

        # Generate white noise
        white = self.rng.randn(nq, n)

        # Apply 1/f^(alpha/2) filter in frequency domain
        freqs = np.fft.rfftfreq(n, d=self.dt)
        freqs[0] = freqs[1]  # avoid division by zero
        filt = 1.0 / np.power(freqs, self.alpha / 2)
        filt[0] = 0  # remove DC component

        colored = np.zeros_like(white)
        for q in range(nq):
            spectrum = np.fft.rfft(white[q])
            spectrum *= filt
            colored[q] = np.fft.irfft(spectrum, n=n)

        # Apply spatial correlations
        corr_matrix = np.zeros((nq, nq))
        for i in range(nq):
            for j in range(nq):
                dist = abs(i - j)
                corr_matrix[i, j] = np.exp(-dist / self.correlation_length)

        # Cholesky decomposition for correlated noise
        L = np.linalg.cholesky(corr_matrix)
        correlated = L @ colored

        self._buffer = self.amplitude * correlated
        self._buffer_idx = 0

    def sample(self) -> np.ndarray:
        """Get the next noise sample for all qubits.

        Returns
        -------
        ndarray of shape (n_qubits,)
            Noise values for each qubit at the current time step.
        """
        if self._buffer is None or self._buffer_idx >= self._buffer_size:
            self._generate_buffer()

        sample = self._buffer[:, self._buffer_idx]
        self._buffer_idx += 1
        return sample

    def reset(self, seed: Optional[int] = None):
        """Reset the noise generator."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._buffer = None
        self._buffer_idx = 0


# ======================================================================
# DFS encoding noise model
# ======================================================================

class DFSNoiseModel:
    """Decoherence-free subspace (DFS) noise model for exchange-only
    encoded qubits.

    In DFS encoding, each logical qubit is encoded in 3 physical spins.
    The encoding is immune to uniform (global) magnetic field
    fluctuations, but susceptible to:
        1. Magnetic field GRADIENTS (cause leakage out of DFS)
        2. Charge noise (affects exchange coupling J)
        3. Crosstalk (classical cross-capacitance)

    The key advantage is quadratic error suppression:
        eps_DFS ~ (t_gate/T2*)^2  vs  eps_bare ~ t_gate/T2*

    Based on Weinstein et al., Nature 615, 817-822 (2023).

    Parameters
    ----------
    n_logical_qubits : int
        Number of logical (encoded) qubits.
    t2_star : float
        T2* dephasing time of the physical spins (seconds).
    gradient_strength : float
        Magnetic field gradient g|dB/dx| (Hz/nm).
        Measured as ~0.04 kHz/mT/nm in the SLEDGE device.
    leakage_rate : float
        Rate of leakage from encoded to gauge states (Hz).
    """

    def __init__(
        self,
        n_logical_qubits: int,
        t2_star: float = 3.5e-6,
        gradient_strength: float = 40.0,  # Hz/nm (0.04 kHz/mT/nm)
        leakage_rate: float = 1e4,
    ):
        self.n_logical = n_logical_qubits
        self.n_physical = 3 * n_logical_qubits
        self.t2_star = t2_star
        self.gradient_strength = gradient_strength
        self.leakage_rate = leakage_rate

    def compute_gate_error(self, t_gate: float) -> float:
        """Compute the DFS-suppressed gate error.

        eps = (120/129) x (t_gate/T2*)^2

        This formula is from the IRB analysis in Weinstein et al.
        The coefficient 120/129 comes from the convolution method
        of Merkel et al.
        """
        return (120.0 / 129.0) * (t_gate / self.t2_star) ** 2

    def compute_idle_error(self, t_idle: float) -> float:
        """Compute error during idle time with DFS protection.

        During idle, exchange rates are very low (J_idle/h < 10 kHz),
        so the DFS protection is maintained and error scales
        quadratically.
        """
        return (t_idle / self.t2_star) ** 2

    def compute_leakage_probability(self, t_gate: float) -> float:
        """Compute probability of leakage from DFS to gauge states.

        Leakage is caused by the FW-CNOT gate spreading leakage
        from one encoded qubit to another. The LCCZ gate was
        designed to reduce this to ~equal probability as direct
        leakage.

        For the SLEDGE device:
            P_leakage ~ (3 +/- 1) x 10-^4 per single-qubit Clifford
        """
        return self.leakage_rate * t_gate

    def compute_gradient_dephasing(
        self, qubit_spacing: float, t: float
    ) -> float:
        """Compute dephasing from magnetic field gradients.

        Gradients break the DFS protection by creating different
        Larmor frequencies for the 3 physical spins within each
        encoded qubit.

        Parameters
        ----------
        qubit_spacing : float
            Physical spacing between dots within the encoded qubit (nm).
        t : float
            Evolution time (seconds).

        Returns
        -------
        float
            Dephasing angle (radians).
        """
        # Frequency difference between adjacent physical spins
        delta_f = self.gradient_strength * qubit_spacing
        return 2 * np.pi * delta_f * t


# ======================================================================
# Noise application functions
# ======================================================================

def apply_t1_noise(
    mps: MPS, site: int, dt: float, t1: float
) -> MPS:
    """Apply T1 relaxation to a single site of an MPS.

    This is an approximation: we apply the Kraus operators
    stochastically (quantum trajectory method).
    """
    be = active_backend()
    gamma = 1 - np.exp(-dt / t1)
    kraus = amplitude_damping_kraus(gamma)

    # Quantum trajectory: choose Kraus operator probabilistically
    A = mps[site]  # (chi_l, d, chi_r)

    # Compute probabilities
    probs = []
    new_states = []
    for K in kraus:
        K_arr = be.array(K)
        new_A = be.einsum("adb,cd->acb", A, K_arr)
        p = float(be.norm(new_A)) ** 2
        probs.append(p)
        new_states.append(new_A)

    probs = np.array(probs)
    probs /= probs.sum()

    # Sample
    idx = np.random.choice(len(kraus), p=probs)
    new_mps = mps.copy()
    new_A = new_states[idx]
    norm = be.norm(new_A)
    if norm > 1e-15:
        new_A = be.array(be.to_numpy(new_A) / norm)
    new_mps[site] = new_A
    return new_mps


def apply_dephasing_noise(
    mps: MPS,
    site: int,
    dt: float,
    t2: float,
    model: str = "exponential",
) -> MPS:
    """Apply T2 dephasing to a single site of an MPS.

    Parameters
    ----------
    mps : MPS
        Current quantum state.
    site : int
        Qubit index.
    dt : float
        Time step (seconds).
    t2 : float
        T2* dephasing time (seconds).
    model : str
        "exponential" or "gaussian" dephasing model.
    """
    be = active_backend()
    gamma = compute_dephasing_gamma(dt, t2, model)
    kraus = phase_damping_kraus(gamma)

    A = mps[site]
    probs = []
    new_states = []
    for K in kraus:
        K_arr = be.array(K)
        new_A = be.einsum("adb,cd->acb", A, K_arr)
        p = float(be.norm(new_A)) ** 2
        probs.append(p)
        new_states.append(new_A)

    probs = np.array(probs)
    probs /= probs.sum()

    idx = np.random.choice(len(kraus), p=probs)
    new_mps = mps.copy()
    new_A = new_states[idx]
    norm = be.norm(new_A)
    if norm > 1e-15:
        new_A = be.array(be.to_numpy(new_A) / norm)
    new_mps[site] = new_A
    return new_mps


def apply_leakage_noise(
    mps: MPS,
    site: int,
    p_leak: float,
) -> MPS:
    """Apply leakage noise to a single site of an MPS.

    Models leakage from the computational subspace using a
    depolarization-like channel.

    Parameters
    ----------
    mps : MPS
        Current quantum state.
    site : int
        Qubit index.
    p_leak : float
        Leakage probability.
    """
    if p_leak <= 0:
        return mps

    be = active_backend()
    kraus = leakage_kraus(p_leak)

    A = mps[site]
    probs = []
    new_states = []
    for K in kraus:
        K_arr = be.array(K)
        new_A = be.einsum("adb,cd->acb", A, K_arr)
        p = float(be.norm(new_A)) ** 2
        probs.append(p)
        new_states.append(new_A)

    probs = np.array(probs)
    probs /= probs.sum()

    idx = np.random.choice(len(kraus), p=probs)
    new_mps = mps.copy()
    new_A = new_states[idx]
    norm = be.norm(new_A)
    if norm > 1e-15:
        new_A = be.array(be.to_numpy(new_A) / norm)
    new_mps[site] = new_A
    return new_mps


def apply_crosstalk_noise(
    mps: MPS,
    crosstalk_model: CrosstalkModel,
    active_qubit: int,
    dt: float,
) -> MPS:
    """Apply crosstalk-induced dephasing to all non-active qubits.

    When a gate is applied to active_qubit, the capacitive crosstalk
    causes spurious phase rotations on neighbouring qubits.

    Parameters
    ----------
    mps : MPS
        Current quantum state.
    crosstalk_model : CrosstalkModel
        The crosstalk model with precomputed coupling matrix.
    active_qubit : int
        Index of the qubit being actively pulsed.
    dt : float
        Duration of the active pulse.
    """
    be = active_backend()
    phases = crosstalk_model.get_crosstalk_dephasing(active_qubit, dt)

    new_mps = mps.copy()
    for q in range(mps.n_sites):
        if q != active_qubit and abs(phases[q]) > 1e-15:
            Rz = be.array(gates.rz(phases[q]))
            A = new_mps[q]
            new_mps[q] = be.einsum("adb,cd->acb", A, Rz)

    return new_mps


def apply_charge_noise_dephasing(
    mps: MPS,
    noise_values: np.ndarray,
    dt: float,
    sensitivity: float = 1e9,
) -> MPS:
    """Apply charge-noise-induced dephasing to all qubits.

    The charge noise causes fluctuations in the qubit frequency,
    leading to pure dephasing: each qubit acquires a random phase
    proportional to the noise value.

    For DFS-encoded qubits, charge noise acts during both pulsing
    and idling, with error scaling as (t_gate/T2_charge*)^2
    (Weinstein et al., Nature 2023).

    Parameters
    ----------
    mps : MPS
        Current quantum state.
    noise_values : ndarray
        Charge noise values for each qubit (from ChargeNoiseGenerator).
    dt : float
        Time step.
    sensitivity : float
        Frequency sensitivity to charge noise (Hz/V).
    """
    be = active_backend()
    new_mps = mps.copy()

    for i in range(mps.n_sites):
        delta_phi = 2 * np.pi * sensitivity * noise_values[i] * dt
        Rz = be.array(gates.rz(delta_phi))
        A = new_mps[i]
        new_mps[i] = be.einsum("adb,cd->acb", A, Rz)

    return new_mps
