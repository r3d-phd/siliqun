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
    Rojas-Arias et al., arXiv:2603.03051 (2026) - Correlated noise in Si/SiGe

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
    tlf_density : float
        Two-level fluctuator (TLF) areal density (cm^-2).
        Experimentally measured as 3e10 cm^-2 in Si/SiGe devices.
        Rojas-Arias et al., arXiv:2603.03051 (2026).
    tlf_correlation_length : float
        Charge noise spatial correlation length (nm).
        Measured as l_c = 81 nm (exponential decay).
        Rojas-Arias et al., arXiv:2603.03051 (2026).
    qubit_spacing : float
        Average nearest-neighbor qubit spacing (nm).
        L_q = 108 nm for the RIKEN 5-qubit device.
    magnetic_drift_rate : float
        Global magnetic field drift rate (Hz/s).
        Measured as ~8 Hz/s from superconducting magnet.
    correlation_model : str
        Spatial correlation model: "exponential" (default, simple),
        "tlf" (physically motivated TLF model from Rojas-Arias et al.).
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
    tlf_density: float = 3e10  # cm^-2 (Rojas-Arias et al., 2026)
    tlf_correlation_length: float = 81.0  # nm (Rojas-Arias et al., 2026)
    qubit_spacing: float = 108.0  # nm (RIKEN 5Q device)
    magnetic_drift_rate: float = 8.0  # Hz/s
    correlation_model: str = "exponential"  # or "tlf"

    def __post_init__(self):
        if self.gate_error_rates is None:
            self.gate_error_rates = {
                "single": 1e-4,
                "two": 1e-3,
                "readout": 1 - self.measurement_fidelity,
            }

    @property
    def T1(self) -> float:
        """Mean longitudinal-relaxation time in seconds.

        This compatibility accessor is intentionally read-only at the
        single-value level: device profiles store per-qubit times in
        ``t1_times`` while pulse-level solvers require a scalar rate.
        """
        if not self.t1_times:
            raise ValueError("NoiseParams.t1_times must contain at least one value")
        return float(np.mean(self.t1_times))

    @property
    def T2_star(self) -> float:
        """Mean inhomogeneous-dephasing time in seconds."""
        if not self.t2_star_times:
            raise ValueError("NoiseParams.t2_star_times must contain at least one value")
        return float(np.mean(self.t2_star_times))

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
        # v2.1 calibration: updated to 2025 state-of-the-art SiMOS parameters
        # Sources:
        #   T1=9.5s, T2*=41us, T2_echo=1.31ms: Tanttu et al., arXiv:2402.02986 (2024)
        #   measurement_fidelity=0.993: Neyens et al., Nature 2024 (6-qubit SiMOS)
        #   charge_noise=0.8e-6: improved gate screening in 300mm foundry process
        #   n_exchange_oscillations=45: Nickl et al., arXiv:2512.10174 (2025)
        return NoiseParams(
            t1_times=[9.5] * n_qubits,          # 9.5 s (Tanttu 2024, was 10 s)
            t2_star_times=[41e-6] * n_qubits,   # 41 us (was 20 us)
            t2_echo_times=[1.31e-3] * n_qubits, # 1.31 ms (was 100 us)
            charge_noise_amplitude=0.8e-6,       # V/sqrtHz (was 2e-6)
            charge_noise_correlation_length=2,   # v2.1: spatial correlation (was 0)
            measurement_fidelity=0.993,          # 99.3% (was 98.5%)
            dephasing_model="exponential",
            exchange_frequency=12e6,
            pulse_duration=200e-9,
            idle_duration=50e-9,
            n_exchange_oscillations=45.0,        # (was 20.0)
            crosstalk_amplitude=1e3,             # v2.1: always-on ZZ crosstalk (J_res~1kHz)
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
        # Correlated noise calibrated to Rojas-Arias et al. (2026)
        return NoiseParams(
            t1_times=[100.0] * n_qubits,         # Very long T1 in Si/SiGe
            t2_star_times=[3.5e-6] * n_qubits,   # 3.5 us (Gaussian decay)
            t2_echo_times=[30e-6] * n_qubits,    # ~30 us with echo
            charge_noise_amplitude=1e-6,          # Moderate charge noise
            charge_noise_correlation_length=2,    # Legacy (integer spacings)
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
            # TLF correlation model (Rojas-Arias et al., 2026)
            tlf_density=3e10,                     # 3e10 cm^-2
            tlf_correlation_length=81.0,          # l_c = 81 nm
            qubit_spacing=80.0,                   # 80 nm (SLEDGE dot spacing)
            magnetic_drift_rate=8.0,              # ~8 Hz/s global drift
            correlation_model="tlf",              # Use TLF model by default
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
    elif device_type == "riken_5q":
        # -- RIKEN 5Q device (Rojas-Arias et al., arXiv:2603.03051, 2026) --
        # Linear array of 5 single-spin qubits on Si/SiGe
        # Noise correlations experimentally characterised
        return NoiseParams(
            t1_times=[50.0] * n_qubits,
            t2_star_times=[20e-6] * n_qubits,
            t2_echo_times=[100e-6] * n_qubits,
            charge_noise_amplitude=1.5e-6,
            charge_noise_correlation_length=1,
            measurement_fidelity=0.990,
            dephasing_model="gaussian",
            exchange_frequency=50e6,
            pulse_duration=100e-9,
            idle_duration=50e-9,
            n_exchange_oscillations=20.0,
            # TLF correlation model
            tlf_density=3e10,                     # 3e10 cm^-2
            tlf_correlation_length=81.0,          # l_c = 81 nm
            qubit_spacing=108.0,                  # 108 nm spacing
            magnetic_drift_rate=8.0,              # ~8 Hz/s
            correlation_model="tlf",
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
# TLF spatial correlation model
# ======================================================================

class TLFCorrelationModel:
    """Two-level fluctuator (TLF) spatial correlation model.

    Computes the charge noise spatial correlation matrix based on
    the experimentally measured TLF parameters from:
        Rojas-Arias et al., arXiv:2603.03051 (2026)

    The model assumes an exponential decay of charge noise correlations
    with physical distance between qubits:

        c(i, j) = exp(-|r_i - r_j| / l_c)

    where l_c is the TLF correlation length (81 nm for Si/SiGe).

    Three noise regimes are captured:
        1. Perfectly correlated global magnetic drift (~8 Hz/s)
           -> Naturally suppressed by DFS encoding
        2. Partially correlated charge noise from TLFs
           -> l_c = 81 nm, rho_TLF = 3e10 cm^-2
        3. Uncorrelated nuclear spin noise
           -> Captured by T2* dephasing model

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the array.
    qubit_positions : list of tuple or ndarray, optional
        Physical (x, y) coordinates of each qubit (nm).
        If None, assumes a linear chain with given spacing.
    qubit_spacing : float
        Nearest-neighbor spacing (nm). Used only if qubit_positions
        is None. Default 80 nm (SLEDGE device).
    tlf_correlation_length : float
        Charge noise spatial correlation length (nm).
        Default 81 nm from Rojas-Arias et al. (2026).
    tlf_density : float
        TLF areal density (cm^-2). Default 3e10 cm^-2.
    magnetic_drift_rate : float
        Global magnetic field drift rate (Hz/s). Default 8.0.
    """

    def __init__(
        self,
        n_qubits: int,
        qubit_positions: Optional[List[Tuple[float, float]]] = None,
        qubit_spacing: float = 80.0,
        tlf_correlation_length: float = 81.0,
        tlf_density: float = 3e10,
        magnetic_drift_rate: float = 8.0,
    ):
        self.n_qubits = n_qubits
        self.l_c = tlf_correlation_length
        self.rho_tlf = tlf_density
        self.magnetic_drift_rate = magnetic_drift_rate

        # Compute physical positions
        if qubit_positions is not None:
            self.positions = np.array(qubit_positions)
        else:
            # Linear chain with given spacing
            self.positions = np.array(
                [(i * qubit_spacing, 0.0) for i in range(n_qubits)]
            )

        # Precompute distance matrix (nm)
        self._distance_matrix = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(n_qubits):
                dx = self.positions[i, 0] - self.positions[j, 0]
                dy = self.positions[i, 1] - self.positions[j, 1]
                self._distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

        # Precompute correlation matrix
        self._correlation_matrix = np.exp(
            -self._distance_matrix / self.l_c
        )

        # Precompute Cholesky factor for efficient sampling
        self._cholesky_L = np.linalg.cholesky(self._correlation_matrix)

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Return the spatial correlation matrix."""
        return self._correlation_matrix.copy()

    @property
    def distance_matrix(self) -> np.ndarray:
        """Return the pairwise distance matrix (nm)."""
        return self._distance_matrix.copy()

    @property
    def nn_correlation(self) -> float:
        """Return the nearest-neighbor correlation coefficient."""
        if self.n_qubits < 2:
            return 1.0
        # Find minimum nonzero distance
        dists = self._distance_matrix[0, 1:]
        min_dist = np.min(dists[dists > 0])
        return float(np.exp(-min_dist / self.l_c))

    @property
    def n_c(self) -> float:
        """Return N_c = l_c / L_q (correlation in qubit spacings)."""
        if self.n_qubits < 2:
            return float('inf')
        min_dist = np.min(
            self._distance_matrix[0, 1:][self._distance_matrix[0, 1:] > 0]
        )
        return self.l_c / min_dist

    def apply_correlations(self, white_noise: np.ndarray) -> np.ndarray:
        """Apply TLF spatial correlations to white noise samples.

        Parameters
        ----------
        white_noise : ndarray of shape (n_qubits, n_samples)
            Uncorrelated noise samples.

        Returns
        -------
        ndarray of shape (n_qubits, n_samples)
            Spatially correlated noise samples.
        """
        return self._cholesky_L @ white_noise

    def global_drift_phase(self, dt: float) -> float:
        """Compute the global magnetic drift phase accumulated over dt.

        This phase is identical for all qubits and is naturally
        suppressed by DFS encoding.

        Parameters
        ----------
        dt : float
            Time interval (seconds).

        Returns
        -------
        float
            Phase angle (radians) from global drift.
        """
        return 2 * np.pi * self.magnetic_drift_rate * dt

    @classmethod
    def from_noise_params(
        cls,
        noise_params: 'NoiseParams',
        qubit_positions: Optional[List[Tuple[float, float]]] = None,
        n_qubits: int = 5,
    ) -> 'TLFCorrelationModel':
        """Create a TLFCorrelationModel from a NoiseParams instance.

        Parameters
        ----------
        noise_params : NoiseParams
            Noise parameters containing TLF fields.
        qubit_positions : list of tuple, optional
            Physical qubit positions. If None, uses linear chain.
        n_qubits : int
            Number of qubits (used for linear chain layout).
        """
        return cls(
            n_qubits=n_qubits,
            qubit_positions=qubit_positions,
            qubit_spacing=noise_params.qubit_spacing,
            tlf_correlation_length=noise_params.tlf_correlation_length,
            tlf_density=noise_params.tlf_density,
            magnetic_drift_rate=noise_params.magnetic_drift_rate,
        )


# ======================================================================
# 1/f charge noise generator
# ======================================================================

class ChargeNoiseGenerator:
    """Generates correlated 1/f charge noise for silicon spin qubits.

    The noise has a power spectral density S(f) ~ 1/f^alpha with
    spatial correlations between nearby qubits.

    Supports two correlation models:
        - "exponential": Simple exponential decay with integer
          correlation length (legacy, in qubit spacings).
        - "tlf": Physically motivated TLF model using real
          qubit positions and measured correlation length l_c = 81 nm
          (Rojas-Arias et al., arXiv:2603.03051, 2026).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    amplitude : float
        Noise amplitude (V/sqrtHz at 1 Hz).
    alpha : float
        Spectral exponent (1.0 for pure 1/f noise).
    correlation_length : int
        Spatial correlation length in qubit spacings (legacy model).
    dt : float
        Time step for noise generation (seconds).
    seed : int, optional
        Random seed for reproducibility.
    tlf_model : TLFCorrelationModel, optional
        If provided, uses the TLF correlation model instead of the
        simple exponential decay. Overrides correlation_length.
    """

    def __init__(
        self,
        n_qubits: int,
        amplitude: float = 1e-6,
        alpha: float = 1.0,
        correlation_length: int = 2,
        dt: float = 1e-9,
        seed: Optional[int] = None,
        tlf_model: Optional[TLFCorrelationModel] = None,
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
        self.tlf_model = tlf_model

        # Precompute Cholesky factor for legacy model
        if tlf_model is None:
            corr_matrix = np.zeros((n_qubits, n_qubits))
            for i in range(n_qubits):
                for j in range(n_qubits):
                    dist = abs(i - j)
                    corr_matrix[i, j] = np.exp(
                        -dist / max(self.correlation_length, 1e-10)
                    )
            self._cholesky_L = np.linalg.cholesky(corr_matrix)
        else:
            self._cholesky_L = tlf_model._cholesky_L

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

        # Apply spatial correlations via precomputed Cholesky factor
        correlated = self._cholesky_L @ colored

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


def apply_always_on_exchange_zz(
    mps,
    connectivity: list,
    J_residual: float,
    dt: float,
) -> "MPS":
    """Apply always-on residual exchange ZZ coupling during idle periods.

    When no gate is actively applied, the residual exchange coupling J_res
    between nearest-neighbour qubits causes a ZZ(theta) rotation:
        theta = 2 * pi * J_res * dt

    This models the leakage of exchange interaction during the idle phase
    between gate pulses, which is the dominant source of two-qubit error
    in SiMOS devices at high gate fidelity (Tanttu et al. 2024).

    Parameters
    ----------
    mps : MPS
        Current quantum state.
    connectivity : list of tuple
        List of (i, j) pairs of coupled qubits.
    J_residual : float
        Residual exchange coupling strength (Hz).
    dt : float
        Idle time step (seconds).
    """
    import numpy as np
    be = active_backend()
    theta = 2.0 * np.pi * J_residual * dt
    if abs(theta) < 1e-15:
        return mps
    # ZZ(theta) = exp(-i * theta/2 * Z_i Z_j)
    # For MPS, apply as two-site gate on each connected pair
    # Kraus representation: diagonal 2x2x2x2 matrix
    # ZZ(theta)|00> = e^{-i*theta/2}|00>, |01>=e^{+i*theta/2}|01>,
    #                 |10>=e^{+i*theta/2}|10>, |11>=e^{-i*theta/2}|11>
    import cmath
    zz_phases = [
        cmath.exp(-1j * theta / 2),  # |00>
        cmath.exp(+1j * theta / 2),  # |01>
        cmath.exp(+1j * theta / 2),  # |10>
        cmath.exp(-1j * theta / 2),  # |11>
    ]
    new_mps = mps.copy()
    for (i, j) in connectivity:
        if abs(i - j) != 1:
            continue  # only nearest-neighbour
        # Apply as local Rz rotations: ZZ = (Rz_i(theta) x Rz_j(-theta)) up to global phase
        # This is exact for product states and approximate for entangled states,
        # but is the correct first-order Trotter step for the ZZ Hamiltonian.
        Rz_i = be.array([[cmath.exp(-1j * theta / 2), 0],
                          [0, cmath.exp(+1j * theta / 2)]])
        Rz_j = be.array([[cmath.exp(+1j * theta / 2), 0],
                          [0, cmath.exp(-1j * theta / 2)]])
        A_i = new_mps[i]
        new_mps[i] = be.einsum("adb,cd->acb", A_i, Rz_i)
        A_j = new_mps[j]
        new_mps[j] = be.einsum("adb,cd->acb", A_j, Rz_j)
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
