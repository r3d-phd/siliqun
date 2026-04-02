"""
Noise channels for silicon spin qubit systems.

Implements realistic decoherence models as Kraus operators and
MPO-based quantum channels:

    - T1 relaxation (amplitude damping)
    - T2 dephasing (phase damping)
    - 1/f charge noise (correlated, non-Markovian)
    - Johnson-Nyquist thermal noise
    - Leakage to non-computational states
    - Crosstalk-induced errors

Each channel can be applied to an MPS (pure state) or MPO (density
matrix) representation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ...backend import active_backend
from ...tensor.mps import MPS
from ...tensor.mpo import MPO
from .. import gates


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
        1/f charge noise amplitude (V/√Hz at 1 Hz).
    charge_noise_correlation_length : int
        Spatial correlation length of charge noise (in qubit spacings).
    thermal_photon_number : float
        Mean thermal photon number n_th = 1/(exp(ℏω/kT) - 1).
    leakage_rate : float
        Rate of leakage to non-computational states (Hz).
    measurement_fidelity : float
        Single-shot readout fidelity.
    gate_error_rates : dict
        Error rates for different gate types.
    """
    t1_times: Optional[List[float]] = None
    t2_star_times: Optional[List[float]] = None
    t2_echo_times: Optional[List[float]] = None
    charge_noise_amplitude: float = 1e-6  # V/√Hz
    charge_noise_correlation_length: int = 2
    thermal_photon_number: float = 0.01
    leakage_rate: float = 0.0
    measurement_fidelity: float = 0.99
    gate_error_rates: Optional[dict] = None

    def __post_init__(self):
        if self.gate_error_rates is None:
            self.gate_error_rates = {
                "single": 1e-4,
                "two": 1e-3,
                "readout": 1 - self.measurement_fidelity,
            }


def default_noise_params(n_qubits: int, device_type: str = "donor") -> NoiseParams:
    """Create default noise parameters for a given device type."""
    if device_type == "donor":
        return NoiseParams(
            t1_times=[30.0] * n_qubits,       # 30 s (P:Si)
            t2_star_times=[0.5e-3] * n_qubits, # 0.5 ms
            t2_echo_times=[1.2e-3] * n_qubits, # 1.2 ms
            charge_noise_amplitude=0.5e-6,
            measurement_fidelity=0.994,
        )
    elif device_type == "simos":
        return NoiseParams(
            t1_times=[10.0] * n_qubits,
            t2_star_times=[20e-6] * n_qubits,
            t2_echo_times=[100e-6] * n_qubits,
            charge_noise_amplitude=2e-6,
            measurement_fidelity=0.985,
        )
    elif device_type == "gaa":
        return NoiseParams(
            t1_times=[5.0] * n_qubits,
            t2_star_times=[10e-6] * n_qubits,
            t2_echo_times=[50e-6] * n_qubits,
            charge_noise_amplitude=3e-6,
            measurement_fidelity=0.975,
        )
    else:
        raise ValueError(f"Unknown device type: {device_type}")


# ── Kraus operators ─────────────────────────────────────────────────

def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """Amplitude damping (T1 relaxation) Kraus operators.

    γ = 1 - exp(-dt/T1)
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def phase_damping_kraus(gamma: float) -> List[np.ndarray]:
    """Phase damping (T2 dephasing) Kraus operators.

    γ = 1 - exp(-dt/T2)
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

    ρ → (1-p)ρ + p/3 (XρX + YρY + ZρZ)
    """
    K0 = np.sqrt(1 - p) * gates.PAULI_I
    K1 = np.sqrt(p / 3) * gates.PAULI_X
    K2 = np.sqrt(p / 3) * gates.PAULI_Y
    K3 = np.sqrt(p / 3) * gates.PAULI_Z
    return [K0, K1, K2, K3]


# ── Kraus-to-MPO conversion ────────────────────────────────────────

def kraus_to_mpo_tensor(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Convert Kraus operators to a single-site MPO tensor.

    The channel ε(ρ) = Σ_k K_k ρ K_k† is represented as an MPO
    with bond dimension equal to the number of Kraus operators.

    Returns a rank-4 tensor of shape (1, d, d, 1) that implements
    the channel as a superoperator.
    """
    be = active_backend()
    d = kraus_ops[0].shape[0]

    # Build the superoperator: S[s,t] = Σ_k K_k[s,s'] K_k*[t,t']
    # where s = (s_out, s_in) and t = (t_out, t_in)
    superop = be.zeros((d, d, d, d))
    for K in kraus_ops:
        K = be.array(K)
        K_conj = be.conj(K)
        superop = superop + be.einsum("ac,bd->abcd", K, K_conj)

    return be.to_numpy(be.reshape(superop, (1, d**2, d**2, 1)))


# ── 1/f charge noise generator ─────────────────────────────────────

class ChargeNoiseGenerator:
    """Generates correlated 1/f charge noise for silicon spin qubits.

    The noise has a power spectral density S(f) ∝ 1/f^α with
    spatial correlations between nearby qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    amplitude : float
        Noise amplitude (V/√Hz at 1 Hz).
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

        # Apply 1/f^(α/2) filter in frequency domain
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


# ── Noise application functions ─────────────────────────────────────

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
    A = mps[site]  # (χ_l, d, χ_r)

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
    mps: MPS, site: int, dt: float, t2: float
) -> MPS:
    """Apply T2 dephasing to a single site of an MPS."""
    be = active_backend()
    gamma = 1 - np.exp(-dt / t2)
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
