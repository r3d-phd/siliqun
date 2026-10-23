"""Shared physics core for SiliQun optimal control and RL evaluation.

2-qubit SiMOS spin system in the single-frame (rotating at the drive
frequencies) picture.  Per-segment Hamiltonian (angular-frequency units,
hbar = 1, basis |00>,|01>,|10>,|11>, |0> = spin up):

    H(t) = sum_i (delta_i / 2) Z_i                      (detuning residual)
         + (J(t) + J_res(t)) / 4 * (XX + YY + ZZ)       (exchange + residual)
         + sum_i (Omega_i(t) / 2) * [cos(phi_i) X - sin(phi_i) Y]  (EDSR)

Piecewise-constant controls over N segments of length dt.  Noise is
applied per segment: (i) residual-exchange / detuning draws enter H,
(ii) exponential phase damping with coherence eta = exp(-dt / T2*),
(iii) amplitude damping with p = 1 - exp(-dt / T1).

All device constants are taken from the SiliQun package itself
(``siliqun.physics``) so the classical and RL arms share one physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

# --------------------------------------------------------------------------
# Pauli matrices (package convention, imported for a single source of truth)
# --------------------------------------------------------------------------
from ..physics.gates import PAULI_X as _X
from ..physics.gates import PAULI_Y as _Y
from ..physics.gates import PAULI_Z as _Z
from ..physics.gates import PAULI_I as _I

Pauli = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# Hardware bounds (SiMOS v2.1): Rabi and exchange modulated within
# device-settable ranges ~1x the nominal exchange scale.
OMEGA_MAX_HZ = 12.0e6     # max EDSR Rabi frequency (1x J_nom)
J_MAX_HZ = 36.0e6         # max exchange (3x J_nom)

_I4 = np.eye(4, dtype=complex)
_XX = np.kron(_X, _X)
_YY = np.kron(_Y, _Y)
_ZZ = np.kron(_Z, _Z)
_EXCHANGE = (_XX + _YY + _ZZ) / 4.0          # J/4 (XX+YY+ZZ) basis
_Z0 = np.kron(_Z, _I)                          # detuning q0
_Z1 = np.kron(_I, _Z)                          # detuning q1
_X0 = np.kron(_X, _I)
_Y0 = np.kron(_Y, _I)
_X1 = np.kron(_I, _X)
_Y1 = np.kron(_I, _Y)

# population of bit 1 (|1...>) for each basis index, used for the
# per-segment phase-damping factor rho_{mn} *= eta^{(pop(m)+pop(n))/2}
_POP = np.array([0, 1, 1, 2])


def target_bell_plus() -> np.ndarray:
    """Target state |Phi+> = (|00> + |11>)/2 (the 2-qubit SiMOS CZ task)."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[3] = 1.0 / np.sqrt(2.0)
    return psi


def input_state_plus_plus() -> np.ndarray:
    """|++> = (|00> + |01> + |10> + |11>)/2."""
    return np.full(4, 0.5, dtype=complex)


@dataclass
class NoiseProfile:
    """A named noise scenario for the robustness sweep.

    All fields are the *parameters* of the per-realization draws;
    ``draw_noise`` samples one realization.
    """
    name: str
    t2_star: float = 41e-6          # s  (SiMOS v2.1, Tanttu 2024)
    t1: float = 9.5                 # s  (SiMOS v2.1)
    sigma_omega: float = 0.01       # relative Rabi-amplitude drift (per run)
    sigma_delta_mhz: float = 0.0    # per-qubit detuning drift, MHz (per run)
    j_res_khz: float = 1.0          # residual always-on exchange, kHz (mean)
    j_res_sigma_frac: float = 1.0   # rel. std of the J_res draw (mean -> std)
    n_realizations: int = 20        # ensemble size for noise_robust_F


DEFAULT_PROFILE = NoiseProfile(name="nominal")


@dataclass
class NoiseDraw:
    """One realization of the noise ensemble."""
    omega_scale: float = 1.0
    delta0: float = 0.0             # angular frequency, rad/s
    delta1: float = 0.0
    j_res: float = 0.0              # Hz (converted to rad/s in H_segment)


def draw_noise(profile: NoiseProfile, rng: np.random.Generator) -> NoiseDraw:
    s = rng.normal(1.0, profile.sigma_omega)
    d0 = 2.0 * np.pi * rng.normal(0.0, profile.sigma_delta_mhz * 1e6)
    d1 = 2.0 * np.pi * rng.normal(0.0, profile.sigma_delta_mhz * 1e6)
    jr = profile.j_res_khz * 1e3 * (1.0 + profile.j_res_sigma_frac * rng.standard_normal())
    jr = abs(jr)  # magnitude of residual exchange; sign is absorbed (ZZ phase)
    return NoiseDraw(omega_scale=float(s), delta0=float(d0), delta1=float(d1),
                     j_res=float(jr))


def H_segment(omega1: float, phi1: float, omega2: float, phi2: float,
              j_hz: float, draw: NoiseDraw | None = None) -> np.ndarray:
    """Per-segment rotating-frame Hamiltonian (angular-frequency units).

    All frequency arguments are Hz (converted to rad/s here).
    ``j_hz`` is the nominal exchange setpoint; residual exchange and
    detuning drift come from the noise draw (zero if ``draw`` is None).
    """
    w1 = 2.0 * np.pi * omega1 * (draw.omega_scale if draw is not None else 1.0)
    w2 = 2.0 * np.pi * omega2 * (draw.omega_scale if draw is not None else 1.0)
    jj = 2.0 * np.pi * (j_hz + (draw.j_res if draw is not None else 0.0))
    h = (0.0)
    if draw is not None:
        h = h + 0.5 * draw.delta0 * _Z0 + 0.5 * draw.delta1 * _Z1
    h = h + jj * _EXCHANGE
    h = h + 0.5 * w1 * (np.cos(phi1) * _X0 - np.sin(phi1) * _Y0)
    h = h + 0.5 * w2 * (np.cos(phi2) * _X1 - np.sin(phi2) * _Y1)
    return h


def _phase_damp(rho: np.ndarray, eta: float) -> np.ndarray:
    """Two-qubit phase damping with single-qubit coherence factor eta."""
    f = np.exp(0.5 * (_POP[:, None] + _POP[None, :]) * np.log(eta))
    return rho * f


def _amplitude_damp(rho: np.ndarray, p: float) -> np.ndarray:
    """Two-qubit amplitude damping (independent qubits), per-qubit prob p."""
    q0 = np.sqrt(p)
    q1 = np.sqrt(1.0 - p)
    # single-qubit Kraus: K00=diag(1, q1), K10=q0|0><1|  on each site
    def k_site(k_a: int, k_b: int) -> np.ndarray:
        # k=0: K0, k=1: K1 on the site
        def site(k: int) -> np.ndarray:
            if k == 0:
                return np.array([[1.0, 0.0], [0.0, q1]], dtype=complex)
            return np.array([[q0, 0.0], [0.0, 0.0]], dtype=complex)
        return np.kron(site(k_a), site(k_b))
    out = np.zeros_like(rho)
    for a in (0, 1):
        for b in (0, 1):
            K = k_site(a, b)
            out += K @ rho @ K.conj().T
    return out


def evolve_segmented(omega1, phi1, omega2, phi2, j, dt: float,
                     target: np.ndarray | None = None,
                     draw: NoiseDraw | None = None,
                     t2_star: float | None = None,
                     t1: float | None = None,
                     return_unitary: bool = False):
    """Evolve |++> through N piecewise-constant segments under noise.

    Returns (fidelity, final_rho, U_gate) — ``U_gate`` is the *noiseless*
    unitary of the same segments when ``return_unitary`` is True.
    """
    n = len(j)
    rho = np.outer(input_state_plus_plus(), input_state_plus_plus().conj())
    eta = np.exp(-dt / t2_star) if t2_star is not None else 1.0
    p_relax = 1.0 - np.exp(-dt / t1) if t1 is not None else 0.0
    do_noise = draw is not None or t2_star is not None
    U_acc = None if return_unitary else None
    f_target = target if target is not None else target_bell_plus()
    proj = np.outer(f_target, f_target.conj())

    for k in range(n):
        H = H_segment(omega1[k], phi1[k], omega2[k], phi2[k], j[k], draw)
        U = expm(-1j * H * dt)
        rho = U @ rho @ U.conj().T
        if do_noise:
            if t2_star is not None:
                rho = _phase_damp(rho, eta)
            if t1 is not None:
                rho = _amplitude_damp(rho, p_relax)
        if return_unitary:
            U_acc = U @ U_acc if U_acc is not None else U
    fid = float(np.real(np.trace(proj @ rho)))
    fid = min(1.0, max(0.0, fid))
    if return_unitary:
        return fid, rho, U_acc
    return fid, rho


def state_fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.clip(np.real(np.trace(np.outer(target, target.conj()) @ rho)), 0.0, 1.0))


def noise_robust_fidelity(omega1, phi1, omega2, phi2, j, dt: float,
                          profile: NoiseProfile,
                          seed: int = 0) -> tuple[float, list[float]]:
    """Mean state fidelity over an ensemble of noise realizations.

    This is the shared ``noise_robust_F`` metric for the whole E5 sweep:
    the same function scores GRAPE pulses, CRAB pulses and SAC policies.
    """
    rng = np.random.default_rng(seed)
    fids = []
    for _ in range(profile.n_realizations):
        draw = draw_noise(profile, rng)
        fid, _ = evolve_segmented(omega1, phi1, omega2, phi2, j, dt,
                                  draw=draw,
                                  t2_star=profile.t2_star, t1=profile.t1)
        fids.append(fid)
    return float(np.mean(fids)), fids


def pulse_from_vector(vec: np.ndarray, n: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
    """Unpack the flat 5*n control vector into segment arrays.

    Layout per segment: [omega1, phi1, omega2, phi2, j] (Hz / rad / Hz).
    Bounds: omega_i in [0, OMEGA_MAX], phi_i in [-pi, pi], j in [0, J_MAX].
    """
    OMEGA_MAX, J_MAX = OMEGA_MAX_HZ, J_MAX_HZ
    v = np.clip(vec, 0.0, 1.0).reshape(n, 5)
    omega1 = v[:, 0] * OMEGA_MAX
    phi1 = (v[:, 1] * 2.0 - 1.0) * np.pi
    omega2 = v[:, 2] * OMEGA_MAX
    phi2 = (v[:, 3] * 2.0 - 1.0) * np.pi
    j = v[:, 4] * J_MAX
    return omega1, phi1, omega2, phi2, j


def vector_from_pulse(omega1, phi1, omega2, phi2, j) -> np.ndarray:
    OMEGA_MAX, J_MAX = OMEGA_MAX_HZ, J_MAX_HZ
    n = len(j)
    v = np.empty((n, 5))
    v[:, 0] = omega1 / OMEGA_MAX
    v[:, 1] = phi1 / np.pi + 0.5
    v[:, 2] = omega2 / OMEGA_MAX
    v[:, 3] = phi2 / np.pi + 0.5
    v[:, 4] = j / J_MAX
    return np.clip(v, 0.0, 1.0).ravel()
