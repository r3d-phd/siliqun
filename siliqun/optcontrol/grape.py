"""GRAPE (gradient ascent pulse engineering) for the SiliQun 2-qubit task.

Objective: state fidelity F = |<t|U(p)|i>|^2 of the noiseless gate
propagator U(p) = prod_k expm(-i H_k dt), with |i> = |++> and
|t> = |Phi+>.  Analytic gradients via the adjoint method with exact
(matrix-integral) derivatives of the segment propagators.

Optimisation: bounded L-BFGS-B (scipy) over the normalized control
vector v in [0,1]^{5N} (see ``core.pulse_from_vector``), matching the
protocol reported in the paper's E4 baseline comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .core import (
    H_segment,
    J_MAX_HZ,
    OMEGA_MAX_HZ,
    input_state_plus_plus,
    pulse_from_vector,
    target_bell_plus,
)
from .core import _EXCHANGE, _X0, _Y0, _X1, _Y1

# precomputed operators (angular-frequency units)
_PI = 2.0 * np.pi

# chain rule from normalized v-coordinates to physical parameters
_JAC5 = np.array([OMEGA_MAX_HZ, 2.0 * np.pi, OMEGA_MAX_HZ, 2.0 * np.pi, J_MAX_HZ])

# 16-point Gauss-Legendre on [0, 1]
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X = 0.5 * _GL_X + 0.5
_GL_W = 0.5 * _GL_W


def _segment_derivatives(h: np.ndarray, w1: float, p1: float,
                         w2: float, p2: float, dt: float):
    """Return [dU/domega1, dU/dphi1, dU/domega2, dU/dphi2, dU/dj] for one
    segment (derivatives w.r.t. physical parameters: Hz, rad, Hz, rad, Hz)."""
    U = expm(-1j * h * dt)

    # precompute the 16 rotated-frame matrices once
    A = [expm(1j * s * h * dt) for s in _GL_X]

    def matint(hp):
        # int_0^1 expm(i s H dt) hp expm(-i s H dt) ds
        acc = np.zeros_like(h)
        for a, w in zip(A, _GL_W):
            acc += w * (a @ hp @ a.conj().T)
        return acc

    # w1, w2 arrive in Hz (as H_segment expects); H depends on them
    # through 2*pi*w, so the derivatives carry that factor.
    hp_o1 = 0.5 * _PI * (np.cos(p1) * _X0 - np.sin(p1) * _Y0)
    hp_p1 = -0.5 * (_PI * w1) * (np.sin(p1) * _X0 + np.cos(p1) * _Y0)
    hp_o2 = 0.5 * _PI * (np.cos(p2) * _X1 - np.sin(p2) * _Y1)
    hp_p2 = -0.5 * (_PI * w2) * (np.sin(p2) * _X1 + np.cos(p2) * _Y1)
    hp_j = _PI * _EXCHANGE
    out = []
    for hp in (hp_o1, hp_p1, hp_o2, hp_p2, hp_j):
        out.append((-1j * dt) * (U @ matint(hp)))
    return out, U


def _fidelity_of(omega1, phi1, omega2, phi2, j, dt, target):
    n = len(j)
    psi = input_state_plus_plus()
    for k in range(n):
        H = H_segment(omega1[k], phi1[k], omega2[k], phi2[k], j[k])
        psi = expm(-1j * H * dt) @ psi
    return float(np.clip(np.abs(np.vdot(target, psi)) ** 2, 0.0, 1.0))


def _physical_grads(omega1, phi1, omega2, phi2, j, dt):
    """dF/d(physical control at each segment) via the adjoint method.

    Returns a flat array of length 5*n in segment-major order:
    [dF/domega1_k, dF/dphi1_k, dF/domega2_k, dF/dphi2_k, dF/dj_k].
    """
    n = len(j)
    target = target_bell_plus()
    states = [input_state_plus_plus()]
    for k in range(n):
        H = H_segment(omega1[k], phi1[k], omega2[k], phi2[k], j[k])
        states.append(expm(-1j * H * dt) @ states[-1])
    c = np.vdot(target, states[-1])
    r_ket = target.copy()
    grads = np.zeros(5 * n)
    for k in range(n - 1, -1, -1):
        H = H_segment(omega1[k], phi1[k], omega2[k], phi2[k], j[k])
        U = expm(-1j * H * dt)
        dU_list, _ = _segment_derivatives(H, omega1[k], phi1[k], omega2[k], phi2[k], dt)
        for p, dU in enumerate(dU_list):
            dc = np.vdot(r_ket, dU @ states[k])
            grads[k * 5 + p] = 2.0 * np.real(dc * np.conj(c))
        r_ket = U.conj().T @ r_ket
    return grads


def analytic_grad_v(v: np.ndarray, n: int, dt: float) -> np.ndarray:
    """dF/dv for the normalized control vector (chain rule included)."""
    omega1, phi1, omega2, phi2, j = pulse_from_vector(v, n)
    return _physical_grads(omega1, phi1, omega2, phi2, j, dt) * np.tile(_JAC5, n)


def run_grape(n_segments: int, dt: float, init_vec: np.ndarray,
              n_iter: int = 300, tol: float = 1e-9, seed: int | None = None,
              verbose: bool = False) -> dict:
    """Run bounded GRAPE (L-BFGS-B in normalized v-space) on the 2-qubit task.

    Minimizes 1 - F with analytic gradients; parameters are held in
    [0, 1] by the L-BFGS-B bounds.  Returns a dict with the optimized
    vector, the final noiseless fidelity, the per-evaluation fidelity
    history, the iteration count, and scipy's success flag.
    """
    target = target_bell_plus()
    v0 = np.clip(init_vec.copy(), 0.0, 1.0)
    hist = []

    def negF(v):
        omega1, phi1, omega2, phi2, j = pulse_from_vector(v, n_segments)
        F = _fidelity_of(omega1, phi1, omega2, phi2, j, dt, target)
        hist.append(F)
        return 1.0 - F

    def neggrad(v):
        return -analytic_grad_v(v, n_segments, dt)

    res = minimize(
        negF, v0, jac=neggrad, method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * v0.size,
        options={"maxiter": n_iter, "maxfun": 5 * n_iter, "gtol": tol},
    )
    v = np.clip(res.x, 0.0, 1.0)
    f_final = _fidelity_of(*pulse_from_vector(v, n_segments), dt, target)
    return {"vec": v, "fidelity": f_final, "history": hist,
            "iters": int(res.nit), "success": bool(res.success)}
