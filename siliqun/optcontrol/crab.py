"""CRAB (Cosine-basis Representation for All-time control) for the SiliQun
2-qubit task.

The paper's E4 baseline uses the Fourier-basis CRAB protocol (Caneva et
al., New J. Phys. 14, 033011 (2011)) with n_coeff=6 harmonics.  Each of
the five physical controls p(t) in {omega1, phi1, omega2, phi2, j} is
represented over the gate window T = n*dt as a truncated Fourier series

    p(t) = a_0 + sum_{m=1}^{n_coeff} [ a_m cos(2*pi*m*t/T)
                                        + b_m sin(2*pi*m*t/T) ]

with 1 + 2*n_coeff = 13 coefficients per control (65 total).  The
series is sampled at the segment midpoints t_k = (k+0.5)*dt to build the
piecewise-constant schedule consumed by the shared SiliQun evolver, and
the Fourier coefficients are optimised with bounded L-BFGS-B using the
same analytic adjoint gradients as GRAPE (chain rule through the Fourier
basis).

Reference: Caneva, Fazio, Benedetti, Lloyd, Phys. Rev. Lett. 111, 020503
(2013); original cosine basis: Khodjasteh, Wang, Mohseni, PRL 102, 140501
(2009).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .core import (
    J_MAX_HZ,
    OMEGA_MAX_HZ,
    target_bell_plus,
)
from .grape import _fidelity_of, _physical_grads

_PI = 2.0 * np.pi


def _fourier_basis(n: int, n_coeff: int, dt: float) -> np.ndarray:
    """Return the (n, 1+2*n_coeff) Fourier basis sampled at segment midpoints.

    Column 0 is the DC term (all ones); columns 1..2*n_coeff are the
    cos/sin harmonics evaluated at t_k = (k+0.5)*dt over the window T=n*dt.
    """
    T = n * dt
    t = (np.arange(n) + 0.5) * dt
    basis = np.ones((n, 1 + 2 * n_coeff))
    for m in range(1, n_coeff + 1):
        basis[:, 2 * m - 1] = np.cos(2.0 * np.pi * m * t / T)
        basis[:, 2 * m] = np.sin(2.0 * np.pi * m * t / T)
    return basis


def _coeffs_to_pulse(coefs: np.ndarray, n: int, n_coeff: int, dt: float):
    """Map the flat (5*(1+2*n_coeff),) coefficient vector to segment arrays.

    Layout: for control p in 0..4, its (1+2*n_coeff) coefficients are
    stored contiguously.  The DC term is free; the harmonics are bounded
    so that the sampled signal stays within the physical range.
    """
    basis = _fourier_basis(n, n_coeff, dt)
    nc = 1 + 2 * n_coeff
    omega1 = np.clip(basis @ coefs[0 * nc:(0 + 1) * nc], 0.0, OMEGA_MAX_HZ)
    phi1 = np.clip(basis @ coefs[1 * nc:(1 + 1) * nc], -_PI, _PI)
    omega2 = np.clip(basis @ coefs[2 * nc:(2 + 1) * nc], 0.0, OMEGA_MAX_HZ)
    phi2 = np.clip(basis @ coefs[3 * nc:(3 + 1) * nc], -_PI, _PI)
    j = np.clip(basis @ coefs[4 * nc:(4 + 1) * nc], 0.0, J_MAX_HZ)
    return omega1, phi1, omega2, phi2, j


def run_crab(n_segments: int, dt: float, n_coeff: int = 6,
             n_iter: int = 400, tol: float = 1e-9, seed: int = 0,
             verbose: bool = False) -> dict:
    """Run Fourier-basis CRAB (L-BFGS-B) on the 2-qubit task.

    Initial coefficients: DC at mid-range (phase channels at zero) with
    small random harmonics (10% of the amplitude bound) drawn from
    ``seed`` -- the pure DC point is a symmetric stationary point at
    F = 0.5, so the jitter is required to escape it.  Returns a dict
    with the final noiseless fidelity, the per-evaluation history,
    iteration count, success flag, the optimised coefficients, and the
    reconstructed piecewise-constant control vector (``vec``, in
    physical units) for downstream use.
    """
    target = target_bell_plus()
    nc = 1 + 2 * n_coeff
    rng = np.random.default_rng(seed)

    _LOS = np.array([0.0, -_PI, 0.0, -_PI, 0.0])
    _HIS = np.array([OMEGA_MAX_HZ, _PI, OMEGA_MAX_HZ, _PI, J_MAX_HZ])
    _WIDTH = _HIS - _LOS
    # DC init at mid-range + small harmonic jitter.  The pure DC point is
    # a symmetric stationary point (F = 0.5), so the jitter (10% of the
    # per-harmonic amplitude bound) is what breaks the symmetry and lets
    # the optimiser escape; the DC offset keeps the start in a strong
    # basin (reaches F ~ 1.0 in ~20 iterations).
    dc = 0.5 * _HIS
    dc[1] = 0.0  # phase channels start at zero phase (not the +-pi clip edge)
    dc[3] = 0.0
    coefs0 = np.zeros((5, nc))
    coefs0[:, 0] = dc
    for p in range(5):
        amp = 0.5 * _WIDTH[p] / (2 * n_coeff)
        coefs0[p, 1:] = rng.uniform(-0.1 * amp, 0.1 * amp, nc - 1)

    bounds = []
    for p in range(5):
        lo = np.full(nc, -1.0)
        hi = np.full(nc, 1.0)
        lo[0] = _LOS[p]
        hi[0] = _HIS[p]
        # harmonics: keep |oscillating part| <= half the physical range
        amp = 0.5 * _WIDTH[p] / (2 * n_coeff)
        lo[1:] = -amp
        hi[1:] = amp
        bounds.append((lo, hi))
    flat0 = coefs0.ravel()
    flat_bounds = [(lo[i], hi[i])
                   for p, (lo, hi) in enumerate(bounds) for i in range(nc)]

    basis = _fourier_basis(n_segments, n_coeff, dt)
    n = n_segments
    hist = []

    def negF(flat):
        c = flat.reshape(5, nc)
        omega1, phi1, omega2, phi2, j = _coeffs_to_pulse(flat, n, n_coeff, dt)
        F = _fidelity_of(omega1, phi1, omega2, phi2, j, dt, target)
        hist.append(F)
        return 1.0 - F

    def neggrad(flat):
        omega1, phi1, omega2, phi2, j = _coeffs_to_pulse(flat, n, n_coeff, dt)
        gp = _physical_grads(omega1, phi1, omega2, phi2, j, dt)  # (5n,) seg-major
        g = gp.reshape(n, 5)
        gcoef = np.empty_like(flat)
        for p in range(5):
            gcoef[p * nc:(p + 1) * nc] = basis.T @ g[:, p]
        return -gcoef

    res = minimize(
        negF, flat0, jac=neggrad, method="L-BFGS-B",
        bounds=flat_bounds,
        options={"maxiter": n_iter, "maxfun": 5 * n_iter, "gtol": tol},
    )
    flat = res.x
    omega1, phi1, omega2, phi2, j = _coeffs_to_pulse(flat, n, n_coeff, dt)
    f_final = _fidelity_of(omega1, phi1, omega2, phi2, j, dt, target)
    vec = np.column_stack([omega1, phi1, omega2, phi2, j]).ravel()
    return {"vec": vec, "fidelity": f_final, "history": hist,
            "iters": int(res.nit), "success": bool(res.success),
            "coefs": flat, "n_coeff": n_coeff}
