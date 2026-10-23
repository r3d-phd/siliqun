"""Classical optimal control for SiliQun physics (GRAPE, CRAB) and a
SAC reference implementation.

Implements the classical baselines referenced in the SiliQun paper:
GRAPE (gradient ascent pulse engineering, L-BFGS-B with analytic
adjoint gradients) and CRAB (Fourier-basis, n_coeff=6) operating on the
same 2-qubit SiMOS Hamiltonian and noise model as the RL agent
(rotating-frame Zeeman residual + exchange + EDSR drives), with a
shared noise-robust fidelity evaluation protocol.

References:
  - GRAPE: Khodjasteh & Liu, Phys. Rev. A 87, 042313 (2013)
  - CRAB:  Caneva, Fazio, Benedetti, Lloyd, PRL 111, 020503 (2013)
"""
from .core import (
    NoiseProfile,
    draw_noise,
    state_fidelity,
    evolve_segmented,
    H_segment,
    Pauli,
    noise_robust_fidelity,
    input_state_plus_plus,
    target_bell_plus,
    pulse_from_vector,
    vector_from_pulse,
)
from .grape import run_grape
from .crab import run_crab
from .sac import SACEnv, SACAgent, train_sac

__all__ = [
    "NoiseProfile",
    "draw_noise",
    "state_fidelity",
    "evolve_segmented",
    "H_segment",
    "Pauli",
    "noise_robust_fidelity",
    "input_state_plus_plus",
    "target_bell_plus",
    "pulse_from_vector",
    "vector_from_pulse",
    "run_grape",
    "run_crab",
    "SACEnv",
    "SACAgent",
    "train_sac",
]
