"""Regression tests for the repaired Lindblad density-matrix solver."""

from __future__ import annotations

import numpy as np

from siliqun.physics.devices.profiles import donor_device
from siliqun.physics.noise.channels import NoiseParams
from siliqun.pulse.lindblad import (
    DrivePulse,
    LindbladSimulator,
    PulseSequence,
    _build_collapse_operators,
)


def _near_unitary_device():
    """One-qubit profile with negligible decoherence for solver comparisons."""
    device = donor_device(n_qubits=1)
    device.noise_params.t1_times = [1.0e12]
    device.noise_params.t2_star_times = [1.0e12]
    return device


def _x_drive_sequence():
    sequence = PulseSequence()
    sequence.add(
        DrivePulse(
            qubit=0,
            amplitude=100e6,
            frequency=0.0,
            phase=0.0,
            duration=1e-9,
        )
    )
    return sequence


def test_rk4_preserves_physical_imaginary_coherence():
    sim = LindbladSimulator(
        device=_near_unitary_device(),
        n_qubits=1,
        method="rk4",
        dt=1e-11,
        use_gpu=False,
    )
    result = sim.evolve(_x_drive_sequence())

    coherence = result.rho_final[0, 1]
    assert abs(np.imag(coherence)) > 1e-4
    assert np.allclose(result.rho_final, result.rho_final.conj().T, atol=1e-10)
    assert np.isclose(np.trace(result.rho_final), 1.0, atol=1e-10)


def test_static_rk4_and_expm_agree():
    sequence = _x_drive_sequence()
    rk4 = LindbladSimulator(
        device=_near_unitary_device(),
        n_qubits=1,
        method="rk4",
        dt=1e-11,
        use_gpu=False,
    ).evolve(sequence).rho_final
    expm = LindbladSimulator(
        device=_near_unitary_device(),
        n_qubits=1,
        method="expm",
        dt=1e-11,
        use_gpu=False,
    ).evolve(sequence).rho_final

    assert np.allclose(rk4, expm, atol=1e-8, rtol=1e-8)


def test_full_sigma_z_rate_matches_t2_star_convention():
    params = NoiseParams(t1_times=[np.inf], t2_star_times=[2.0])
    collapse_ops = _build_collapse_operators(n_qubits=1, noise_params=params)

    # The first operator is relaxation (zero rate); the second is dephasing.
    _, sqrt_gamma_phi = collapse_ops[1]
    gamma_phi = float(sqrt_gamma_phi**2)

    # For gamma D[Z], off-diagonal coherence decays as exp(-2 gamma t).
    assert np.isclose(gamma_phi, 1.0 / (2.0 * params.T2_star), atol=1e-12)
