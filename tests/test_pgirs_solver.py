"""Regression test: PGIRSValidator must run the Lindblad solver.

This guards against the historical bug where all three PGIRS Phase-1
benchmarks compared an analytic expression to *itself* (0 solver calls),
which made the paper's "machine-precision validation, max delta <= 2.78e-16"
a self-comparison artifact rather than a measurement.

A correct validator:
  * actually integrates the Lindblad master equation (>= 1 solver call), and
  * agrees with the analytic targets to within the declared tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from siliqun.core.abc import PGIRSValidator
from siliqun.physics.devices.technology_profiles import SiMOSNominalProfile
from siliqun.pulse import lindblad as lindblad_mod


def test_pgirs_runs_the_solver():
    """PGIRS.run() must exercise the Lindblad solver, not just analytic math."""
    calls = {"n": 0}
    orig = lindblad_mod.LindbladSimulator.evolve

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    lindblad_mod.LindbladSimulator.evolve = spy
    try:
        validator = PGIRSValidator(SiMOSNominalProfile())
        results = validator.run(tolerance=0.02)
    finally:
        lindblad_mod.LindbladSimulator.evolve = orig

    # The tautology made 0 solver calls. A real validation needs several
    # (Rabi x2 + Bell x2 + dephasing x1 = 5 for the SiMOS profile).
    assert calls["n"] >= 1, "PGIRSValidator.run() did not run the Lindblad solver"
    assert calls["n"] >= 3, f"only {calls['n']} solver call(s) - not a real validation"
    assert results == {"rabi": True, "bell": True, "t2_echo": True}, results


def test_pgiris_rabi_pi_pulse_transfer():
    """A resonant pi-pulse must transfer pop|0>->|1> (coherence limit)."""
    validator = PGIRSValidator(SiMOSNominalProfile())
    validator._benchmark_rabi(tolerance=0.02)
    d = validator.last_deltas["rabi"]
    assert d["delta_pi"] <= 0.02
    assert d["pop1_pi"] > 0.99          # full transfer, not the buggy ~0


def test_pgirs_bell_concurrence():
    """sqrt-SWAP from |01> must produce a maximally entangled state (C=1)."""
    validator = PGIRSValidator(SiMOSNominalProfile())
    validator._benchmark_bell(tolerance=0.02)
    d = validator.last_deltas["bell"]
    assert abs(d["concurrence"] - 1.0) <= 0.02
    assert abs(d["F_full_swap_10"] - 1.0) <= 0.02


def test_pgirs_t2star_coherence_decay():
    """Free dephasing must match |rho01|(T2*) = 0.5 e^-1."""
    validator = PGIRSValidator(SiMOSNominalProfile())
    validator._benchmark_t2_echo(tolerance=0.02)
    d = validator.last_deltas["t2_echo"]
    assert d["relative_error"] <= 0.02
    # analytic reference
    assert abs(d["analytic"] - 0.5 * np.exp(-1.0)) < 1e-12


def test_fidelity_estimator_bounded_01():
    """The state-fidelity estimator must be clamped to [0, 1].

    Guards against the unclipped Tr(rho_target @ rho) that produced the
    previously reported unphysical F = 1.0009.
    """
    from siliqun.physics.devices.profiles import get_device_profile
    from siliqun.physics.noise.channels import NoiseParams
    from siliqun.pulse.lindblad import LindbladSimulator, PulseSequence, DrivePulse

    dev = get_device_profile("simos", n_qubits=1)
    dev.noise_params = NoiseParams(
        t1_times=[float("inf")], t2_star_times=[float("inf")],
        measurement_fidelity=1.0, dephasing_model="exponential",
    )
    sim = LindbladSimulator(dev, n_qubits=1, method="rk4", dt=1e-11, use_gpu=False)
    A = 1.0 / (2 * 10e-9)
    sim.reset()
    res = sim.evolve(
        PulseSequence().add(DrivePulse(0, A, 5e9, 0.0, 10e-9)),
        target_state=np.array([0.0, 1.0], complex),
    )
    # Every recorded fidelity must be physically bounded.
    assert all(0.0 <= f <= 1.0 for f in res.fidelity_history), (
        f"unphysical fidelity: min={min(res.fidelity_history):.6f} "
        f"max={max(res.fidelity_history):.6f}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
