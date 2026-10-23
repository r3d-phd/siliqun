"""Regenerate traceable small-system validation evidence for the SiliQun paper.

The workflow uses the released Lindblad solver after remediation and compares
its outcomes with analytic amplitude-damping and pure-dephasing predictions.
It also compares RK4 and matrix-exponential propagation on a static drive.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from siliqun.physics.devices.profiles import donor_device
from siliqun.pulse.lindblad import DrivePulse, LindbladSimulator, PulseSequence


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
plt.rcParams["axes.unicode_minus"] = False


def idle_sequence(duration: float) -> PulseSequence:
    sequence = PulseSequence()
    sequence.add(
        DrivePulse(
            qubit=0,
            amplitude=0.0,
            frequency=0.0,
            phase=0.0,
            duration=duration,
        )
    )
    return sequence


def device(t1: float, t2_star: float):
    profile = donor_device(n_qubits=1)
    profile.noise_params.t1_times = [t1]
    profile.noise_params.t2_star_times = [t2_star]
    return profile


def final_density(profile, sequence, initial_state, method="rk4"):
    sim = LindbladSimulator(
        device=profile,
        n_qubits=1,
        method=method,
        dt=1e-11,
        record_every=1,
        use_gpu=False,
    )
    sim.reset(initial_state)
    return sim.evolve(sequence, reset_before=False).rho_final


def main():
    # Amplitude damping: P(|1>) = exp(-t/T1).
    t1 = 4e-9
    t = 2e-9
    rho_t1 = final_density(device(t1, 1e12), idle_sequence(t), np.array([0.0, 1.0]))
    observed_p1 = float(np.real(rho_t1[1, 1]))
    analytic_p1 = float(np.exp(-t / t1))

    # Pure dephasing: |rho_01(t)| = 0.5 exp(-t/T2*).
    t2_star = 4e-9
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_t2 = final_density(device(1e12, t2_star), idle_sequence(t), plus)
    observed_coherence = float(abs(rho_t2[0, 1]))
    analytic_coherence = float(0.5 * np.exp(-t / t2_star))

    # Static drive: RK4 and Liouvillian exponential must agree.
    drive = PulseSequence()
    drive.add(
        DrivePulse(
            qubit=0,
            amplitude=100e6,
            frequency=0.0,
            phase=0.0,
            duration=1e-9,
        )
    )
    near_unitary = device(1e12, 1e12)
    rho_rk4 = final_density(near_unitary, drive, np.array([1.0, 0.0]), method="rk4")
    rho_expm = final_density(near_unitary, drive, np.array([1.0, 0.0]), method="expm")
    rk4_expm_error = float(np.linalg.norm(rho_rk4 - rho_expm, ord="fro"))

    checks = {
        "amplitude_damping": {
            "t1_s": t1,
            "duration_s": t,
            "observed_excited_population": observed_p1,
            "analytic_excited_population": analytic_p1,
            "absolute_error": abs(observed_p1 - analytic_p1),
        },
        "pure_dephasing": {
            "t2_star_s": t2_star,
            "duration_s": t,
            "observed_coherence_magnitude": observed_coherence,
            "analytic_coherence_magnitude": analytic_coherence,
            "absolute_error": abs(observed_coherence - analytic_coherence),
        },
        "integrator_agreement": {
            "rk4_expm_frobenius_error": rk4_expm_error,
            "rk4_trace_error": float(abs(np.trace(rho_rk4) - 1.0)),
            "rk4_hermiticity_error": float(np.linalg.norm(rho_rk4 - rho_rk4.conj().T, ord="fro")),
        },
    }

    with (OUTPUTS / "solver_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)

    labels = ["T1 population", "T2* coherence", "RK4/expm error"]
    values = [
        checks["amplitude_damping"]["absolute_error"],
        checks["pure_dephasing"]["absolute_error"],
        checks["integrator_agreement"]["rk4_expm_frobenius_error"],
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bars = ax.bar(labels, values, color=["#2e6f95", "#4c956c", "#8c6bb1"])
    ax.set_yscale("log")
    ax.set_ylabel("Absolute numerical error")
    ax.set_title("SiliQun repaired Lindblad solver: analytic validation")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2e}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "solver_validation.png", dpi=220)

    tolerance = 1e-8
    failures = [key for key, value in zip(labels, values) if value > tolerance]
    if failures:
        raise RuntimeError(f"Validation errors exceed {tolerance}: {failures}")


if __name__ == "__main__":
    main()
