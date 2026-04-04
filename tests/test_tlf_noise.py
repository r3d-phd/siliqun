"""Test the TLF correlation model and updated noise generator."""

import sys
sys.path.insert(0, '/home/ubuntu/siliqun')

import numpy as np
from siliqun.physics.noise.channels import (
    NoiseParams,
    default_noise_params,
    TLFCorrelationModel,
    ChargeNoiseGenerator,
)

def test_tlf_correlation_model():
    """Test TLFCorrelationModel with RIKEN 5Q parameters."""
    print("=" * 60)
    print("TEST 1: TLFCorrelationModel (RIKEN 5Q)")
    print("=" * 60)

    model = TLFCorrelationModel(
        n_qubits=5,
        qubit_spacing=108.0,  # RIKEN device
        tlf_correlation_length=81.0,
        tlf_density=3e10,
        magnetic_drift_rate=8.0,
    )

    print(f"  N_c (correlation in spacings): {model.n_c:.4f}")
    print(f"  NN correlation: {model.nn_correlation:.4f}")
    print(f"  Expected NN: exp(-108/81) = {np.exp(-108/81):.4f}")
    assert abs(model.nn_correlation - np.exp(-108/81)) < 1e-10, "NN correlation mismatch!"

    print(f"\n  Correlation matrix:")
    C = model.correlation_matrix
    for i in range(5):
        row = "    " + "  ".join(f"{C[i,j]:.4f}" for j in range(5))
        print(row)

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(C)
    print(f"\n  Min eigenvalue: {eigenvalues.min():.6f} (must be > 0)")
    assert eigenvalues.min() > 0, "Correlation matrix not positive definite!"

    print("  PASSED\n")


def test_tlf_sledge():
    """Test TLFCorrelationModel with SLEDGE parameters."""
    print("=" * 60)
    print("TEST 2: TLFCorrelationModel (SLEDGE)")
    print("=" * 60)

    model = TLFCorrelationModel(
        n_qubits=9,
        qubit_spacing=80.0,  # SLEDGE device
        tlf_correlation_length=81.0,
    )

    print(f"  N_c: {model.n_c:.4f}")
    print(f"  NN correlation: {model.nn_correlation:.4f}")
    print(f"  Expected NN: exp(-80/81) = {np.exp(-80/81):.4f}")
    assert abs(model.nn_correlation - np.exp(-80/81)) < 1e-10

    # Compare SLEDGE vs RIKEN
    riken_nn = np.exp(-108/81)
    sledge_nn = model.nn_correlation
    ratio = sledge_nn / riken_nn
    print(f"\n  SLEDGE NN / RIKEN NN = {ratio:.4f} (SLEDGE {(ratio-1)*100:.1f}% stronger)")
    print("  PASSED\n")


def test_tlf_2d_grid():
    """Test TLFCorrelationModel with 2D grid positions."""
    print("=" * 60)
    print("TEST 3: TLFCorrelationModel (3x3 2D grid)")
    print("=" * 60)

    spacing = 80.0
    positions = []
    for r in range(3):
        for c in range(3):
            positions.append((c * spacing, r * spacing))

    model = TLFCorrelationModel(
        n_qubits=9,
        qubit_positions=positions,
        tlf_correlation_length=81.0,
    )

    print(f"  NN correlation (adjacent): {model.nn_correlation:.4f}")
    # Diagonal distance = 80*sqrt(2) = 113.14 nm
    diag_corr = np.exp(-80 * np.sqrt(2) / 81)
    actual_diag = model.correlation_matrix[0, 4]  # (0,0) to (1,1)
    print(f"  Diagonal correlation: {actual_diag:.4f}")
    print(f"  Expected diagonal: exp(-{80*np.sqrt(2):.1f}/81) = {diag_corr:.4f}")
    assert abs(actual_diag - diag_corr) < 1e-10

    print("  PASSED\n")


def test_charge_noise_with_tlf():
    """Test ChargeNoiseGenerator with TLF model."""
    print("=" * 60)
    print("TEST 4: ChargeNoiseGenerator with TLF model")
    print("=" * 60)

    tlf = TLFCorrelationModel(
        n_qubits=5,
        qubit_spacing=108.0,
        tlf_correlation_length=81.0,
    )

    gen = ChargeNoiseGenerator(
        n_qubits=5,
        amplitude=1e-6,
        dt=1e-9,
        seed=42,
        tlf_model=tlf,
    )

    # Generate many samples and check correlations
    n_samples = 10000
    samples = np.zeros((5, n_samples))
    for i in range(n_samples):
        samples[:, i] = gen.sample()

    # Compute empirical correlation
    empirical_corr = np.corrcoef(samples)
    expected_nn = np.exp(-108/81)

    print(f"  Empirical NN correlation: {empirical_corr[0, 1]:.4f}")
    print(f"  Expected NN correlation:  {expected_nn:.4f}")
    print(f"  Difference: {abs(empirical_corr[0, 1] - expected_nn):.4f}")

    # Allow some statistical tolerance
    assert abs(empirical_corr[0, 1] - expected_nn) < 0.1, \
        f"Empirical correlation too far from expected!"

    print("  PASSED\n")


def test_noise_params_presets():
    """Test that noise parameter presets include TLF fields."""
    print("=" * 60)
    print("TEST 5: NoiseParams presets")
    print("=" * 60)

    sledge = default_noise_params(5, "sledge")
    print(f"  SLEDGE:")
    print(f"    correlation_model: {sledge.correlation_model}")
    print(f"    tlf_density: {sledge.tlf_density:.0e}")
    print(f"    tlf_correlation_length: {sledge.tlf_correlation_length} nm")
    print(f"    qubit_spacing: {sledge.qubit_spacing} nm")
    print(f"    magnetic_drift_rate: {sledge.magnetic_drift_rate} Hz/s")
    assert sledge.correlation_model == "tlf"
    assert sledge.tlf_density == 3e10
    assert sledge.tlf_correlation_length == 81.0
    assert sledge.qubit_spacing == 80.0

    riken = default_noise_params(5, "riken_5q")
    print(f"\n  RIKEN 5Q:")
    print(f"    correlation_model: {riken.correlation_model}")
    print(f"    tlf_density: {riken.tlf_density:.0e}")
    print(f"    tlf_correlation_length: {riken.tlf_correlation_length} nm")
    print(f"    qubit_spacing: {riken.qubit_spacing} nm")
    print(f"    magnetic_drift_rate: {riken.magnetic_drift_rate} Hz/s")
    assert riken.correlation_model == "tlf"
    assert riken.qubit_spacing == 108.0

    print("  PASSED\n")


def test_from_noise_params():
    """Test TLFCorrelationModel.from_noise_params factory."""
    print("=" * 60)
    print("TEST 6: TLFCorrelationModel.from_noise_params")
    print("=" * 60)

    params = default_noise_params(5, "sledge")
    model = TLFCorrelationModel.from_noise_params(params, n_qubits=5)

    print(f"  l_c: {model.l_c} nm")
    print(f"  NN correlation: {model.nn_correlation:.4f}")
    print(f"  N_c: {model.n_c:.4f}")
    assert model.l_c == 81.0
    assert abs(model.nn_correlation - np.exp(-80/81)) < 1e-10

    print("  PASSED\n")


def test_global_drift():
    """Test global magnetic drift phase computation."""
    print("=" * 60)
    print("TEST 7: Global magnetic drift phase")
    print("=" * 60)

    model = TLFCorrelationModel(
        n_qubits=5,
        qubit_spacing=108.0,
        magnetic_drift_rate=8.0,
    )

    # After 1 second, drift = 8 Hz -> phase = 2*pi*8 rad
    phase_1s = model.global_drift_phase(1.0)
    expected = 2 * np.pi * 8.0
    print(f"  Phase after 1s: {phase_1s:.4f} rad")
    print(f"  Expected: {expected:.4f} rad")
    assert abs(phase_1s - expected) < 1e-10

    # After 10 ns (typical gate), drift is negligible
    phase_10ns = model.global_drift_phase(10e-9)
    print(f"  Phase after 10 ns: {phase_10ns:.2e} rad (negligible)")
    assert phase_10ns < 1e-6

    print("  PASSED\n")


if __name__ == "__main__":
    test_tlf_correlation_model()
    test_tlf_sledge()
    test_tlf_2d_grid()
    test_charge_noise_with_tlf()
    test_noise_params_presets()
    test_from_noise_params()
    test_global_drift()
    print("=" * 60)
    print("ALL 7 TESTS PASSED")
    print("=" * 60)
