"""
siliqun_gaas.gaa_benchmarks
============================
Three PGIRS Phase-1 validation benchmarks for the GAA profile.
All three must pass to ±2 % before the plugin can be registered.
"""

from siliqun.core.abc import PGIRSValidator
from .gaa_profile import GAAProfile


def run_pgirs_phase1(tolerance: float = 0.02) -> dict:
    """Run all three PGIRS Phase-1 benchmarks for the GAA profile.

    Returns
    -------
    results : dict
        ``{"rabi": bool, "bell": bool, "t2_echo": bool}``
    """
    profile = GAAProfile()
    validator = PGIRSValidator(profile)
    return validator.run(tolerance=tolerance)


def assert_pgirs_phase1(tolerance: float = 0.02) -> None:
    """Assert that all three PGIRS Phase-1 benchmarks pass."""
    results = run_pgirs_phase1(tolerance=tolerance)
    failures = [k for k, v in results.items() if not v]
    assert not failures, (
        f"GAA plugin failed PGIRS Phase-1 benchmarks: {failures}"
    )
