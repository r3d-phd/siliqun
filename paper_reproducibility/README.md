# SiliQun Paper Reproducibility Package

This directory is the paper-specific reproduction entry point for the SiliQun simulator manuscript. It replaces the repository-root guide that documented an unrelated control project.

## Status and scope

This is the **remediation baseline**. It verifies the repaired density-matrix solver and its analytic small-system properties. It does not yet claim to reproduce historical manuscript figures or any result that has not been regenerated from the exact source revision and configuration listed in `evidence_ledger.csv`.

## Exact source revision

| Field | Value |
|---|---|
| Baseline public revision | `14a49bf957b7aeacffc1bd15ba91293d6fd31b83` |
| Remediation branch | `fix/siliqun-paper-remediation-20260825` |
| Python | 3.12 |
| Tested numerical packages | NumPy, SciPy, Gymnasium, pytest |

## Reproduction steps

Create the environment from `environment.yml`, then run:

```bash
python -m pip install -e ".[dev]"
bash paper_reproducibility/run_smoke_validation.sh
```

The command executes the Lindblad regression suite. It checks trace preservation, Hermiticity, physical complex coherence, the T2* rate convention, and agreement between RK4 and the matrix-exponential propagator for a static one-qubit reference problem.

The optional interoperability and generalised-platform smoke scripts are intentionally executed outside the default pytest collection because they rely on optional stacks and historically call `sys.exit()` at module-import time:

```bash
python tests/test_standards.py
python tests/test_generalised.py
python tests/test_integration.py
python tests/test_mpo_simulator.py
```

Its optional dependencies and pass/fail status must be recorded in a future release manifest.

## Evidence ledger

Every manuscript claim retained in a future paper revision must have a `verified` row in `evidence_ledger.csv`. A verified row requires the exact command, configuration, source revision, raw output, and validation test. Claims marked `pending_regeneration` are deliberately excluded from paper-level conclusions until that evidence exists.

## Release requirements before submission

Before submission, create an immutable GitHub release and archive it to a DOI-bearing repository. The release must contain this directory, the manuscript source, the exact generated figures/tables, raw outputs, package lockfile, test report, and a completed evidence ledger.
