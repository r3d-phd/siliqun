# Contributing to SiliQun

Thank you for your interest in contributing to SiliQun. This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository and install in development mode:

```bash
git clone https://github.com/r3d-phd/siliqun.git
cd siliqun
pip install -e ".[dev]"
```

2. Run the test suite to verify your setup:

```bash
python -m pytest tests/ -v
```

## Code Style

SiliQun follows standard Python conventions. Please ensure that all new code includes docstrings for public methods and classes, type hints for function signatures, and unit tests for new functionality.

## Adding a New Hardware Platform (Plugin)

SiliQun v5 uses a formal plugin interface based on the `TechnologyProfile`
abstract base class.  Adding a new qubit platform requires **under 50 lines**
of platform-specific code — the Lindblad solver, Gymnasium interface, and RL
agent are provided by the core and require no modification.

### Minimum Plugin Contract

A plugin consists of exactly three files:

| File | Lines | Purpose |
|------|-------|--------|
| `my_profile.py` | ~28 | `TechnologyProfile` subclass with 5 abstract methods |
| `my_calibration.json` | ~12 | Calibration data with source DOIs |
| `my_benchmarks.py` | ~7 | Three PGIRS Phase-1 benchmark calls |

### The Five Abstract Methods

Every `TechnologyProfile` subclass must implement:

1. `device_parameters()` — dict of scalar physical parameters (T1, T2, tau1, tau2, p2, …)
2. `drift_hamiltonian()` — always-on Hamiltonian as a complex NumPy array
3. `control_hamiltonians()` — list of control Hamiltonians (one per drive line)
4. `noise_channels()` — list of Lindblad collapse operators
5. `gate_library()` — list of native gate names

Plus the `calibration_record` property returning a `CalibrationRecord` with
the source DOI and raw parameter values.

### Reference Implementation

The `plugins/siliqun-gaas/` directory contains the complete reference
implementation for GAA silicon spin qubits.  Copy it as a starting point:

```bash
cp -r plugins/siliqun-gaas plugins/my-platform
# Edit siliqun_gaas/ → my_platform/, update profile values and calibration JSON
pip install -e plugins/my-platform/
```

### Validation

Before a plugin can be registered, it must pass all three PGIRS Phase-1
benchmarks to within ±2 %:

```python
from my_platform.my_benchmarks import assert_pgirs_phase1
assert_pgirs_phase1()   # raises AssertionError if any benchmark fails
```

See the **"Writing a Plugin"** section in `README.md` for a complete
step-by-step example with code snippets.

## Adding a New Gate

Gates are implemented as methods on both `SiliQunSimulator` (MPS engine) and `StateVectorSimulator` (SV engine). When adding a new gate, implement it in both engines to maintain API compatibility, add a unit test in `tests/test_statevector_simulator.py`, and verify unitarity in the test.

## Reporting Issues

Please open a GitHub issue with a clear description of the problem, steps to reproduce, and your environment details (Python version, NumPy version, CuPy version if applicable, GPU model).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
