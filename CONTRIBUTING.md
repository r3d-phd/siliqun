# Contributing to SiliQun

Thank you for your interest in contributing to SiliQun. This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository and install in development mode:

```bash
git clone https://github.com/ralshehri/siliqun.git
cd siliqun
pip install -e ".[dev]"
```

2. Run the test suite to verify your setup:

```bash
python -m pytest tests/ -v
```

## Code Style

SiliQun follows standard Python conventions. Please ensure that all new code includes docstrings for public methods and classes, type hints for function signatures, and unit tests for new functionality.

## Adding a New Device Profile

Device profiles are defined in `siliqun/physics/devices/profiles.py`. To add a new profile, create a function that returns a dictionary with the required fields (see existing profiles for the expected structure). Each profile should cite the experimental source for its parameter values.

## Adding a New Gate

Gates are implemented as methods on both `SiliQunSimulator` (MPS engine) and `StateVectorSimulator` (SV engine). When adding a new gate, implement it in both engines to maintain API compatibility, add a unit test in `tests/test_statevector_simulator.py`, and verify unitarity in the test.

## Reporting Issues

Please open a GitHub issue with a clear description of the problem, steps to reproduce, and your environment details (Python version, NumPy version, CuPy version if applicable, GPU model).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
