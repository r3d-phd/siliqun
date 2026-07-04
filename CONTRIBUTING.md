# Contributing to SiliQun

Thank you for your interest in contributing to SiliQun. This document
describes the development workflow, coding standards, and contribution
guidelines.

---

## Development Setup

```bash
git clone https://github.com/ralshehri0468/siliqun.git
cd siliqun
pip install -e ".[dev]"
```

---

## Code Standards

- **Formatting**: `black` (line length 100) and `ruff` for linting.
- **Type hints**: All public functions must have type annotations.
- **Docstrings**: NumPy-style docstrings for all public classes and functions.
- **Tests**: All new features must include `pytest` tests in `tests/`.

Run checks before submitting a pull request:

```bash
black siliqun tests
ruff check siliqun tests
pytest tests/ -v
```

---

## Adding a New Module

### New Device Architecture

1. Create a class in `siliqun/physics/devices/profiles.py` that inherits
   from `DeviceProfile`.
2. Register it with `@DEVICE_REGISTRY.register("device_name")`.
3. Implement `get_noise_params()`, `get_hamiltonian()`, and
   `get_native_gates()`.
4. Add a test in `tests/test_devices.py`.

### New Noise Model

1. Add Kraus operator generators to `siliqun/physics/noise/channels.py`
   following the existing pattern (e.g., `amplitude_damping_kraus`).
2. Add an `apply_*` function that applies the channel to an MPS or
   state vector.
3. Add the new model to `NoiseParams` if it requires configuration.

### New Plugin

1. Create a new file in `siliqun/plugins/`.
2. Implement the plugin using the `SiliQunPlugin` base class from
   `siliqun/plugins/__init__.py`.
3. Document the plugin in the README under the Plugin Interface section.

---

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Implement your changes with tests and documentation.
3. Ensure all tests pass and linting is clean.
4. Open a pull request with a clear description of the change.

---

## Reporting Issues

Please use the [GitHub issue tracker](https://github.com/ralshehri0468/siliqun/issues)
to report bugs or request features. Include a minimal reproducible example
where possible.
