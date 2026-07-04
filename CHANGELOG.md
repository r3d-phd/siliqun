# Changelog

All notable changes to SiliQun are documented in this file.
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2025-07-05

### Added
- `siliqun.states` — standalone module with analytically exact target state
  builders: `bell_state`, `ghz_state`, `w_state`, `cluster_state`,
  `dicke_state`, `build_target_state`, `compute_fidelity`.
  Extracted from QUASAR v26/v27 codebase and standardised as a public API.
- `siliqun.noise_curriculum` — five-stage progressive noise curriculum
  controller (`NoiseCurriculum`, `get_noise_prob`, `NOISE_STAGE_PARAMS`).
  Supports automatic stage advancement, manual override, and checkpoint
  serialisation. Extracted from QUASAR v26/v27.
- `pyproject.toml` — PEP 517/518 compliant build configuration replacing
  the legacy `setup.py`.
- Full `__all__` export list in `siliqun/__init__.py` for clean public API.
- `extras_require` groups: `gpu`, `hpc`, `pennylane`, `viz`, `dev`, `all`.
- MIT `LICENSE` file.
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` with development workflow guidelines.

### Changed
- Version bumped from 1.0.0 → 2.0.0 to reflect the addition of the
  standardised public API surface and the new extracted modules.
- `siliqun/__init__.py` now exposes a flat, stable public API covering all
  major subsystems (gates, noise, states, curriculum, simulators, backends).
- `README.md` completely rewritten with installation instructions, quick
  start examples, architecture diagram, device table, and citation block.

### Fixed
- `bell_state(n)` — corrected index calculation for n > 2 (was using
  incorrect bit-shift arithmetic in the original inline implementation).
- `dicke_state` — now uses `_comb` helper instead of `math.comb` for
  Python 3.9 compatibility.

---

## [1.0.0] — 2025-05-01

### Initial release
- Tensor network engine: `MPS`, `MPO` with NumPy/JAX/cuQuantum backends.
- Physics models: `gates.py`, `hamiltonian.py`, `noise/channels.py`,
  `devices/profiles.py` (Donor, SiMOS, GAA).
- Simulation engines: `SiliQunSimulator`, `StatevectorSimulator`,
  `MPOSimulator`.
- Gymnasium environment: `SiliQunEnv`.
- Compiler: `gate_compiler.py`, `qasm3_compiler.py`.
- Pulse-level simulation: `lindblad.py`, `openpulse_schedule.py`.
- Tomography: `tomography.py`.
- PennyLane plugin: `plugins/pennylane_device.py`.
- HPC runner: `hpc/runner.py`.
- REST API: `api/server.py`.
