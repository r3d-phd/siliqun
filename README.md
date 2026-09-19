# SiliQun V2 Standalone Baseline

SiliQun V2 is a **standalone silicon-spin simulation foundation**. This branch establishes a small, source-controlled core for circuit representation, pulse schedules, technology-profile provenance, and ideal state-vector simulation. It is the canonical starting point for the V2 software line.

The baseline is intentionally narrow. It does not include learned-control environments, training code, external-controller integrations, hardware export, virtual-QEC bridges, or named-device calibration claims. The scope is recorded in [`BASELINE_MANIFEST.json`](BASELINE_MANIFEST.json).

The branch is intentionally an orphan baseline. [`BASELINE_PURIFICATION_RECEIPT.json`](BASELINE_PURIFICATION_RECEIPT.json) records the legacy revision, confirms that the baseline root has no parent, and checks that the only retained path shared with the legacy source tree is the MIT license.

## What is implemented

The core package provides an immutable circuit model, a documented OpenQASM 3 input subset, an ideal state-vector simulator, a pulse-schedule schema, a small native-operation pulse compiler, and profile metadata with explicit calibration status. The built-in SiMOS profile is **literature-parameterised**, not a named-device digital twin.

```python
from siliqun import Circuit, StateVectorSimulator, fidelity

circuit = Circuit(2).h(0).cx(0, 1)
result = StateVectorSimulator(2).run(circuit)
print(result.probabilities)
```

## Installation and verification

No optional runtime packages are required for the baseline.

```bash
python -m unittest discover -s tests -v
```

For local source use, run the same command with `PYTHONPATH=src`.

## Input boundary

`Circuit.from_openqasm3` parses a restricted static subset: a single `qubit[n]` register and the `x`, `y`, `z`, `h`, `rx`, `ry`, `rz`, `cx`, `cz`, and `swap` instructions. It does not claim full OpenQASM 3 conformance. `PulseSchedule` represents named drive, exchange, virtual-Z, and measurement events, but it is not a full OpenPulse implementation.

The repository includes [`examples/bell_state.qasm`](examples/bell_state.qasm), which is parsed and simulated by the regression tests.

## Technology profiles and calibration language

`TechnologyProfile` carries technology identity, connectivity, native operations, numerical parameters, DOI-bearing citations, and a declared calibration status. The class rejects a profile labelled `named_device` unless a device identifier, uncertainty model, held-out validation descriptor, measurement model, and noise cross-spectrum descriptor are provided. The baseline includes no such profile.

## Extension model

The code is divided into three levels. The **core simulator** contains circuits, ideal evolution, pulse schedules, and profile validation. **Technology modules** contribute profile objects and physics-specific mappings. **Plugins** may add importers, compilers, or solvers only through a separately reviewed extension interface. The current baseline contains the core and one literature-parameterised technology module; plugin APIs remain an explicit next-stage task.

## Static translation and DRL preparation

The branch now includes a versioned [profile–circuit translation contract](docs/V2_PROFILE_CIRCUIT_TRANSLATION_CONTRACT_V1.md) and a [safety-constrained PPO calibration-plugin protocol](docs/V2_SAFE_PPO_CALIBRATION_PLUGIN_PROTOCOL_V1.md). Both are **design-only**. The static contract maps a declared circuit to a non-executable profile-bound pulse representation. The PPO specification fixes one algorithm, a maximum of five qubits, a pre-approved action-template registry, mandatory non-learning comparators, convergence and scaling ledgers, and a shadow-to-active progression. It includes no learned-control environment, training loop, checkpoint, provider API, hardware command, SiMORA modification, or performance result.

## Evidence boundary

Passing the unit tests confirms only the implemented software behaviours. It does not validate hardware physics, a real device, pulse calibration, QEC, or external-controller performance. See [`docs/LEGACY_BOUNDARY.md`](docs/LEGACY_BOUNDARY.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full boundary.

## License

MIT. See [`LICENSE`](LICENSE).
