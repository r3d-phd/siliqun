# SiliQun V2 Standalone Architecture

## Design decision

SiliQun V2 has three software levels. The separation keeps the simulation core reusable and prevents application-specific logic from becoming a hidden dependency of physics results.

| Level | Responsibility | Current baseline status |
|---|---|---|
| Core simulator | Circuit semantics, ideal state-vector evolution, pulse-schedule representation, validation primitives, and profile invariants. | Implemented. |
| Technology modules | Silicon-spin parameters, connectivity, native operations, provenance metadata, and later device-specific physics. | One literature-parameterised SiMOS nominal module is implemented. |
| Plugins | Optional circuit importers, solvers, SDK adapters, compiler backends, and visualisation. | Deferred; no plugin may change the core claim boundary without a separate review. |

## Current data flow

A user creates a `Circuit` or parses the declared OpenQASM subset. `StateVectorSimulator` evolves the circuit using ideal unitary gates. Separately, `GateToPulseCompiler` maps declared native operations to a `PulseSchedule` using a selected `TechnologyProfile`. These two paths are intentionally distinct in this baseline: pulse schedules are validated representations, not calibrated device executions.

## Profile provenance rules

Every profile declares a `calibration_status`. A `literature_parameterised` profile may contain DOI-linked nominal values but cannot be described as a named-device calibration. A `named_device` profile is rejected unless it supplies a device identifier, uncertainty description, held-out validation description, measurement model, and noise cross-spectrum description. This is a metadata guard, not a substitute for independent empirical review.

## Explicit non-goals

The baseline does not execute hardware jobs, export arbitrary waveform instructions, construct a virtual-QEC bridge, or provide an external-controller environment. It does not claim full standards conformance, physical pulse accuracy, process tomography, fault tolerance, or performance beyond its unit tests.
