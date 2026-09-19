# SiliQun V2 Standalone Baseline Disposition

**Disposition:** **Published software-boundary milestone**

## Decision

The `v2-standalone-baseline` branch establishes the canonical, ML-free software starting point for the SiliQun V2 line. It is a new orphan branch, intentionally isolated from the legacy application tree. Its root commit is `143cf5100504ab378f91c61f49c3e14b3901f04d`; the committed legacy reference is `14a49bf957b7aeacffc1bd15ba91293d6fd31b83`.

This is not a revalidation of prior experimental results. It is a source and interface boundary: the branch defines the software objects that a later profile/circuit translation contract may refer to without coupling that contract to a mutable, mixed-purpose application repository.

## Implemented and verified baseline

| Component | Evidence | Disposition |
|---|---|---|
| Source lineage | `BASELINE_MANIFEST.json` pins the legacy commit and tree. | Present. |
| Purification | `BASELINE_PURIFICATION_RECEIPT.json` reports a 748-file legacy tree, an 18-file orphan baseline root, and `LICENSE` as the only byte-identical shared path. | PASS. |
| Circuit interface | `Circuit` and a static OpenQASM 3 subset parser; a documented Bell-state example is regression-tested. | Implemented and tested. |
| Pulse interface | `Pulse`, `PulseSchedule`, and a profile-bound native-operation compiler. | Implemented and tested. |
| Profile provenance | `TechnologyProfile` records calibration status and rejects incomplete `named_device` metadata. | Implemented and tested. |
| Simulator | Bounded ideal state-vector evolution and pure-state fidelity. | Implemented and tested. |
| Functional verification | `BASELINE_VERIFICATION_RECEIPT.json` reports 12 passing unit tests. | PASS. |

## Claim ceiling

The branch supports the narrow statement that **SiliQun V2 now has a source-isolated and test-verified standalone software baseline**. It does not support a claim about a named device, calibrated SiMOS noise, pulse fidelity, full OpenQASM/OpenPulse conformance, hardware execution, QEC, fault tolerance, or any external-controller performance.

The built-in SiMOS nominal profile is literature-parameterised metadata. The `named_device` profile guard is a schema safeguard, not empirical evidence or a replacement for a measured calibration package.

## Independent review record

Veritas factual review of the software-boundary claim returned **VERIFIED** at **95% confidence**. It recommended publishing the manifest and receipt, documenting the parser, and explaining the orphan-branch rationale; this branch now includes all three.

The first Veritas reasoning-chain review identified an incomplete link between a baseline and later contract design. A revised, receipt-backed chain explicitly supplied that link, but the service returned a null-processing error rather than a structured verdict. `localize_error` attributed the rendering failure to the receipt-processing step. This is recorded as **NO VERDICT**, not as a reasoning approval or a scientific clearance.

## Next controlled step

Design a **versioned profile/circuit translation contract**, without implementing an adapter yet. The contract should bind: the baseline root commit; a profile identifier and declared calibration status; a circuit grammar version; native-operation and pulse-schema versions; expected outputs; retained evidence; and the explicit prohibition on promoting baseline simulation outputs to physical or hardware claims. The contract must be independently reviewed before any cross-project interface code is added.
