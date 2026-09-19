# SiliQun V2 B1 Fixed-Circuit Cross-Simulator Sanity Protocol V1

**Status:** Prespecified local software-only export. This protocol is a successor to the static B0 bridge boundary; it does not merge SiliQun with SiVQD or authorize a SiMORA runtime.

## Purpose

B1 creates a small, reproducible SiliQun V2 export that a separately written SiVQD evaluator may inspect without importing SiliQun. The study checks only whether the two implementations share the declared **ideal circuit ordering and output convention** for a fixed native-gate suite. It does not compare pulse physics, noise, fidelity, hardware behavior, QEC, learned control, or algorithms.

## Frozen suite

The suite contains two circuits that use only the currently declared native operations: (i) one `rx(π/2)` operation on one qubit, and (ii) `ry(π/2)`, `rx(π/2)`, `cz`, and `rz(π/3)` on two qubits. The SiliQun translator and ideal state-vector simulator execute each circuit locally. No provider, network service, device metadata, or external package is used.

## SiliQun export rule

For each circuit, the exporter must create a genuine `TranslationRequest` and `TranslationReceipt` with `simos_nominal_profile()`, run `StateVectorSimulator`, and retain only the circuit specification, source identities, profile status, circuit digest, translation-manifest digest, pulse-schedule digest, operation categories and target arities, and an SHA-256 digest of a canonical ideal probability vector. It must discard the state vector and probability vector before writing the receipt.

The export must not retain or emit a pulse amplitude, pulse phase, pulse duration, fidelity, noise draw, reward, target state, measurement result, logical observable, or algorithm answer. It must state `literature_parameterised` exactly and cannot be described as a named-device calibration or physical prediction.

## Independent SiVQD role

The later SiVQD B1 evaluator must independently reimplement the ideal matrices and declared least-significant-bit qubit convention. It may read the B1 receipt as data but must not import the SiliQun package or reuse SiliQun simulation code. It must compare its own probability digest to the SiliQun-exported digest, validate the circuit and operation-category identities, and retain aggregate pass/fail outcomes only.

## Acceptance criteria

| Gate | Requirement | Failure disposition |
|---|---|---|
| B1-S1 | Exact baseline root, profile identifier, and literature-parameterised status. | `b1_refused_lineage_or_profile` |
| B1-S2 | Exactly the two prespecified circuits and declared gate orders. | `b1_refused_fixed_suite` |
| B1-S3 | Each circuit produces a real translation receipt and only declared operation categories. | `b1_refused_translation` |
| B1-S4 | Export contains digests and categories only; every forbidden field is absent. | `b1_refused_retention_boundary` |
| B1-S5 | SiVQD independent evaluator agrees on both ideal probability digests. | `b1_refused_cross_simulator_sanity` |
| B1-S6 | No SiMORA controller, silicon adapter, detector, decoder, frame action, algorithm ingress, PPO, provider, or hardware path is bound. | `b1_refused_scope_boundary` |

A B1 pass demonstrates only fixed-suite, ideal, software-level agreement under a declared convention. It cannot establish a digital twin, pulse accuracy, real SiMOS noise, cross-simulator physical validation, SiMORA execution, algorithm execution, fault tolerance, or hardware performance.

## LEAP pre-execution review and role separation

**Learn.** Veritas rejected the first B1 reasoning chain because it blurred the roles of the SiliQun exporter and a future independent SiVQD evaluator. The SiliQun export is not itself a cross-simulator comparison.

**Evaluate.** The exporter generates the declared source-side inputs only: fixed circuit specifications, translation and schedule digests, operation categories, and ideal probability digests. The independently authored SiVQD evaluator generates its own outputs and determines whether the fixed digest pairs match. Neither side can independently support a cross-simulator agreement claim.

**Apply.** B1 now has two linked but distinct evidence stages. The present SiliQun stage may claim only an actual, bounded source export. A combined B1 agreement claim is permitted only if the later evaluator independently reproduces the declared probability digests, the separately written audit passes, and the final disposition preserves the same claim ceiling.

**Persist.** The final record must distinguish `source_export_completed` from `cross_simulator_agreement_evaluated`; it must record an absent evaluator as a non-result, not as agreement or failure.

## References

[1]: [SiliQun V2 profile–circuit translation contract](V2_PROFILE_CIRCUIT_TRANSLATION_CONTRACT_V1.md)
[2]: [SiliQun V2 standalone baseline disposition](V2_STANDALONE_BASELINE_DISPOSITION.md)
[3]: file:///home/ubuntu/sivqd_private/docs/SILIQUN_SIVQD_SIMORA_STATIC_BRIDGE_DISPOSITION_V1.md "SiVQD B0 static bridge disposition"
