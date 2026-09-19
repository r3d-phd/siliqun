# SiliQun V2 B2 Prespecified Native-Gate Suite Protocol V1

**Status:** A bounded, local, software-only successor to B1. B2 broadens ideal convention coverage without merging SiliQun with SiVQD or authorizing a SiMORA runtime.

## Question

Can a source-side SiliQun V2 export provide a fixed, digest-only representation of five native-gate circuits for a later independently written SiVQD evaluator? B2 tests circuit-convention coverage only. It does not compare pulse physics, noise, fidelity, hardware behavior, QEC, learned control, or algorithms.

## Frozen suite and coverage

The suite is fixed before execution and contains five circuits that use only `rx`, `ry`, `rz`, and `cz`. The first is `rx(π/3)` on one qubit. The second is `ry(−π/4)` followed by `rz(π/5)` on one qubit. The third is `rx(π/2)` followed by `rz(π/2)` on one qubit. The fourth is `ry(π/3)` on q0, `rx(π/5)` on q1, `cz(q0,q1)`, `rz(π/7)` on q0, `ry(π/4)` on q0, and `rx(π/6)` on q1. The fifth is `rz(2π)` on one qubit.

The suite adds non-B1 rotation angles, a negative angle, phase after superposition, explicitly ordered two-qubit local operations around `cz`, and a probability-level global-phase invariant. It is not an algorithm suite and it is not chosen from observed result values.

## Canonical endpoint

For each circuit, SiliQun constructs a genuine `TranslationRequest` using `simos_nominal_profile()`, runs `StateVectorSimulator`, and retains only the circuit specification, source identities, profile status, circuit digest, translation-manifest digest, pulse-schedule digest, operation categories and target arities, and an SHA-256 digest of the ideal probability vector. Qubit 0 is the least-significant bit. For two qubits, the list order is `|q1 q0> = |00>, |01>, |10>, |11>`.

Before hashing, every probability is converted with `float()` and then Python `round(value, 15)` is applied. This uses IEEE-754 round-half-to-even semantics. The rounded list is serialized using `json.dumps(values, separators=(",", ":"), ensure_ascii=True)`, UTF-8 encoded, and SHA-256 hashed. The state vector and probability vector are process-local values; both are deleted before the receipt is written.

The receipt must not retain a pulse amplitude, pulse phase, pulse duration, fidelity, noise draw, reward, target state, measurement result, logical observable, or algorithm answer. The profile status must remain `literature_parameterised`; it cannot be presented as named-device calibration or physical prediction.

## Independent SiVQD role

A later SiVQD B2 evaluator must independently implement the ideal matrices and declared least-significant-bit convention. It may read the B2 receipt as data but cannot import SiliQun or reuse SiliQun simulation code. It must validate source bytes and lineage, independently compute its own canonical probability digest for every declared circuit, compare only the digest endpoint, and retain aggregate boolean/digest outcomes. The source export alone is not cross-simulator agreement.

## Acceptance criteria

| Gate | Requirement | Failure disposition |
|---|---|---|
| B2-S1 | Baseline root, source identity, profile identifier, and `literature_parameterised` status are explicit. | `b2_refused_lineage_or_profile` |
| B2-S2 | Exactly the five prespecified circuits and declared native gate orders are exported. | `b2_refused_fixed_suite` |
| B2-S3 | Every circuit produces a genuine translation receipt and the declared operation categories. | `b2_refused_translation` |
| B2-S4 | The export contains permitted digest/category fields only; all forbidden fields are absent. | `b2_refused_retention_boundary` |
| B2-S5 | A separately authored SiVQD evaluator later matches all five ideal probability digests. | `b2_refused_cross_simulator_sanity` |
| B2-S6 | No SiMORA controller, silicon adapter, detector, decoder, frame action, algorithm ingress, PPO, provider, or hardware path is bound. | `b2_refused_scope_boundary` |
| B2-S7 | A receipt-bound final validator and a separately invoked release-validation test confirm every upstream receipt and preserve the claim ceiling. | `b2_refused_release_validation` |

A B2 pass demonstrates only fixed-suite, ideal, software-level agreement under the declared convention. It cannot establish a digital twin, pulse accuracy, realistic SiMOS noise, physical validation, SiMORA execution, Shor, Grover, Deutsch–Jozsa, any other algorithm execution, fault tolerance, or hardware performance.

## Role separation

The SiliQun exporter creates source-side input evidence only. The later independent SiVQD evaluator produces an independent calculation and determines whether the digest pairs match. A combined B2 agreement claim requires a passing source export, passing source audit, source-byte-pinned independent evaluation, passing evaluator audit, receipt-bound final validation, and a separately invoked release-validation test that confirms the final validation receipt passed. The release-validation test must read the final receipt afresh; it cannot merely assume that the final validator was run.

## References

[1]: [SiliQun V2 profile–circuit translation contract](V2_PROFILE_CIRCUIT_TRANSLATION_CONTRACT_V1.md)
[2]: [SiliQun V2 B1 cross-simulator sanity protocol](V2_B1_CROSS_SIMULATOR_SANITY_PROTOCOL_V1.md)
[3]: file:///home/ubuntu/sivqd_private/docs/SILIQUN_SIVQD_SIMORA_STATIC_BRIDGE_DISPOSITION_V1.md "SiVQD B0 static bridge disposition"
