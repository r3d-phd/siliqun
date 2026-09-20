# SiliQun V2 B3 Prespecified Algorithm-Representation Protocol V1

**Status:** A bounded, local, software-only extension of B1/B2. B3 checks fixed circuit representations associated with selected algorithms. It neither runs an algorithm end-to-end nor binds SiliQun to SiVQD or SiMORA at runtime.

## Question

Can SiliQun produce a digest-only source export for three frozen, native-gate circuit representations that a separately written SiVQD evaluator may reproduce? The question concerns only declared ideal circuit ordering and output conventions. It does not compare pulse physics, noise, fidelity, hardware behavior, QEC, learned control, or algorithm success.

## Frozen representations

The suite is fixed before execution and uses only `rx`, `ry`, `rz`, and `cz`.

| Representation | Fixed construction | Bounded role |
|---|---|---|
| `deutsch_jozsa_two_qubit_balanced_native` | A two-qubit Deutsch–Jozsa balanced-oracle representation. The initial ancilla preparation, Hadamard-equivalent basis changes, and CNOT-equivalent oracle use only native rotations and CZ. | Tests phase-kickback circuit convention; no retained decision outcome. |
| `grover_two_qubit_fixed_phase_oracle_native` | A two-qubit fixed phase-oracle and diffusion representation, with every Hadamard and X compiled to native rotations and every controlled-not eliminated in favor of CZ and local rotations. | Tests a bounded amplitude-amplification circuit representation; no retained marked item or search answer. |
| `shor_n15_order4_compiled_orbit_component_native` | A three-qubit controlled cyclic work-orbit component. Its two work bits encode `1,2,4,8`, and conditioned on the control, the component implements the declared cyclic increment corresponding to multiplication by 2 on that restricted orbit modulo 15. A native Toffoli decomposition is used. | Tests one compiled order-four modular-multiplication component only; it contains no phase estimation, inverse QFT, measurement, classical post-processing, order, factor, or full Shor execution. |

The third representation is intentionally limited. Its encoding is a restricted work-orbit component, not reversible modular multiplication over every residue class modulo 15. No result from B3 may be called a Shor run, order-finding outcome, or factorization result.

Before any source export, `V2_B3_ALGORITHM_REPRESENTATION_STATIC_SPECIFICATION_RECEIPT_V1.json` publishes the complete ordered JSON gate list for all three representations, including every qubit target and floating-point angle, together with a SHA-256 hash of each circuit specification. The source exporter must reproduce those hashes. This static receipt is circuit-structure evidence only; it contains no state, probability, translation, pulse, answer, or algorithm-success data.

Each static circuit specification has the JSON form `{"n_qubits": integer, "gates": [{"name": string, "targets": [integer, ...], "parameters": [float, ...]}]}`. Its digest is SHA-256 of `json.dumps(specification, sort_keys=True, separators=(",", ":"))` encoded as UTF-8. The native ideal matrices are \(R_x(θ)=\begin{bmatrix}\cos(θ/2)&-i\sin(θ/2)\\-i\sin(θ/2)&\cos(θ/2)\end{bmatrix}\), \(R_y(θ)=\begin{bmatrix}\cos(θ/2)&-\sin(θ/2)\\\sin(θ/2)&\cos(θ/2)\end{bmatrix}\), \(R_z(θ)=\operatorname{diag}(e^{-iθ/2},e^{iθ/2})\), and `cz(q0,q1)=diag(1,1,1,-1)` in the two-qubit `|q1 q0>` order. They are ideal conventions only, not physical gate or pulse models.

## Native compilation and endpoint

An X-equivalent is `rx(π)` up to global phase. A Hadamard-equivalent is the ordered native sequence `rz(π)` then `ry(π/2)`, also up to global phase. A CNOT-equivalent is this Hadamard-equivalent on the target, followed by `cz(control,target)`, followed by the same Hadamard-equivalent. The order-four component uses a standard Toffoli decomposition rendered entirely through those CNOT-equivalents, native Hadamard-equivalents, and `rz(±π/4)` phase rotations.

For each representation, the exporter constructs a genuine `TranslationRequest` using `simos_nominal_profile()`, runs `StateVectorSimulator`, and retains only declared provenance, circuit/translation/schedule digests, operation categories, circuit-specification hash, and an SHA-256 digest of a canonical ideal probability list. Qubit 0 is the least-significant bit. Each probability is converted with `float()` then rounded using Python `round(value, 15)` with IEEE-754 round-half-to-even semantics. The rounded list is encoded with `json.dumps(values, separators=(",", ":"), ensure_ascii=True)`, UTF-8 encoded, and SHA-256 hashed. Raw state and probability arrays are deleted before receipt writing.

The agreement mechanism is deliberately narrow and explicit. A later evaluator receives the byte-pinned source receipt containing the exact gate list and source digest. It applies the listed gates to a fresh local state using independently written ideal matrices under the same least-significant-bit basis convention, creates a second transient probability list, applies the same rounding, serialization, and SHA-256 rule, and records only whether its digest equals the source digest. Agreement of that boolean for every fixed row, together with the source and evaluator audits, is the only basis for a B3 cross-software convention statement.

## Independent evaluator seeding and ordered evidence chain

Before the source export runs, the independently authored SiVQD B3 evaluator catalogue must copy the three static specification hashes from the frozen B3 static receipt into a separately versioned evaluator catalogue. That catalogue may contain only the three representation identifiers, static hashes, native gate set, LSB/canonicalization rules, retained-field requirements, forbidden-field requirements, and claim ceiling. It must not contain a source probability digest, source receipt byte hash, simulation output, or source-export result.

After the SiliQun source export is committed, the evaluator receives one copied source receipt. It computes the receipt SHA-256 locally, records that byte hash as a post-export input pin, and refuses every row whose `circuit_specification_sha256` differs from the corresponding pre-execution static pin. This separates pre-execution specification seeding from post-export input-integrity verification. The evaluator's source receipt hash is not a source-provided evaluator value.

The evidence order is mandatory: static specification and readiness; source export; source audit; byte-pinned independent evaluator; evaluator audit; receipt-bound final validation; and a separately invoked release-validation test. At static readiness, dynamic source/evaluator scope flags are **pending**, not implicitly true. B3-S6 passes only after both execution receipts explicitly report every prohibited runtime binding as false.

The export cannot retain a state vector, probability vector, pulse amplitude, pulse phase, pulse duration, fidelity, noise draw, reward, target state, measurement result, logical observable, algorithm answer, algorithm success probability, factoring output, or oracle answer. The profile remains `literature_parameterised`, not named-device evidence or a physical prediction.

## Independent SiVQD role

A later SiVQD evaluator must read a byte-pinned B3 source receipt, independently implement the ideal matrices and least-significant-bit convention, and calculate its own canonical probability digests. It cannot import SiliQun or reuse SiliQun simulation code. The source export alone is not a cross-simulator agreement result.

## Acceptance criteria

| Gate | Requirement | Failure disposition |
|---|---|---|
| B3-S1 | Baseline root, source identity, profile identifier, and literature-parameterised status are explicit. | `b3_refused_lineage_or_profile` |
| B3-S2 | Exactly the three prespecified representations and their pre-execution static specification hashes are exported. | `b3_refused_fixed_representations` |
| B3-S3 | Every representation produces a genuine translation receipt using only the declared native operations. | `b3_refused_translation_or_native_set` |
| B3-S4 | Exported records contain permitted digest/category fields only; every forbidden field is absent. | `b3_refused_retention_boundary` |
| B3-S5 | A separately authored SiVQD evaluator later matches every ideal probability digest. | `b3_refused_cross_simulator_sanity` |
| B3-S6 | No SiMORA controller, silicon adapter, detector, decoder, frame action, algorithm ingress, PPO, provider, or hardware path is bound. | `b3_refused_scope_boundary` |
| B3-S7 | A receipt-bound final validator and a separately invoked release-validation test confirm every upstream receipt and preserve this claim ceiling. | `b3_refused_release_validation` |

A B3 pass demonstrates only fixed-representation ideal software-level agreement under the stated convention. It cannot establish a digital twin, pulse accuracy, realistic SiMOS noise, physical validation, SiMORA execution, Deutsch–Jozsa execution, Grover success, Shor execution, factorization, order finding, fault tolerance, or hardware performance.

## References

[1]: https://quantum.cloud.ibm.com/learning/modules/computer-science/deutsch-jozsa "IBM Quantum Learning: The Deutsch–Jozsa Algorithm"
[2]: https://arxiv.org/abs/quant-ph/9605043 "Grover, A fast quantum mechanical algorithm for database search"
[3]: https://arxiv.org/abs/quant-ph/9508027 "Shor, Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer"
