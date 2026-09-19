# SiliQun V2 DRL Preparation Disposition V1

**Disposition:** **D0 static preparation completed and published.**

## Result

SiliQun V2 now contains two source-isolated, versioned prerequisites for a future safety-constrained SiMOS calibration-support study. The first is a profile–circuit translation contract that validates circuit, profile, native-operation mapping, and schema lineage before producing a **non-executable** pulse-schedule representation. The second is a design-only PPO calibration-plugin protocol that fixes a five-qubit limit, a bounded action-template registry, a disabled command mode, mandatory non-learning comparators, convergence diagnostics, scaling diagnostics, and a shadow-to-active progression.

The static readiness receipt passed **11/11** checks. It confirms that the nominal SiMOS profile is still `literature_parameterised`, that the translation output is not a device command, that PPO is the sole designated algorithm, that the scope remains at five qubits or fewer, that TE-PWS is reserved rather than implemented, and that every declared runtime operation is explicitly refused. The V2 branch test suite passed **19 tests**. Baseline verification and source-purification receipts also passed.

## What is established

| Item | Established status |
|---|---|
| Versioned profile–circuit representation contract | Implemented and statically validated |
| Non-executable pulse-schedule receipt | Implemented and validated |
| Safety-constrained PPO plugin specification | Implemented as a static contract only |
| PPO algorithm limit and five-qubit scope | Enforced by static validation |
| Approved action-template registry | Declared and statically bounded |
| Fixed, Bayesian, and physics-informed comparator identities | Declared as mandatory future baselines |
| Explicit runtime-operation refusals | Implemented and unit-tested |
| SiMORA deterministic supervisor | Explicitly outside the plugin’s authority |

## What is not established

This milestone does not train PPO, produce a policy checkpoint, estimate TE-PWS, access a named device, obtain calibration records, use a provider API, submit a job, export a pulse, alter SiMORA, run QEC, or demonstrate hardware or physical performance. The literature-parameterised nominal profile is not a named-device calibration package.

## Independent review status

Veritas’s final factual scope review returned **VERIFIED at 95% confidence** after explicit runtime-operation refusal guards and tests were added. The LEAP first-pass reasoning review returned `VALID` at 95%, but its meta-review disagreed because it treated the deliberately unattempted D1–D6 gates as a contradiction rather than the intended D0-only result. A revised D0-only reasoning request then failed with a `NoneType` service error; `localize_error` returned a logical-gap diagnosis. These two reasoning endpoints are recorded as **NO OVERALL REASONING APPROVAL**. They do not change the narrower factual verdict, and they are not presented as scientific clearance. The full record is in `V2_DRL_PREPARATION_VERITAS_REVIEW_RECORD_V1.json`.

## LEAP conclusion

**Locate:** prior local DRL studies did not establish a transferable hardware or SiMORA controller. **Evaluate:** a genuine future calibration-support task requires partial observations, sequential actions, fixed non-learning comparators, safety masks, and temporal holdout. **Assess:** the present package supplies only D0 software contracts and mechanically enforces their non-executable boundary. **Publish:** the D0 artifact is released with all later gates explicitly unattempted.

## Next permitted action

The next action is **not PPO training**. It is a separately authorized, read-only D1 eligibility review for one named silicon device. That review must supply device identity, topology, declared native operations, control/readout semantics, calibration-record availability, and authorization boundaries. Without those records, the project remains at D0.

## References

[1]: https://github.com/r3d-phd/siliqun/tree/v2-standalone-baseline "SiliQun V2 standalone baseline branch"
[2]: https://arxiv.org/abs/1707.06347 "Proximal Policy Optimization Algorithms"
