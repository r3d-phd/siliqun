# SiliQun V2 Profile–Circuit Translation Contract V1

**Status:** Design and static-validation contract only. This document does not authorise hardware access, circuit submission, pulse export, calibration, learned-policy training, or a SiMORA integration.

## Purpose

This contract defines the minimum immutable objects needed to translate a SiliQun V2 circuit into a profile-bound pulse-schedule representation without confusing that representation with a physical device program. It turns the published V2 baseline into a stable reference point for future interface work. The contract is deliberately independent of SiMORA, device-provider APIs, and learned control.

## Contract identity

| Field | Required value or rule |
|---|---|
| Baseline source | SiliQun V2 orphan root `143cf5100504ab378f91c61f49c3e14b3901f04d` |
| Translation version | `siliqun-profile-circuit-translation-v1` |
| Circuit grammar | `siliqun-circuit-v1`, the declared static OpenQASM subset and `Circuit` model |
| Pulse schema | `siliqun-pulse-schedule-v1` |
| Profile schema | `siliqun-technology-profile-v1` |
| Current profile | `simos-nominal-literature-v1`, marked `literature_parameterised` |

Every translation request must carry all five version identities, a SHA-256 manifest hash for the source objects, and an acquisition timestamp when any non-static profile evidence is used. The translator rejects any missing identity, mismatched profile identifier, unsupported circuit operation, or incompatible native gate.

## Required inputs and permitted output

A `TranslationRequest` contains a validated `Circuit`, a `TechnologyProfile`, and the declared version fields above. The profile is required to expose its native operation set and calibration status. A future named-device profile must also include device identity, uncertainty, held-out validation, readout/measurement characterization, and noise cross-spectrum descriptors. The existing nominal SiMOS profile does not meet those named-device conditions and must remain labelled literature-parameterised.

The only permitted output is a `TranslationReceipt`. It records the request identity, accepted profile identity and calibration status, circuit digest, declared native operation mapping, pulse-schedule digest, and a machine-readable claim ceiling. It is not an executable provider payload, arbitrary waveform, calibration update, or hardware job.

## Validation rules

| Gate | Condition | Refusal status |
|---|---|---|
| T0 — baseline lineage | Root commit and contract versions match the catalogue. | `translation_refused_lineage` |
| T1 — circuit validity | Circuit passes the V2 grammar and only uses declared operations. | `translation_refused_circuit` |
| T2 — profile integrity | Profile schema, connectivity, citations, and calibration status are valid. | `translation_refused_profile` |
| T3 — native mapping | Each translated operation has an explicit profile-bound mapping. | `translation_refused_native_mapping` |
| T4 — named-device evidence | Any `named_device` label includes all required evidence fields. | `translation_refused_named_device_evidence` |
| T5 — output boundary | Output is a receipt and pulse-schedule representation only. | `translation_refused_output_scope` |

The current baseline may pass T0–T3 and T5 with its nominal profile. It cannot pass a named-device interpretation because its calibration status is deliberately not `named_device`.

## Relationship to future DRL work

The contract is a prerequisite for, not an implementation of, learned calibration. A future PPO plugin may read only a versioned translation receipt and an independently authorized calibration-observation schema. It cannot alter a circuit, pulse schedule, profile, decoder, SiMORA decision, or calibration manifest. This separation preserves an auditable boundary between simulation, recommendation, and physical control.

## Claim ceiling

A valid translation receipt establishes only **software-level compatibility between one declared circuit representation and one declared profile-bound pulse representation**. It does not establish OpenQASM or OpenPulse conformance, pulse accuracy, physical implementability, calibrated SiMOS noise, hardware availability, QEC performance, or algorithm execution on SiMORA.

## References

[1]: https://github.com/r3d-phd/siliqun/tree/v2-standalone-baseline "SiliQun V2 standalone baseline branch"
[2]: https://arxiv.org/abs/1707.06347 "Proximal Policy Optimization Algorithms"
