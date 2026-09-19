# SiliQun V2 D1 Silicon Eligibility Disposition V1

**Disposition:** **D1 refused because authorized device-specific read-only metadata is absent from the evidence available to this task.**

## Main finding

The D1 review identified **Intel Tunnel Falls through the LPS Qubit Collaboratory/Qubits for Computing Foundry** as the most suitable *conditional* silicon-spin collaboration candidate. Its public record contains a named 12-quantum-dot silicon-spin platform, a public linear-array characterization, and an explicit research-laboratory distribution model.[1] [2] This is stronger public feasibility evidence than the Quantum Motion/NQCC route. It is not a D1 pass.

Neither Tunnel Falls nor Quantum Motion’s Silicon Cloverleaf testbed has publicly established the combination required by D1: a particular device or revision plus a provider-authorized read-only metadata route covering topology, native operations, control/readout configuration, and calibration-record availability. The local workspaces also contain no such physical-device record. The resulting status is therefore `D1_REFUSED_METADATA_AUTHORIZATION_ABSENT`.

## Evidence interpretation

Tunnel Falls should not be described as an accessible cloud backend. Intel and the LQC describe a collaborative research-device distribution model. Published device characterization establishes that the platform has a linear twelve-dot architecture and documents control/readout mechanisms, but it does not identify a particular supplied sample, export a calibration history, or authorize a reviewer to inspect device metadata.[1] [2]

Quantum Motion should not be described as a metadata-qualified device either. Its NQCC deployment announcement establishes an installed full-stack silicon-CMOS system with a silicon-spin QPU and Qiskit/Cirq-compatible stack. NQCC identifies the Silicon Cloverleaf testbed as spin-qubit technology with a two-dimensionally scalable QPU. Neither source publishes the installed QPU revision, topology, gate set, control/readout configuration, calibration records, or a read-only metadata endpoint.[3] [4]

## Consequence

This is a meaningful feasibility reduction, not a negative result about silicon-spin hardware. It identifies a prioritized collaboration path and enumerates the exact evidence needed to continue without inventing data. However, it does not permit PPO training, SiMORA integration, offline device-data analysis, shadow recommendations, hardware control, or claims about realistic SiMOS noise.

## Independent review

Veritas reviewed the factual public-evidence conclusion as **VERIFIED at 90% confidence**. The review explicitly preserves uncertainty about undisclosed agreements and is limited to the checked public record. The reasoning-chain endpoint returned a `NoneType` service error rather than a structured verdict; `localize_error` labelled the processing failure a logical gap. Because the submitted chain contained all four required Locate–Evaluate–Assess–Publish steps, this service behavior is recorded as **NO REASONING VERDICT**, not as an approval or a scientific rejection. The complete record is retained in `V2_D1_SILICON_ELIGIBILITY_VERITAS_REVIEW_RECORD_V1.json`.

## Required next event

The next event must come from a provider or laboratory: a scope-limited authorization naming a device or revision and granting read-only inspection of the required metadata. That evidence would allow a new D1 reassessment. It would not itself authorize D2 data transfer, PPO training, circuit submission, or hardware operation.

## References

[1]: https://www.intel.com/content/www/us/en/newsroom/news/quantum-computing-chip-to-advance-research.html "Intel’s New Chip to Advance Silicon Spin Qubit Research for Quantum Computing"
[2]: https://pubs.acs.org/nalefd/article/25/2/793/3751802 "Tunnel Falls: A 12-Quantum-Dot Spin-Qubit Device Made Using a 300 mm Wafer Process"
[3]: https://quantummotion.com/quantum-motion-delivers-the-industrys-first-full-stack-silicon-cmos-quantum-computer/ "Quantum Motion Delivers the Industry’s First Full-Stack Silicon CMOS Quantum Computer"
[4]: https://www.nqcc.ac.uk/quantum-computing-testbeds-in-the-uk/ "NQCC quantum computing testbeds"
