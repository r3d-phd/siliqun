# SiliQun V2 D1 Read-Only Silicon Eligibility Protocol V1

## Scope and completed action

This protocol records a **public-evidence-only D1 eligibility assessment** conducted after D0 static preparation. It evaluates whether a candidate silicon-spin platform can supply the minimum metadata required before any calibration-support or PPO work is contemplated. The assessment performed desk research only. It did not create an account, log into a provider, contact a laboratory, submit a job, access an API, retrieve a calibration record, train PPO, or send a hardware command.

D1 is an access-and-provenance gate. Passing D1 would not demonstrate PPO efficacy, SiMORA execution, QEC performance, practical noise, or silicon-hardware performance. It would only permit a later, separately authorized D2 offline-data eligibility assessment.

## Strict D1 pass criterion

A candidate passes only if an authorized read-only route permits inspection of a **named device or revision** and exposes, at minimum, the device identity, connectivity or topology, native operations, control/readout metadata, and the availability and provenance of calibration records. A public platform description, publication, benchmark, or collaboration opportunity cannot substitute for any missing element.

## Evidence tracks

Three independently researched tracks were reconciled. The Quantum Motion/NQCC track examined the Silicon Cloverleaf testbed and the 2025 deployment announcement. The Intel/LQC track examined the Tunnel Falls silicon-spin platform, its published device characterization, and the QCF collaboration model. The local track checked whether either existing workspace already contained an authorized physical-device manifest or retained calibration record. The detailed source record is retained in `V2_D1_SILICON_ELIGIBILITY_SOURCE_NOTES_V1.md`.

## Outcome

| Candidate | Public evidence that is sufficient | Missing D1 evidence | Strict D1 status |
|---|---|---|---|
| **Intel Tunnel Falls through LQC/QCF** | Named 12-quantum-dot silicon-spin research platform; linear-array characterization; collaboration-based research-device distribution | Specific supplied device or revision; authorized read-only metadata path; device-specific topology/native-operation interface; control/readout configuration; calibration-record schema and availability | **REFUSED** |
| **Quantum Motion Silicon Cloverleaf at the NQCC** | Installed full-stack silicon-CMOS system; silicon-spin QPU; control stack compatible with Qiskit and Cirq; NQCC testbed identity | QPU revision; qubit count; connectivity; native operation set; control/readout configuration; calibration records; authorized read-only metadata route | **REFUSED** |
| **Existing SiVQD/SiMORA/SiliQun local records** | Versioned virtual models, synthetic calibration procedures, and unbound hardware templates | Named physical device; provider authorization; physical topology/operations/readout/calibration records | **REFUSED** |

## Selection decision

**Intel Tunnel Falls is the preferred conditional collaboration candidate.** The selection is based only on the relative maturity of its *public platform-level evidence*: its named silicon-spin device, twelve-dot scale, public technical characterization, and explicit research-laboratory distribution route. It does not confer access, identify a particular chip supplied to this project, establish calibration availability, or authorize the project to examine any metadata.

Quantum Motion’s NQCC system remains a valuable **conditional full-stack silicon-CMOS candidate**, especially for future deployment-level integration. Public information is presently less sufficient for the strict D1 metadata predicate because it does not expose an installed QPU revision, a coupling map, a native-operation manifest, or a documented read-only inspection route.

## Consequence for DRL and SiMORA

The D0 PPO plugin remains disabled. It cannot start training because the observation semantics, permitted actions, safety envelope, calibration provenance, and held-out evaluation definition must be bound to a named device’s authorized metadata. SiMORA remains outside the PPO plugin’s authority. No D2, D3, shadow-mode, active-control, or hardware claim is permitted from this result.

## Next required evidence

Progress requires a new authorization originating from the relevant provider or laboratory. The authorization must identify a device or revision and provide read-only access to the metadata specified by the D1 predicate. If that authorization arrives, its scope should be recorded in a `SiliconBackendManifest` before any circuit compilation, PPO training, or command path is considered.

## References

[1]: https://quantummotion.com/quantum-motion-delivers-the-industrys-first-full-stack-silicon-cmos-quantum-computer/ "Quantum Motion Delivers the Industry’s First Full-Stack Silicon CMOS Quantum Computer"
[2]: https://www.nqcc.ac.uk/quantum-computing-testbeds-in-the-uk/ "NQCC quantum computing testbeds"
[3]: https://www.intel.com/content/www/us/en/newsroom/news/quantum-computing-chip-to-advance-research.html "Intel’s New Chip to Advance Silicon Spin Qubit Research for Quantum Computing"
[4]: https://pubs.acs.org/nalefd/article/25/2/793/3751802 "Tunnel Falls: A 12-Quantum-Dot Spin-Qubit Device Made Using a 300 mm Wafer Process"
[5]: https://www.qubitcollaboratory.org/qubits-for-computing-foundry-qcf-single-page/ "LPS Qubit Collaboratory Qubits for Computing Foundry"
