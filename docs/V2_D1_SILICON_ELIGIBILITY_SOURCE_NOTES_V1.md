# SiliQun V2 D1 Silicon Eligibility: Public Source Notes V1

## Scope

These notes preserve the public-source evidence used for a **read-only D1 eligibility assessment**. They do not report authenticated metadata access, a device login, a provider API call, a calibration record, a circuit submission, or a hardware result. The assessment asks a narrow question: does a candidate presently have a named device or revision **and** an authorized read-only path to the metadata required to create a `SiliconBackendManifest`?

The required manifest fields are device identity and revision, qubit count, connectivity, native operations, control/readout architecture, capability timestamp, and access class. Calibration-record availability is included because the planned PPO calibration-support work cannot be designed against unavailable records.

## Quantum Motion at the NQCC

Quantum Motion announced on 15 September 2025 that its first deployed full-stack silicon CMOS system had been installed at the UK National Quantum Computing Centre (NQCC) in Harwell. The announcement identifies a silicon-spin QPU, a Qiskit/Cirq-compatible user interface and control stack, and 300 mm CMOS fabrication. It does **not** state an installed-QPU revision, qubit count, coupling map, native operation set, readout configuration, calibration export, or an authorized read-only metadata route.[1]

The NQCC identifies the Quantum Motion testbed project as **Silicon Cloverleaf**, a spin-qubit system with a two-dimensionally scalable QPU, control systems, and a user interface. That is a project/testbed identifier and qualitative platform description, not a device serial/revision or a topology and calibration manifest.[2]

**Source-bound conclusion:** Quantum Motion/NQCC has strong public platform evidence but does not meet the strict D1 eligibility predicate. It remains a conditional target for an authorized metadata review, not a D1-qualified hardware record.

## Intel Tunnel Falls through the LQC/QCF route

Intel publicly identifies **Tunnel Falls** as a 12-qubit silicon-spin research chip fabricated on 300 mm wafers. The company states that it is made available to research laboratories through a collaboration with the LPS Qubit Collaboratory (LQC) and the Qubits for Computing Foundry (QCF) program. Intel further states that the 12-dot devices can form four to twelve qubits depending on laboratory operation.[3]

A public technical article characterizes Tunnel Falls as a linear 12-quantum-dot array with 25 consecutive gates, four single-electron-transistor detectors, plunger and barrier control, individual electric-dipole spin-resonance control, and tunable nearest-neighbour exchange coupling. These are platform-level design and characterization observations. They are not a supplied-device topology manifest, a versioned public gate interface, or an authorized calibration-record export.[4]

The LQC/QCF describes a collaboration and foundry-distribution model. It states that Intel provides 12-quantum-dot devices to research groups and that the user–foundry process includes design rules, pre-screening, and measurement feedback. It does not describe a public or reviewer-authorized read-only metadata portal, a device-specific manifest, a calibration-record repository, or an entitlement to inspect such data.[5]

**Source-bound conclusion:** Tunnel Falls is the most concrete public silicon-spin candidate because the platform name, nominal dot count, technology, and collaboration route are explicit. It still does not meet the strict D1 eligibility predicate: no specific supplied device/revision, authorized read-only metadata route, per-device topology/gate/control/readout manifest, or accessible calibration-record schema is public in the checked sources.

## Boundary for the next stage

A public testbed description or a collaboration invitation is not a substitute for the required read-only evidence. D1 begins only when a provider or laboratory authorizes inspection of the named system’s metadata. Until then, the project may state only that **Tunnel Falls is the preferred conditional collaboration candidate**, while **Quantum Motion/NQCC remains a conditional full-stack silicon-CMOS candidate**.

## References

[1]: https://quantummotion.com/quantum-motion-delivers-the-industrys-first-full-stack-silicon-cmos-quantum-computer/ "Quantum Motion Delivers the Industry’s First Full-Stack Silicon CMOS Quantum Computer"
[2]: https://www.nqcc.ac.uk/quantum-computing-testbeds-in-the-uk/ "NQCC quantum computing testbeds"
[3]: https://www.intel.com/content/www/us/en/newsroom/news/quantum-computing-chip-to-advance-research.html "Intel’s New Chip to Advance Silicon Spin Qubit Research for Quantum Computing"
[4]: https://pubs.acs.org/nalefd/article/25/2/793/3751802 "Tunnel Falls: A 12-Quantum-Dot Spin-Qubit Device Made Using a 300 mm Wafer Process"
[5]: https://www.qubitcollaboratory.org/qubits-for-computing-foundry-qcf-single-page/ "LPS Qubit Collaboratory Qubits for Computing Foundry"
