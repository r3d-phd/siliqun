# Claim disposition for the remediated SiliQun paper

This register translates the forensic audit into publication decisions. A retained claim must be supported by a reproducible command and a raw output listed in `evidence_ledger.csv`.

| Previous manuscript claim class | Decision | Basis for the remediated paper |
|---|---|---|
| Pulse-level density-matrix evolution | **Retain, narrowed** | Retain only the implemented T1/T2* Lindblad solver after the complex-coherence and vectorisation fixes, with analytic one-qubit validation. |
| GPU-accelerated Lindblad propagation | **Remove** | The present RK4 and matrix-exponential paths use NumPy/SciPy. GPU support must not be claimed until implemented and benchmarked. |
| Lindblad charge-noise/depolarisation modelling | **Remove or label future work** | An identity collapse operator is physically null. No independent implementation/validation supports the historical statement. |
| Hardware-calibrated digital-twin status | **Replace** | Describe the device profiles as literature-informed parameter presets, not calibration to a specific device. |
| Large platform-comparison fidelity table | **Remove pending rerun** | The reported values conflict internally and lack released matched workloads and raw outputs. |
| DRL/PPO SiMOS control performance | **Remove pending exact release** | The released example does not reproduce the paper’s stated algorithm, device, noise model, episode length, or results. |
| Statevector/MPS scaling narrative | **Retain, narrowed** | State asymptotic complexity and distinguish it from demonstrated wall-clock scaling. Do not present unverified memory or performance estimates as measured facts. |
| Code availability | **Retain, corrected** | Point to the SiliQun-specific reproducibility package and require a tagged archival release before submission. |

## Evidence that may be reported now

| Validation | Command | Result |
|---|---|---|
| Amplitude damping | `bash paper_reproducibility/run_smoke_validation.sh` | $|P_1^{\mathrm{sim}}-e^{-t/T_1}| = 9.70\times10^{-14}$ for the declared one-qubit reference. |
| Pure dephasing | `bash paper_reproducibility/run_smoke_validation.sh` | $||\rho_{01}^{\mathrm{sim}}|-\frac12e^{-t/T_2^*}| = 4.85\times10^{-14}$ for the declared one-qubit reference. |
| Integrator agreement | `bash paper_reproducibility/run_smoke_validation.sh` | $\|\rho_{\mathrm{RK4}}-\rho_{\exp}\|_F = 5.55\times10^{-12}$ for the declared static-drive reference. |

These checks establish numerical consistency for limited analytic references. They do not validate a calibrated hardware device, an unmodelled noise process, a multi-qubit benchmark, or DRL efficacy.
