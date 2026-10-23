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

## Round 2 (2026-10-13): full numerical re-grounding of the IFS551 paper

The IFS551 manuscript (private `siliqun-paper` repo, `main_tqe.tex`) was re-grounded end-to-end against per-seed artifacts in this repository. Every fidelity number in the paper now traces to a file listed below or in `evidence_ledger.csv`.

| Previous manuscript claim class | Decision | Basis |
|---|---|---|
| Replication "success" figures (PPO/DQL CZ $0.9996$/$1.0000$, donor DQN $0.9748$/$0.9999962$, GAA $0.9999$/$0.9877$, TQC "2 of 4 seeds > 0.99", 8-qubit $0.9957\pm0.0026$) | **Remove** | No per-seed artifacts exist; the one artifact claiming $1.0000\times5$ (r3_v5) is a degenerate placeholder (constant history from episode 0). Paper now reports the audited values instead (EXP-002..EXP-006). |
| `results_summary.csv` aggregate table (PPO 1.0000, E2 0.9999, E3 0.9995, E4-3Q 0.9999) | **Corrected in-repo** | Rewritten 2026-10-13 to audited values with PENDING_REGENERATION rows for missing artifacts (EXP-008). |
| SiliQunLib "50 pre-trained checkpoints on Zenodo" | **Remove** | No checkpoint artifacts in the repository; the `\cite{AlshehriSiliQunLib2026}` bibitem did not exist in the manuscript. |
| `qutip_qtrl` adapter (`siliqun/plugins/qutip_ctrl.py`) | **Label as earlier-draft interface** | Module not present in the released repository; the paper's QOC subsection now describes the repository's QOC runner and marks the listing as illustrative pseudocode. GRAPE/CRAB results remain retained (EXP-005). |
| E4 "N=2 up to N=12" scalability | **Narrow to N=2 and N=3** | Executed runs exist only for 2-qubit target families (3 seeds) and 3-qubit multi-target (jobs 180521/180658); N=4--12 was never executed (EXP-006). |
| DQL/SARSA E1 rows, E2 donor DQN, E3 "DQN" rows | **Remove pending regeneration** | No per-seed artifacts in the repository (DRL-002). |
