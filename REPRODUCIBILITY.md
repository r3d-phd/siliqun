# QUASAR v13 Adaptive — Reproducibility Guide

This document maps every claim in the QUASAR v13 paper to the exact log file, result JSON, and PBS job ID that produced it. All experiments were run on the Aziz HPC supercomputer at King Abdulaziz University (KAU).

---

## Computational Environment

| Component | Details |
|-----------|---------|
| Supercomputer | Aziz HPC, KAU (https://hpcc-kau.com) |
| GPU (GHZ/W/Dicke jobs) | NVIDIA A100 80 GB (queue: A100) |
| GPU (Cluster jobs) | NVIDIA H100 80 GB (queue: H100) |
| CPUs per job | 32 cores |
| RAM per job | 64 GB |
| Walltime limit | 72 hours |
| Conda environment | `quantum_drl_gpu` at `/app/utils/anaconda3-2024.02` |
| Python | 3.10 (conda) |
| PYTHONPATH | `/home/ralshehri0468/quasar_v13:/home/ralshehri0468/quasar_v12` |
| PBS scheduler | OpenPBS / Torque |

---

## Paper Claims → Evidence Map

### Claim 1: QUASAR v13 achieves F ≥ 0.9997 on the 2-qubit GHZ target

| Item | Value |
|------|-------|
| PBS Job ID | 181395 (fix2 run) |
| Log file | `results/logs/v13_quasar_v13_ghz_fix2.log` |
| Result JSON | `results/json/v13_ghz_w__2Q_GHZ_s42.json` |
| Key log line | `2Q/ghz/s42 step=346,685 best_F=0.9997 → PLATEAU early stop` |
| DEHB best α | 0.0190 |
| Seed | 42 |
| Compute node | kcn507 (A100) |
| Training steps | 346,685 (early stop via plateau detection) |
| SLM corrections | 4 (at steps 84,113 / 168,xxx / 252,339 / 336,452) |

### Claim 2: QUASAR v13 achieves F ≥ 0.9998 on the 2-qubit Cluster-linear target

| Item | Value |
|------|-------|
| PBS Job ID | 181404 (fix5 run) |
| Log file | `results/logs/v13_quasar_v13_cluster_fix5.log` |
| Result JSON | `results/json/v13_adaptive__2Q_Cluster_s42.json` |
| Key log line | `2Q/cluster_linear/s42 best_F=0.9998` |
| DEHB best α | 0.0148 (trial 1/12) |
| Seed | 42 |
| Compute node | kcn522 (H100) |

### Claim 3: QUASAR v11 baseline fails to exceed F = 0.99 for GHZ/W/Dicke at 2 qubits

| Item | Value |
|------|-------|
| Summary JSON | `results/json/v11_scaling_summary.json` |
| GHZ 2Q | best_F = 0.9869 (F < 0.99) |
| W 2Q | best_F = 0.9880 (F < 0.99) |
| Dicke-k3 2Q | best_F = 0.9876 (F < 0.99) |
| Cluster-linear 2Q | best_F = 0.9994 (F > 0.99 — already solved by v11) |

### Claim 4: DEHB entropy search selects α = 0.0190 as optimal for GHZ

| Item | Value |
|------|-------|
| Log file | `results/logs/v13_quasar_v13_ghz_fix2.log` |
| HP trial lines | `HP trial N/12: best_F=...` |
| Best trial | Trial with α=0.0190, best_F=0.9925 (from 12-trial DEHB search) |
| Figure | `results/figures/fig3_dehb_alpha_search.pdf` |

### Claim 5: SLM (Stochastic Landscape Mapping) corrections improve convergence

| Item | Value |
|------|-------|
| Log pattern | `SLM correction applied at step=N` |
| GHZ log | 4 SLM events in `v13_quasar_v13_ghz_fix2.log` |
| Cluster log | 1 SLM event in `v13_quasar_v13_cluster_fix5.log` |
| Figure | `results/figures/fig2_v13_ghz_fidelity_curve.pdf` |

---

## Ablation Study (v11 vs v13 at 2 Qubits)

| Target | v11 best_F | v13 best_F | Improvement |
|--------|-----------|-----------|-------------|
| GHZ | 0.9869 | **0.9997** | +0.0128 (+1.3%) |
| W | 0.9880 | pending | — |
| Cluster-linear | 0.9994 | **0.9998** | +0.0004 |
| Dicke-k3 | 0.9876 | pending | — |

Figure: `results/figures/fig4_ablation_v11_vs_v13.pdf`

---

## v11 Scaling Degradation

| Target | 2Q | 3Q | 4Q | 5Q |
|--------|----|----|----|----|
| GHZ | 0.9869 | 0.7927 | — | — |
| W | 0.9880 | 0.7896 | — | — |
| Cluster-linear | 0.9994 | 0.8883 | 0.3379 | — |
| Dicke-k3 | 0.9876 | 0.9906 | 0.4817 | 0.2798 |

Source: `results/json/v11_scaling_summary.json`  
Figure: `results/figures/fig1_v11_fidelity_vs_qubits.pdf`

---

## PBS Job History (Cumulative Bug Fix Log)

| Job ID | Status | Bug Fixed | Notes |
|--------|--------|-----------|-------|
| 181048 | COMPLETED | — | v11 All-Targets, completed Jun 15 08:27 KSA |
| 181395 | RUNNING | `factory` NameError (fix2) | GHZ+W+Dicke-k3, A100 kcn507, started Jun 16 18:23 |
| 181404 | RUNNING | `add_task_sample` → `.add()` (fix5) | Cluster 2Q→8Q, H100 kcn522, started Jun 16 20:24 |

### Cumulative Bug History in `quasar_v13_adaptive.py`

1. **`factory` NameError** (L691–697): `factory` was a stale reference to an old agent factory. Fixed by replacing with `_best_agent_cell`.
2. **`add_task_sample()` AttributeError** (L729): Method renamed to `.add()` in the curriculum buffer API. Fixed via `sed -i` on Aziz.
3. **Wrong conda path in PBS**: Corrected to `/app/utils/anaconda3-2024.02`, env `quantum_drl_gpu`.
4. **Missing `PYTHONPATH` in PBS**: Added `quasar_v13:quasar_v12` to PBS script.
5. **`--budget` not a valid arg**: Replaced with `--max-steps-base` in the DEHB call.

---

## File Index

### Result JSONs (`results/json/`)
- `v13_ghz_w__2Q_GHZ_s42.json` — **PRIMARY**: 2Q GHZ seed 42, best_F=0.9997
- `v13_adaptive__2Q_Cluster_s42.json` — **PRIMARY**: 2Q Cluster seed 42, best_F=0.9998
- `v11_scaling_summary.json` — v11 baseline across all targets and qubit counts
- `siliqun_r1_*.json`, `siliqun_r3_*.json`, `siliqun_r4_*.json` — SiliQun simulator validation results

### Training Logs (`results/logs/`)
- `v13_quasar_v13_ghz_fix2.log` — **PRIMARY**: Full training log for job 181395 (GHZ fix2)
- `v13_quasar_v13_cluster_fix5.log` — **PRIMARY**: Full training log for job 181404 (Cluster fix5)
- `v13_quasar_v13_ghz_w.log` — Previous GHZ+W run (completed 2Q/GHZ before factory crash)
- `LIVE_181395_ghz.log` — Live snapshot of job 181395 (downloaded Jun 17 ~10:00 KSA)
- `LIVE_181404_cluster.log` — Live snapshot of job 181404 (downloaded Jun 17 ~10:00 KSA)

### Publication Figures (`results/figures/`)
- `fig1_v11_fidelity_vs_qubits.{pdf,png}` — v11 baseline scaling degradation
- `fig2_v13_ghz_fidelity_curve.{pdf,png}` — v13 GHZ full-budget training curve
- `fig3_dehb_alpha_search.{pdf,png}` — DEHB entropy coefficient search
- `fig4_ablation_v11_vs_v13.{pdf,png}` — Ablation: v11 vs v13 at 2Q
- `fig5_v13_cluster_fidelity_curve.{pdf,png}` — v13 Cluster DEHB search phase

### Source Code (`results/source/`)
- `quasar_v13_adaptive.py` — Fixed training script (47 KB, all 5 bugs resolved)

---

## How to Reproduce

### Prerequisites
```bash
conda activate quantum_drl_gpu
export PYTHONPATH=/path/to/quasar_v13:/path/to/quasar_v12
```

### Run GHZ 2-qubit experiment (seed 42)
```bash
python quasar_v13_adaptive.py \
  --target ghz \
  --n-qubits 2 \
  --seed 42 \
  --max-steps-base 500000 \
  --dehb-trials 12 \
  --slm-interval 80000
```

### Run Cluster-linear 2-qubit experiment (seed 42)
```bash
python quasar_v13_adaptive.py \
  --target cluster_linear \
  --n-qubits 2 \
  --seed 42 \
  --max-steps-base 500000 \
  --dehb-trials 12 \
  --slm-interval 80000
```

### Regenerate publication figures
```bash
python gen_figures.py
# Output: results/figures/fig{1-5}_*.{pdf,png}
```

---

## Contact

**Researcher**: Abdulrahman Alshehri (ralshehri0468@kau.edu.sa)  
**Institution**: King Abdulaziz University, Faculty of Computing and Information Technology  
**Supervisor**: [Supervisor name]  
**Date**: June 2026
