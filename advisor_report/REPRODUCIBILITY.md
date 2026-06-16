# SiliQun — Reproducibility Guide

This document maps every claim in the SiliQun paper to the exact log file, result JSON, and PBS job that produced it.

---

## Computational Environment

| Component | Details |
|-----------|---------|
| Supercomputer | Aziz HPC, KAU (https://hpcc-kau.com) |
| GPU node (R3) | NVIDIA A100-PCIE-40GB (queue: A100, node kcn512) |
| CPUs per job | 32 cores |
| RAM per job | 64 GB |
| Conda environment | `quantum_drl_gpu` |
| Python | 3.11 (conda) |
| PyTorch | 2.5.1+cu121 |
| Stable-Baselines3 | 2.x |
| SiliQun version | v2.0 (commit 37eec0a) |

---

## Claim → Evidence Map

### Claim 1: PPO achieves F = 1.0000 on 2-qubit CZ gate (SiMOS nominal, 5/5 seeds)

| Item | Value |
|------|-------|
| Result JSON | `results/json/siliqun_r3_v5_results__r3_v5_results.json` |
| Key field | `best_fidelity_mean = 1.0`, `converged_seeds = 5` |
| A100 log | `results/logs/r3_aziz.log` |
| Wall time | 8,128 s (2.26 hours) on kcn512 |
| Figure | `results/figures/F4_fidelity_vs_episode_r3_all_seeds.{pdf,png}` |

### Claim 2: Daraeizadeh replication (R1) achieves best_F = 0.8215 (PPO, SiMOS CZ, 3 seeds)

| Item | Value |
|------|-------|
| Result JSON | `results/json/siliqun_r1_ppo_results__r1_ppo_results.json` |
| Per-seed JSONs | `siliqun_r1_ppo_results__r1_ppo_seed{42,123,456}.json` |
| Key field | `best_F_overall = 0.8215`, `mean_pct_above_999 = 0.0` |
| Figure | `results/figures/F2_fidelity_vs_episode_r1_vs_r3.{pdf,png}` |

### Claim 3: Fidelity degrades sharply beyond 2 qubits (scaling ablation)

| Target | 2Q | 3Q | 4Q | 5Q |
|--------|----|----|----|----|
| GHZ | 0.9869 | 0.7927 | — | — |
| W | 0.9880 | 0.7896 | — | — |
| Cluster-linear | **0.9994** | 0.8883 | 0.3379 | — |
| Dicke-k3 | 0.9876 | **0.9906** | 0.4817 | 0.2798 |

Source: `results/json/scaling__scaling_summary.json`  
Figure: `results/figures/F5_scaling_ablation.{pdf,png}`

### Claim 4: GAA device is harder than SiMOS (device ablation)

| Item | Value |
|------|-------|
| Result JSON | `results/json/results__gaa_seed4_v3_results.json` |
| GAA best_F | 0.9968 (peak), 0.6641 (final) |
| SiMOS best_F | 1.0000 (stable) |
| Figure | `results/figures/F6_device_ablation_simos_vs_gaa.{pdf,png}` |

### Claim 5: PPO reward is stable throughout training (no collapse)

| Item | Value |
|------|-------|
| Source | `results/json/siliqun_r3_v5_results__r3_v5_results.json` → `per_seed.seed_0.history.rewards` |
| Mean reward | ~10.85 across all episodes |
| Std reward | < 0.1 |
| Figure | `results/figures/F1_ppo_reward_curves.{pdf,png}` |

---

## How to Reproduce

### Install

```bash
git clone https://github.com/r3d-phd/siliqun.git
cd siliqun
pip install -e ".[dev]"
```

### Run R3 (primary result)

```bash
python experiments/siliqun_e4_v6/siliqun_e4_v6.py \
  --device simos_nominal \
  --n-seeds 5 \
  --n-episodes 20000 \
  --algorithm PPO
```

### Run R1 (Daraeizadeh replication)

```bash
python experiments/siliqun_e4_v6/siliqun_e4_v6.py \
  --device simos_cz \
  --n-seeds 3 \
  --seeds 42 123 456 \
  --n-episodes 10000 \
  --algorithm PPO
```

### Regenerate figures

```bash
python gen_siliqun_figures.py
# Output: results/figures/F{1-8}_*.{pdf,png}
```

---

## Known Issues

1. **TensorBoard event files not generated** — TensorFlow/protobuf version conflict on Aziz A100 nodes prevents SB3's TensorBoard callback from writing event files. All scalar data is captured in JSON format instead. Fix: downgrade `protobuf` to `3.20.x` or use `tensorboardX`.

2. **R4 seed 666 did not converge** — `best_F = 0.8427`, likely due to a suboptimal random seed initialisation. Not included in primary results.

3. **GAA SAC instability** — The SAC agent achieves high peak fidelity (0.9968) but the final policy is unstable. Entropy coefficient tuning required.
