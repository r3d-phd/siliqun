# Session Memory — May 1, 2026

## Key Outcomes

### 1. ReadTheDocs `latest` Build — FIXED ✅
- Root cause: v1.0.0 tag created before `.readthedocs.yaml` was added
- Fix: Deleted old tag, recreated pointing to commit `9e2ddcc`
- Build #32502529 succeeded (confirmed from screenshot)

### 2. Zenodo DOI — PUBLISHED ✅
- DOI: `10.5281/zenodo.19890642`
- URL: https://doi.org/10.5281/zenodo.19890642
- Status: Published, 1 file uploaded

### 3. SiliQun Paper Updated ✅
- File: `paper/softwarex_siliqun_v3.tex`
- C1 (code version): `v1.0.0`
- C3 (reproducible capsule): `https://doi.org/10.5281/zenodo.19890642`
- Pushed to GitHub: commit `b87bfab`

### 4. QUASAR v3 Root Cause Identified ✅
- All 12/15 seeds completed (3 PBT seeds missing due to DEHB `verbose` bug)
- All `eval_fidelity = 0.5000` throughout training — SAC stuck at zero-action local min
- Zero actions → fidelity = 0.5000 (initial state fidelity for 3-qubit GHZ)
- `max_fidelity` near 1.0 in JSON = per-episode training fidelity (stochastic), not eval

### 5. QUASAR v4 Submitted ✅
- Job ID: `174020.khead2` (A100 queue, 24h walltime)
- Status: Running
- Fixes applied:
  - `alpha = 1.0` (was 0.2) — stronger initial entropy
  - `WARMUP_STEPS = 10,000` (was 1,000) — more random exploration
  - `TOTAL_STEPS = 500,000` (was 100,000) — 5× training budget
  - `verbose=False` in DEHB (was True — caused crash)
  - Results dir: `~/quasar_v4_results/`

## Action Items for Next Session
1. Check job 174020 status (~24h from now)
2. Trigger ReadTheDocs `stable` build manually
3. Analyze v4 results when job completes

## Critical Lessons Learned
- SAC zero-action local min: `tanh(0) = 0` → agent outputs zero actions → 0.50 fidelity
- Fix: High initial alpha (1.0) forces entropy → exploration → escape local min
- Always verify eval_fidelity improves during training, not just max_fidelity
- DEHB `verbose` parameter removed in newer versions — always use `verbose=False`
