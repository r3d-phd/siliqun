# DeepSeek Evaluation Feedback on SiliQun SoftwareX Paper

## Verdict: Accept with Minor Revisions

## Strengths Identified
1. Clear, significant problem addressed
2. Novel technical approach (DFS projection)
3. Impressive performance gains (30-93x)
4. High usability and integration (Gymnasium, device profiles)
5. Honest comparison with existing tools
6. Reproducibility and open science

## Weaknesses & Required Fixes

### CRITICAL (Must Fix)
1. **Missing validation against ground truth** — No comparison showing DFS-projected dynamics match full physical-space simulations for small qubit counts (2-3 logical qubits where 2^3n is tractable)
2. **Limited DRL results** — No actual DRL training results shown (learning curves, policies, convergence). Code snippet shows PPO agent but no output.
3. **Leakage tracking is diagnostic only** — Does not quantify regimes where leakage becomes problematic

### IMPORTANT (Should Fix)
4. **Benchmarking details incomplete** — CPU baseline not described (single-core? multi-threaded? what CPU model?)
5. **Limited novelty in MPS engine** — MPS backend not benchmarked or compared to existing tensor network libraries
6. **Single-GPU focus** — No discussion of multi-GPU scaling for larger grids

### MINOR (Easy Fixes)
7. **Author affiliation discrepancy** — Student email with faculty affiliation
8. **Fig. 4 typo** — Referenced as "igure 4"
9. **Reference [8] incomplete** — Schaal et al. missing volume/page
10. **Abstract "exact simulations"** — Nuance about "within logical subspace" should appear earlier
