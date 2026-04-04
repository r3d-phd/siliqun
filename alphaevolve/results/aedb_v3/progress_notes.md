# AEDB v3 Experiment Progress

## Launch Time: 2026-04-04 15:40:32 UTC

## Phase 1: DEHB Meta-Level Optimization
- **Status**: COMPLETE (14.2s, 30 evaluations)
- **Best fidelity**: 0.9911
- **DEHB-learned configs**:
  - Noise: amp=0.0176, spacing=164.1nm, corr_len=100.8nm
  - LLM: temp=1.06, top_p=0.84, mut_rate=0.49
  - BRFD: reward_lr=0.04982, policy_lr=0.0060, outer=15, hidden=112
  - Fitness: samples=186, max_seq=139

## Phase 2: Seed Evaluation (DEHB configs + BRFD)
- **Status**: IN PROGRESS
- DEHB strategy injected: fid=0.4040, combined=0.3278
- BRFD training started: 15 outer steps

## Key Observations
- DEHB found very low noise amplitude (0.0176) as optimal - makes sense as lower noise = higher fidelity
- LLM temperature slightly above 1.0 (1.06) - encouraging exploration
- BRFD configured with 15 outer steps (DEHB learned this, more than our minimum of 10)
