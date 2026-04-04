# AEDB v3 Bottleneck Analysis

## Current Architecture Review

### Fidelity Pipeline
1. LLM generates `generate_gate_sequence()` code
2. Fitness evaluator applies gates + noise after EACH gate
3. Noise: TLF-correlated Z-rotations (Gaussian, spatially correlated)
4. Fidelity = |<target|final_state>|^2 averaged over noise realizations
5. Combined score = 0.70*fidelity + 0.15*gate_efficiency + 0.15*corr_use

### Identified Bottlenecks

#### B1: Noise applied AFTER every gate (line 558-560 of fitness.py)
- Every gate gets a fresh noise sample: `state = noise_model.apply_noise(state)`
- More gates = more noise accumulation
- The best strategy uses 5 gates for GHZ (3 Rz + 2 CNOT) → 5 noise injections
- Standard GHZ uses 3 gates (H + 2 CNOT) → 3 noise injections
- The correlation-aware strategy ADDS gates (Rz rotations) which adds MORE noise

#### B2: BRFD uses tabular REINFORCE (SimplePolicy)
- Only a tabular softmax over (step, action) pairs
- No state-dependent policy (doesn't see the quantum state)
- Very limited expressiveness → BRFD fidelity stuck at 0.27
- The BRFD inner loop is essentially random search with a learned reward

#### B3: LLM generates static gate sequences (no adaptive feedback)
- The evolved function takes (target, n_qubits, correlation, spacing, corr_length)
- It does NOT see the actual noise realization
- Cannot adapt gates based on measured/estimated noise
- This is open-loop control, not closed-loop

#### B4: Noise amplitude 0.5 rad/gate is VERY high
- DEHB learned to use 0.0176 internally but the experiment runs at 0.5
- At 0.5 rad/gate, even 3 gates accumulate ~1.5 rad total dephasing
- This is a fundamental physics limit: no static sequence can achieve >99% at this noise level

#### B5: No dynamical decoupling in the evolved strategies
- The seed strategies include echo/DD but the LLM didn't evolve them further
- DD sequences (CPMG, XY-4, etc.) can suppress low-frequency noise
- The DEHB learned dd_sequence='cpmg' and dd_repetitions=4 but these aren't used in the evolved code

#### B6: No composite pulse sequences
- BB1, CORPSE, Solovay-Kitaev decomposition not in the gate library
- These can provide error-robust implementations of standard gates

#### B7: Single noise realization per gate (no time-correlated noise)
- Current model: independent noise sample per gate
- Real TLF noise has temporal correlations (1/f spectrum)
- DD effectiveness depends on temporal correlation structure
