# AlphaEvolve Noise Mitigation Discovery - Experiment Analysis

## Experiment Summary

| Parameter | Value |
|-----------|-------|
| Total time | 2,738 seconds (~46 minutes) |
| LLM calls | 150 (budget exhausted) |
| Total tokens | 353,409 |
| Generations completed | 14 (of 20 planned) |
| Models used | DeepSeek Chat, Gemini 2.5 Flash, Claude Sonnet 4 |
| Noise amplitude | 0.05 rad/gate |
| NN correlation | 0.264 (RIKEN 5Q calibration) |

## Key Result

**The standard textbook gate sequence (H + CNOT for Bell, CNOT for CNOT) was NOT beaten.**

- Best fitness: **0.9974** (seed standard strategy)
- Best raw fidelity: **0.9989**
- No LLM-evolved variant exceeded this across 14 generations

## Why the LLMs Failed to Beat the Baseline

### 1. The Problem Was Too Easy (Noise Too Mild)

At 0.05 rad/gate noise amplitude, the standard 2-gate Bell sequence already achieves 99.74% fidelity. There is only **0.26% room for improvement**. This is within the statistical noise of the evaluator (100 noise realizations). The LLMs correctly identified that the shortest sequence is optimal when noise is mild - every additional gate adds more noise exposure than it can cancel.

### 2. Convergence to the Optimal Solution

By Generation 11, the population achieved 100% valid rate with mean fitness 0.896 - the LLMs were converging to the standard strategy. This is actually the CORRECT behavior: when the problem is easy, evolution should converge to the known optimum.

### 3. DeepSeek Indentation Issues

DeepSeek Chat had a persistent problem generating Python code with correct indentation. Approximately 70% of DeepSeek outputs were invalid due to mixed tabs/spaces or incorrect indentation levels. This significantly reduced the effective population diversity.

## Per-Model Performance

| Model | Valid Rate | Best Fitness | Typical Strategy |
|-------|-----------|-------------|-----------------|
| Claude Sonnet 4 | ~75% | 0.9974 | Converges to standard; creative variants use echo pulses |
| Gemini 2.5 Flash | ~65% | 0.9974 | Converges to standard; good at crossover |
| DeepSeek Chat | ~30% | 0.9974 | Persistent indentation errors; when valid, often creative but low-fidelity |

## Interesting Evolved Variants (Non-Standard)

Several LLM-generated strategies showed novel approaches, even though they didn't beat the baseline:

1. **Echo-wrapped Bell** (Claude, Gen 5): Added Rz echo pulses around the CNOT to refocus correlated Z-noise. Achieved 0.9959 - slightly worse due to extra gate noise.

2. **Correlation-aware CNOT** (DeepSeek, Gen 10): Used the noise_correlation parameter to conditionally add echo pulses only when correlation > 0.5. Achieved 0.9929.

3. **sqrt(SWAP) decomposition** (Gemini, Gen 8): Decomposed CNOT into native sqrt(SWAP) gates. Achieved 0.9918 - worse because 6 gates vs 1.

## Diagnosis: What Needs to Change

### For the Noise Problem to Be Interesting:
1. **Increase noise amplitude to 0.2-0.5 rad/gate** - This would drop the baseline fidelity to ~0.90-0.95, creating room for improvement
2. **Add time-correlated noise** (1/f spectrum) - Static noise is too simple; temporal correlations create opportunities for dynamical decoupling
3. **Target longer circuits** (GHZ states, 4+ qubits) - More gates = more noise accumulation = more room for mitigation
4. **Add leakage penalty** - Currently only measuring fidelity within the encoded subspace; adding leakage tracking would create a multi-objective problem

### For the Evolution to Be More Effective:
1. **Fix DeepSeek indentation** - Pre-process outputs to normalize whitespace
2. **Increase population to 30-50** - More diversity per generation
3. **Add structured mutations** - Instead of free-form code rewriting, provide specific mutation operators (insert gate, delete gate, swap gate order, change angle)
4. **Use Gemini 2.5 Pro** instead of Flash for better code reasoning
5. **Add a "physicist advisor" prompt** - Include physics insights about dynamical decoupling, spin echo, and Carr-Purcell sequences

## Conclusion

This PoC successfully demonstrated that:
1. SiliQun can serve as a fitness evaluator for LLM-driven code evolution
2. The evolutionary loop with OpenRouter multi-model support works end-to-end
3. The LLMs correctly converge to the known optimum when the problem is easy

**The next step is to make the problem HARDER** so there's room for the LLMs to discover non-obvious solutions. This requires:
- Higher noise (0.2+ rad/gate)
- Temporal correlations (1/f noise spectrum)
- Longer target circuits (4-qubit GHZ, multi-gate sequences)
- Multi-objective optimization (fidelity + leakage + gate count)

This harder problem formulation would be the basis for a potential publication.
