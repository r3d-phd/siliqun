# AEDB: AlphaEvolve-DEHB-BRFD Integrated Architecture for Quantum Noise Mitigation

## 1. Vision

The core insight is that the correlated noise mitigation problem in silicon spin qubits has **three distinct optimization layers**, each requiring a fundamentally different search strategy:

| Layer | What is Optimized | Method | Why This Method |
|-------|------------------|--------|-----------------|
| **Structure** | Gate sequence logic (Python code) | AlphaEvolve (LLM-driven code evolution) | Only LLMs can search the space of algorithms |
| **Parameters** | All continuous/discrete hyperparameters | DEHB (evolutionary multi-fidelity HPO) | Handles mixed dimensions, multi-fidelity evaluation |
| **Reward** | DRL reward function | BRFD (bilevel reward function discovery) | Discovers dense, informative rewards from sparse signals |

No single method can handle all three layers. AlphaEvolve cannot efficiently tune continuous parameters. DEHB cannot invent new algorithms. BRFD cannot optimize gate sequences. Together, they form a complete optimization stack.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AEDB Orchestrator                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  AlphaEvolve  │  │     DEHB     │  │     BRFD     │       │
│  │  (Structure)  │  │  (Parameters)│  │   (Reward)   │       │
│  │              │  │              │  │              │       │
│  │ LLM mutates  │  │ DE+HB tunes  │  │ Meta-gradient│       │
│  │ gate sequence│  │ all hyperparams│ │ discovers R* │       │
│  │ Python code  │  │ across budgets│  │ for DRL agent│       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └────────┬────────┴──────────┬───────┘               │
│                  │                   │                        │
│         ┌────────▼───────────────────▼────────┐              │
│         │         SiliQun Simulator            │              │
│         │   (TLF-correlated noise model)       │              │
│         │   3x3 → 4x4 → 5x5 (multi-fidelity)  │              │
│         └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 3. How the Three Components Interact

### 3.1 Outer Loop: AlphaEvolve (every ~10 minutes)

AlphaEvolve maintains a population of **gate sequence generators** — Python functions that take (n_qubits, target_state, noise_params) and return a list of gates. The LLM mutates the code to discover new algorithmic structures: novel echo patterns, correlation-aware gate orderings, adaptive pulse sequences.

Each candidate gate sequence generator is evaluated by the **inner loops** (DEHB + BRFD) before its fitness is determined. This means AlphaEvolve doesn't just test one parameterization of each algorithm — it tests the **best possible parameterization** found by DEHB.

### 3.2 Middle Loop: DEHB (every ~1 minute)

For each candidate gate sequence from AlphaEvolve, DEHB optimizes all continuous and discrete parameters:

**Parameters DEHB tunes:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| noise_amplitude | continuous | [0.01, 1.0] | Charge noise strength (rad/gate) |
| correlation_length | continuous | [20, 200] nm | TLF spatial correlation |
| gate_angle_1..k | continuous | [0, 2π] | Rotation angles in the sequence |
| echo_spacing | continuous | [0.1, 10] ns | Timing between echo pulses |
| n_repetitions | integer | [1, 20] | Number of DD repetitions |
| drl_learning_rate | continuous | [1e-5, 1e-2] | DRL agent learning rate |
| drl_gamma | continuous | [0.9, 0.999] | Discount factor |
| drl_entropy_coeff | continuous | [0.001, 0.1] | Exploration coefficient |
| reward_lr | continuous | [1e-5, 1e-3] | BRFD reward network LR |
| reward_hidden_dim | integer | [32, 256] | BRFD network width |

**Multi-fidelity budget levels (the key DEHB advantage):**

| Budget Level | Grid Size | Time/Eval | Purpose |
|-------------|-----------|-----------|---------|
| b_min = 1 | 3×3 (9 qubits) | ~1 ms | Cheap screening |
| b_mid = 3 | 4×4 (16 qubits) | ~5 ms | Intermediate validation |
| b_max = 9 | 5×5 (25 qubits) | ~430 ms | Full evaluation |

This is a **natural multi-fidelity hierarchy** — small grids are cheap but noisy proxies for large grids. DEHB exploits this perfectly: it evaluates 27 configurations on 3×3, promotes the top 9 to 4×4, and the top 3 to 5×5. This is ~100x more efficient than evaluating everything on 5×5.

### 3.3 Inner Loop: BRFD (every ~30 seconds)

For each (gate_sequence, hyperparameters) pair, BRFD discovers the optimal reward function for training the DRL agent to execute that gate sequence under noise.

**Why BRFD is critical here:** The current SiliQun reward function is hand-designed: `R = fidelity - λ * leakage`. But this may not be optimal. BRFD can discover:

- Non-linear reward shaping: `R = f(fidelity, leakage, entropy, correlator_values)`
- State-dependent rewards: Different rewards for different stages of the gate sequence
- Dense intermediate signals: Rewards for partial entanglement, not just final fidelity
- Noise-aware rewards: Rewards that account for the current noise realization

**BRFD mechanism:**
- Reward function R_ω parameterized as a small neural network (2 hidden layers, 64 units)
- Inner loop: DRL agent trains for N episodes under R_ω
- Outer loop: Meta-gradient updates ω to minimize regret (gap between true optimal and learned policy)
- Advantage product: A^R(s,a) × A^π(s,a) guides the reward update

## 4. Information Flow

```
AlphaEvolve generates candidate gate sequence code
    │
    ▼
DEHB receives the code + parameter search space
    │
    ├── Budget 1 (3×3): Evaluate 27 configs
    │   └── For each config: BRFD discovers reward → DRL trains → fidelity
    │
    ├── Budget 3 (4×4): Evaluate top 9
    │   └── For each config: BRFD discovers reward → DRL trains → fidelity
    │
    └── Budget 9 (5×5): Evaluate top 3
        └── For each config: BRFD discovers reward → DRL trains → fidelity
            │
            ▼
        Best (config, reward, fidelity) returned to AlphaEvolve as fitness
            │
            ▼
AlphaEvolve uses fitness for selection + LLM mutation → next generation
```

## 5. Why This Could Discover Something Novel

The key hypothesis: **correlated noise creates structure that can be exploited, but the exploitation strategy is too complex for human intuition.**

Consider what the system searches over:
1. **AlphaEvolve** can discover that interleaving X and Z rotations in a specific pattern cancels correlated phase errors — something no textbook describes
2. **DEHB** can find that the optimal echo spacing is not uniform but follows a specific function of the correlation length — a relationship too complex for analytical derivation
3. **BRFD** can discover that rewarding the DRL agent for maintaining a specific correlator value (not just fidelity) leads to better final states — a reward signal no human would design

The combination searches a space that no individual method can explore.

## 6. Estimated Computational Cost

| Component | Calls per Run | Cost per Call | Total Cost |
|-----------|--------------|--------------|------------|
| AlphaEvolve (LLM) | 200 | $0.01 (Gemini Flash) | $2 |
| DEHB evaluations | 2000 | ~1 ms (3×3) to 430 ms (5×5) | ~10 min GPU |
| BRFD training | 200 | ~30 s (1000 episodes) | ~100 min GPU |
| **Total** | | | **~$2 + 2 hrs A100** |

This is remarkably cheap for a system that searches over algorithms, parameters, AND reward functions simultaneously.

## 7. Implementation Plan

### Phase 1: DEHB Integration (1 day)
- Install DEHB (`pip install dehb`)
- Define the hyperparameter search space (ConfigSpace)
- Wrap SiliQun fitness evaluator as DEHB objective function
- Define multi-fidelity budget mapping (grid size)
- Test: DEHB alone should find better noise parameters than our defaults

### Phase 2: BRFD Integration (2 days)
- Implement reward network R_ω as a small MLP
- Implement meta-gradient computation (advantage product)
- Modify SiliQun's gym environment to accept learned reward functions
- Test: BRFD alone should discover a reward that outperforms hand-designed R

### Phase 3: AlphaEvolve v2 (1 day)
- Upgrade the existing AlphaEvolve PoC with harder noise settings
- Increase noise to 0.3 rad/gate (baseline ~92% fidelity)
- Add 1/f temporal correlations
- Target 4-qubit GHZ state (more complex than Bell)
- Fix DeepSeek indentation issues

### Phase 4: AEDB Integration (2 days)
- Build the orchestrator that coordinates all three loops
- AlphaEvolve outer → DEHB middle → BRFD inner
- Implement checkpointing and result logging
- Run the full integrated experiment on Aziz A100

### Phase 5: Analysis and Paper (2 days)
- Analyze discovered gate sequences
- Compare with textbook strategies (DD, composite pulses)
- Write up as a standalone paper or thesis chapter
- Visualize the evolutionary trajectory

## 8. Publication Potential

**Title:** "LLM-Evolved Noise Mitigation for Silicon Spin Qubits: A Tri-Level Optimization Approach"

**Target venues (ranked):**
1. **Nature Machine Intelligence** — Novel AI methodology applied to quantum hardware
2. **Physical Review Letters** — If the discovered strategy has deep physical insight
3. **NeurIPS / ICML** — If framed as a general framework for scientific discovery
4. **Quantum Science and Technology** — If framed as a quantum engineering contribution

**Novelty claims:**
- First application of LLM-driven code evolution to quantum noise mitigation
- First integration of AlphaEvolve + DEHB + BRFD as a tri-level optimization stack
- First use of multi-fidelity HPO with quantum simulator grid sizes as budget proxy
- Discovery of non-obvious noise mitigation strategies for correlated TLF noise

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LLMs generate mostly invalid code | Medium | Error feedback loop, constrained templates |
| DEHB converges to local optimum | Low | Multiple restarts, global population pool |
| BRFD reward overfits to noise realization | Medium | Evaluate across multiple noise seeds |
| No improvement over textbook strategies | Medium | Increase noise severity, target harder circuits |
| Computational cost exceeds budget | Low | Multi-fidelity reduces cost by ~100x |
| Results not reproducible | Low | Fix random seeds, log all LLM prompts/responses |
