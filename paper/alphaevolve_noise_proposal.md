# AlphaEvolve + SiliQun: Evolving Noise Mitigation Strategies for Silicon Spin Qubits

## The Core Idea

AlphaEvolve discovers novel algorithms by evolving Python source code using LLMs as intelligent mutation operators. SiliQun provides a fast, differentiable simulator for silicon spin qubits with realistic TLF-correlated noise. The combination: **use AlphaEvolve to discover novel noise mitigation strategies, with SiliQun as the automated fitness evaluator.**

## What Problem Are We Solving?

Correlated TLF charge noise in silicon spin qubits causes:
1. Spatially correlated detuning errors across neighboring qubits
2. Time-correlated drift that accumulates over gate sequences
3. Leakage out of the DFS-encoded subspace

Current mitigation approaches are all **human-designed**:
- DFS encoding (passive, exploits symmetry)
- Dynamical decoupling (periodic pulse sequences)
- Composite pulse sequences (fixed error-canceling designs)
- Optimal control theory (gradient-based pulse shaping)

**None of these were discovered by automated search.** AlphaEvolve could find non-intuitive strategies that humans wouldn't design.

## Three Concrete Formulations

### Formulation 1: Evolve Gate Pulse Sequences (Most Feasible)

**Search space:** A Python function `generate_gate_sequence(target_gate, noise_params) -> List[Gate]` that takes a target unitary and noise parameters, and returns a sequence of primitive gates that implements the target while mitigating correlated noise.

**Seed program:** Standard decomposition (e.g., Rx-Ry-Rx for arbitrary single-qubit gate, or CNOT decomposition for two-qubit gates).

**Fitness function:** Run the generated gate sequence in SiliQun with TLF-correlated noise enabled. Fitness = average gate fidelity over 1000 noise realizations.

**Why it works:**
- SiliQun evaluates one gate sequence in ~1 ms (3x3 grid) to ~430 ms (5x5 grid)
- AlphaEvolve needs ~1000 evaluations per generation
- Total: ~1 second to ~7 minutes per generation on a single A100
- Feasible to run 100+ generations in a few hours

**What it might discover:**
- Non-obvious gate orderings that exploit TLF correlation structure
- Adaptive pulse timings that cancel correlated errors
- Novel composite sequences tailored to the DFS encoding

### Formulation 2: Evolve DRL Reward Functions (Medium Feasibility)

**Search space:** A Python function `compute_reward(state, action, next_state, fidelity, leakage) -> float` that defines the reward signal for the DRL agent.

**Seed program:** Current SiliQun reward: `reward = fidelity - lambda * leakage`

**Fitness function:** Train a DRL agent for N episodes using the evolved reward function, then evaluate final gate fidelity under correlated noise. Fitness = final fidelity after training.

**Why it works:**
- Reward shaping is known to dramatically affect DRL convergence
- The current reward function is hand-designed and may be suboptimal
- AlphaEvolve could discover reward functions that implicitly encode noise structure

**Challenge:** Each fitness evaluation requires training a DRL agent (~50 episodes for 2-qubit), so it's slower. But for small systems (2-3 encoded qubits), this is feasible.

### Formulation 3: Evolve Noise-Aware Encoding Schemes (Most Ambitious)

**Search space:** A Python function `encode_logical_state(physical_spins, noise_correlation_matrix) -> encoded_state` that defines how to map logical information into the physical spin space, potentially discovering encodings beyond the standard DFS.

**Seed program:** Standard DFS encoding (3 spins -> 1 encoded qubit).

**Fitness function:** Measure how well the encoding preserves quantum information under realistic TLF noise over a fixed time window. Fitness = average state fidelity.

**Why it's exciting:** Could discover entirely new noise-resilient encodings that exploit the specific spatial correlation structure of TLF noise (l_c = 81 nm). The standard DFS encoding only protects against *global* collective dephasing; a TLF-aware encoding could protect against the *spatially structured* correlations.

**Challenge:** The search space is very large and the physics constraints are subtle. Would need careful design of the code skeleton to ensure physical validity.

## Recommended Approach: Start with Formulation 1

**Why:**
- Fastest evaluation loop (~1 ms per fitness evaluation on 3x3 grid)
- Clearest fitness metric (gate fidelity)
- Most constrained search space (sequence of discrete gates)
- Results are immediately interpretable and publishable
- Can be done WITHOUT access to Google's AlphaEvolve infrastructure

**Implementation plan:**
1. Define the evolvable code skeleton (gate sequence generator)
2. Implement the fitness evaluator using SiliQun's noise model
3. Use our own LLM (Gemini API or OpenAI API) for mutations
4. Run the evolutionary loop on the Aziz A100 GPU

## Can We Actually Run This?

**Yes, but with a key distinction.** The original AlphaEvolve (Novikov et al., 2025) is Google's internal infrastructure. However, the paper we're reading (Li et al., 2026) shows that the *methodology* can be replicated using:
- Any capable LLM (Gemini 2.5 Pro, GPT-4, etc.) for code mutations
- A custom evolutionary loop (population management, selection, etc.)
- An automated evaluator (SiliQun)

We have all three ingredients:
- Gemini API key (GEMINI_API_KEY) and OpenAI API key (OPENAI_API_KEY)
- Python for the evolutionary loop
- SiliQun for evaluation

**Estimated compute cost:**
- Formulation 1 on 3x3 grid: ~1000 evaluations/generation × 1 ms = 1 second/generation
- 100 generations × 50 population = 5000 LLM calls
- At ~$0.01/call (Gemini 2.5 Flash) = ~$50 total
- Wall time: ~2-4 hours including LLM latency

## Publication Potential

This could be a **standalone paper** or a **thesis chapter**:

**Title idea:** "LLM-Evolved Noise Mitigation Strategies for Silicon Spin Qubits"

**Novelty:**
1. First application of LLM-driven code evolution to quantum noise mitigation
2. First use of a quantum digital twin (SiliQun) as AlphaEvolve's fitness evaluator
3. Potential discovery of non-intuitive noise-resilient gate sequences

**Venue:** Physical Review Letters (if the discovered strategy is physically insightful), or Nature Machine Intelligence (if the methodology is the main contribution)

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM generates invalid quantum operations | Validate all generated sequences before evaluation |
| Search space too large for convergence | Start with highly constrained skeleton (e.g., fixed gate count) |
| Discovered strategy is trivially equivalent to known techniques | Compare against all known composite pulse sequences |
| Compute cost exceeds budget | Use 3x3 grid (1 ms/eval) for discovery, validate on 5x5 |
| Results not physically interpretable | Analyze discovered sequences with XAI-Nexus tools |
