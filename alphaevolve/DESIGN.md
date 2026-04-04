# AlphaEvolve Full Architecture Design

## Module Structure

```
alphaevolve/
├── alpha_evolve.py          # Main AlphaEvolve engine (replaces evolve_v2.py)
├── program_database.py      # MAP-Elites + island-based program database
├── prompt_sampler.py        # Rich prompt construction with inspirations
├── llm_ensemble.py          # Multi-model LLM ensemble (Ollama/Gemini/OpenRouter)
├── evaluation_cascade.py    # Multi-stage evaluation with early pruning
├── aedb_orchestrator.py     # AEDB integration (AlphaEvolve + DEHB + BRFD)
├── fitness.py               # Quantum fidelity evaluator (existing)
├── dehb_optimizer.py        # DEHB hyperparameter optimizer (existing)
├── brfd_reward.py           # BRFD reward function learner (existing)
└── gpu_server.py            # Local GPU inference server (existing)
```

## Component Design

### 1. ProgramDatabase (program_database.py)
- MAP-Elites grid: feature dimensions = [gate_count, correlation_usage, fidelity_tier]
- Island model: N islands with periodic migration
- Each cell stores top-K programs ranked by fitness
- Sampling: weighted by fitness with diversity bonus
- Tracks: code, fitness_dict (multi-metric), generation, parent_id, model

### 2. PromptSampler (prompt_sampler.py)
- Samples parent program from database
- Samples K inspiration programs (diverse, high-fitness)
- Includes rendered evaluation results (scores + outputs)
- Stochastic formatting: randomized instruction variants
- Supports SEARCH/REPLACE diff format for mutations
- Optional: meta-prompt evolution (co-evolved prompts)

### 3. LLMEnsemble (llm_ensemble.py)
- Manages multiple LLM backends: Ollama (local GPU), Gemini, OpenRouter
- Flash model (Ollama/Gemini Flash): high throughput, lower quality
- Pro model (Gemini Pro/large OpenRouter): lower throughput, higher quality
- Async-ready: can dispatch multiple calls concurrently
- Rate limit tracking per backend
- Model rotation on failure

### 4. EvaluationCascade (evaluation_cascade.py)
- Stage 1: Quick syntax/compile check (0.01s)
- Stage 2: Small-scale fidelity (10 samples, 0.1s)
- Stage 3: Medium-scale fidelity (50 samples, 0.5s)
- Stage 4: Full-scale fidelity (200 samples, 2s)
- Early termination: skip later stages if score < threshold
- Multi-metric: base_fidelity, gate_count_penalty, correlation_exploitation

### 5. AlphaEvolve (alpha_evolve.py)
- Async controller loop (asyncio)
- Concurrent LLM sampling + evaluation
- Evolution: tournament selection from ProgramDatabase
- Diff-based mutations (SEARCH/REPLACE) for refinement
- Full-function generation for exploration
- Configurable: population_size, n_generations, ensemble_ratio
