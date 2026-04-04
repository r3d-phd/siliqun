# AEDB v2 Experiment Report: AlphaEvolve + DEHB + BRFD

## Executive Summary

The AEDB v2 system successfully integrates a faithful AlphaEvolve implementation with DEHB hyperparameter optimization and BRFD reward learning for automated discovery of quantum noise mitigation strategies. Using a locally deployed Qwen2.5-Coder-7B model on an RTX 2070 GPU, the system achieved **93% valid strategy generation rate** with **zero rate limits** and completed a full 10-generation evolutionary run in **11.8 minutes**.

## Architecture

The AEDB v2 system implements the following AlphaEvolve components faithfully from the DeepMind paper:

| Component | Implementation | Status |
|-----------|---------------|--------|
| **Program Database** | MAP-Elites with island-based populations | 7 behavioral cells discovered |
| **Prompt Sampler** | Rich context with parent + inspirations + eval feedback | Working, 7B-optimized prompts |
| **LLM Ensemble** | Ollama GPU (Qwen2.5-Coder-7B on RTX 2070) | 34.8 tok/s, zero rate limits |
| **Evaluation Cascade** | 3-stage filtering (quick → medium → full) | 93% pass rate |
| **Multi-Metric Scoring** | combined, base_fidelity, gate_efficiency, correlation_use | 4 metrics tracked |

## Infrastructure

### GPU Inference Server

- **Hardware**: NVIDIA GeForce RTX 2070 Mobile (8GB VRAM)
- **Model**: Qwen2.5-Coder-7B-Instruct (Q4_K_M, 4.7GB)
- **Framework**: llama-cpp-python with CUDA 12.4
- **Speed**: 34.8 tokens/second (GPU-accelerated)
- **Connection**: ngrok TCP tunnel → SSH port forwarding → sandbox
- **Chat Format**: ChatML (proper `<|im_start|>` template)

### Previous Attempts and Lessons Learned

| Run | Backend | Issue | Valid Rate | Time |
|-----|---------|-------|-----------|------|
| Run 1-3 | OpenRouter free | 429 rate limits on all models | 0-50% | N/A (stalled) |
| Run 2 | Gemini 2.5 Flash | Rate limited after 15 calls | 66% | 38 min |
| Run 4 | Ollama CPU | 4.7 tok/s, too slow | 50% | 5 min |
| **Run 5** | **Ollama GPU** | **None - fully working** | **93%** | **11.8 min** |

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Noise Model | Coherent rotation, 0.5 rad/gate |
| Qubits | 2 |
| Target Gates | Bell state (H+CNOT), CNOT |
| Generations | 10 |
| Population per Gen | 3 |
| DEHB Evaluations | 8 per strategy |
| BRFD Outer Steps | 4 per strategy |
| Fitness Samples | 100 (quick), 500 (final) |

## Results

### Generation-by-Generation Progress

| Gen | Best Combined | Fidelity | Valid | DB Programs | MAP-Elites Cells | Time (s) |
|-----|--------------|----------|-------|-------------|-------------------|----------|
| 1 | 0.7684 | 0.9097 | 2/3 | 7 | 4 | 31.1 |
| 2 | 0.7684 | 0.9097 | 3/3 | 10 | 4 | 44.1 |
| 3 | 0.7484 | 0.8697 | 3/3 | 12 | 6 | 42.8 |
| 4 | 0.7684 | 0.9097 | 3/3 | 12 | 6 | 53.6 |
| 5 | 0.7684 | 0.9097 | 3/3 | 14 | 7 | 68.9 |
| 6 | 0.7684 | 0.9097 | 3/3 | 14 | 7 | 83.8 |
| 7 | 0.7684 | 0.9097 | 3/3 | 14 | 7 | 101.1 |
| 8 | 0.7684 | 0.9097 | 2/3 | 14 | 7 | 85.1 |
| 9 | 0.7684 | 0.9097 | 3/3 | 14 | 7 | 56.7 |
| 10 | 0.7684 | 0.9097 | 3/3 | 14 | 7 | 117.6 |

### Final Statistics

| Metric | Value |
|--------|-------|
| Best Combined Score | 0.7684 |
| Best Base Fidelity | 0.9097 |
| Final Fidelity (500 samples) | **0.9127** |
| Total Programs Explored | 14 |
| MAP-Elites Cells Discovered | 7 |
| Total LLM Calls | 30 |
| Total DEHB Evaluations | 216 |
| Total BRFD Training Steps | 108 |
| Total Wall Time | 707.6s (11.8 min) |
| Valid Strategy Rate | 28/30 = **93.3%** |

### Best Evolved Strategy

The best strategy discovered by AlphaEvolve uses **correlation-aware Ry pre/post-rotations** when nearest-neighbor correlation exceeds 0.7:

```python
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    gates = []
    if nn_correlation > 0.7:
        # Pre-rotation: correlation-dependent Ry gates
        for i in range(n_qubits):
            angle = np.pi * (i + 1) / n_qubits
            ry_angle = np.arccos(nn_correlation) * angle
            gates.append(Gate("ry", [i], {"angle": ry_angle}))
        
        if target_gate == "bell":
            gates.append(Gate("h", [0]))
            gates.append(Gate("cnot", [0, 1]))
            gates.append(Gate("h", [1]))
            gates.append(Gate("cnot", [1, 0]))
            gates.append(Gate("h", [0]))
        elif target_gate == "cnot":
            gates.append(Gate("cnot", [0, 1]))
        
        # Post-rotation: reverse correlation compensation
        for i in range(n_qubits-1, -1, -1):
            angle = np.pi * (i + 1) / n_qubits
            ry_angle = np.arccos(nn_correlation) * angle
            gates.append(Gate("ry", [i], {"angle": ry_angle}))
    else:
        # Standard textbook decomposition for low correlation
        if target_gate == "bell":
            gates.append(Gate("h", [0]))
            gates.append(Gate("cnot", [0, 1]))
        elif target_gate == "cnot":
            gates.append(Gate("cnot", [0, 1]))
    return gates
```

**Key insight**: The strategy branches on `nn_correlation` and applies pre/post Ry rotations with angles derived from `arccos(nn_correlation)`. For the Bell state, it uses a more complex decomposition (H-CNOT-H-CNOT-H) in the high-correlation regime. However, the additional gates in the high-correlation path add noise (0.5 rad each), so the standard path (low correlation) achieves the best fidelity.

## Analysis and Observations

### What Worked

1. **AlphaEvolve Architecture**: MAP-Elites successfully explored 7 distinct behavioral niches, diversifying the search beyond simple hill-climbing.
2. **Local GPU Inference**: Eliminated all rate-limiting issues. 34.8 tok/s with zero delays.
3. **Chat Template Fix**: Using `create_chat_completion` with ChatML format was critical for the Qwen2.5-Coder model to generate meaningful responses with long prompts.
4. **Evaluation Cascade**: 3-stage filtering efficiently rejected invalid strategies early.
5. **93% Valid Rate**: The simplified, 7B-friendly prompts produced compilable, executable code in nearly all cases.

### Limitations

1. **No Improvement Over Baseline**: The best strategy (0.9127 fidelity) matches but doesn't exceed the textbook H+CNOT approach. The 2-qubit Bell state is too simple for novel strategies to outperform.
2. **7B Model Quality**: While structurally valid, the 7B model's mutations are mostly trivial (adding correlation branches that don't help). A 14B+ model would generate more creative strategies.
3. **BRFD Underfitting**: Only 4 outer steps yields 0.3443 reward, limiting the combined score. More training would help.
4. **DEHB Plateau**: All DEHB runs converge to 0.8157, suggesting the search space is well-covered but the strategy itself is the bottleneck.

### Recommendations for Next Steps

1. **Scale to 3+ qubits** (GHZ state) where the search space is larger and novel strategies can truly outperform textbook approaches.
2. **Deploy Qwen2.5-Coder-14B** (Q4 fits in 8GB VRAM) for higher-quality code generation.
3. **Increase BRFD training** to 10+ outer steps for better reward learning.
4. **Run longer evolution** (20-50 generations) to explore more of the strategy space.
5. **Add noise model diversity** (depolarizing, amplitude damping) to test generalization.

## Files

| File | Description |
|------|-------------|
| `alpha_evolve.py` | Main AlphaEvolve engine |
| `program_database.py` | MAP-Elites + island model |
| `prompt_sampler.py` | Rich context prompt builder |
| `llm_ensemble.py` | Ollama-only GPU LLM backend |
| `evaluation_cascade.py` | Multi-stage evaluation pipeline |
| `aedb_orchestrator.py` | AEDB v2 orchestrator |
| `gpu_server.py` | GPU inference server for RTX 2070 |
| `fitness.py` | Quantum fidelity evaluator |
| `dehb_optimizer.py` | DEHB hyperparameter optimizer |
| `brfd_reward.py` | BRFD reward function learner |
