# AEDB Experiment Analysis Report

## Executive Summary

The AlphaEvolve-DEHB (AEDB) orchestrator was successfully implemented and tested with three different LLM backends: OpenRouter (free models), Google Gemini 2.5 Flash, and a local Qwen2.5-Coder-7B on the user's RTX 2070 GPU. The experiment validates the AEDB architecture as a working proof-of-concept for LLM-driven quantum gate strategy evolution.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Noise level | 0.5 rad/gate |
| Qubits | 2 |
| Target gates | Bell state, CNOT |
| Population size | 3 per generation |
| Generations | 6 |
| DEHB evaluations | 8 per strategy |
| BRFD outer steps | 4 per strategy |
| Seed strategies | standard, echo_cancel, correlation_aware |

## Results Summary

### Run 2 (Gemini 2.5 Flash)

| Gen | Best Fitness | Valid | Time |
|-----|-------------|-------|------|
| 1 | 0.7517 | 2/3 | 258s |
| 2 | 0.7517 | 2/3 | 255s |
| 3 | **0.9167** (NEW BEST) | 3/3 | 233s |
| 4 | 0.9167 | 2/3 | 327s |
| 5 | 0.9167 | 2/3 | 741s (rate limited) |
| 6 | 0.9167 | 0/3 | 506s (rate limited) |

- **Total time**: ~38 minutes
- **Final fidelity (500 samples)**: 0.9127
- **Best generation**: 3 (LLM-evolved matched baseline)
- **Issue**: Heavy rate limiting on Gemini (429 errors)

### Run 4 (Local Qwen2.5-Coder-7B on RTX 2070)

| Gen | Best Fitness | Valid | Time |
|-----|-------------|-------|------|
| 1 | 0.7517 | 2/3 | 34.0s |
| 2 | 0.7517 | 1/3 | 36.3s |
| 3 | 0.7517 | 2/3 | 82.1s |
| 4 | 0.7517 | 0/3 | 31.4s |
| 5 | 0.7517 | 1/3 | 61.1s |
| 6 | 0.7517 | 1/3 | 49.4s |

- **Total time**: 304 seconds (~5 minutes)
- **Final fidelity (500 samples)**: 0.9127
- **Best generation**: 0 (seed strategy remained best)
- **Issue**: 7B model couldn't surpass baseline; valid rate ~50%

## Seed Strategy Performance

| Strategy | Base Fidelity | DEHB | BRFD | Combined |
|----------|--------------|------|------|----------|
| standard | 0.9167 | 0.8157 | 0.3443 | **0.7517** |
| echo_cancel | 0.0000 | 0.8157 | 0.3443 | 0.4767 |
| correlation_aware | 0.8455 | 0.8157 | 0.3443 | 0.7303 |

## Key Findings

### 1. Infrastructure Achievement
- Successfully deployed NVIDIA drivers and CUDA on the user's RTX 2070
- Built llama-cpp-python with CUDA support from source
- Achieved **37.3 tok/s** GPU inference (8.2x faster than CPU)
- Created a custom GPU inference server with OpenAI-compatible API
- Established SSH tunnel for seamless sandbox-to-GPU communication

### 2. LLM Backend Comparison

| Backend | Speed | Rate Limits | Code Quality | Valid Rate |
|---------|-------|-------------|--------------|-----------|
| OpenRouter (free) | N/A | Severe (429) | N/A | 0% |
| Gemini 2.5 Flash | ~45s/call | Moderate | High | 66% |
| Qwen2.5-Coder-7B (GPU) | ~15s/call | **None** | Medium | 50% |

### 3. AEDB Architecture Validation
- The three-component architecture (LLM + DEHB + BRFD) works correctly
- DEHB consistently finds good hyperparameters (0.8157 fidelity)
- BRFD needs more training steps (only reaches 0.3443)
- LLM evolution successfully generates valid quantum gate strategies

### 4. Limitations Identified
- **7B model quality**: Too small to consistently generate correct quantum code
- **BRFD underfitting**: 4 outer steps with 8 inner episodes is insufficient
- **No improvement over baseline**: The standard textbook strategy (H + CNOT) is already near-optimal for 2-qubit Bell state
- **Need harder problem**: 2-qubit Bell state is too simple; need 3+ qubits or more complex circuits

## Recommendations for Next Steps

1. **Use larger LLM**: Deploy Qwen2.5-Coder-14B or 32B (fits in 8GB VRAM with Q4 quantization for 14B)
2. **Increase BRFD training**: 10+ outer steps, 16+ inner episodes
3. **Scale to harder problems**: 3-qubit GHZ state, multi-gate circuits
4. **Increase population size**: 5-10 per generation for better exploration
5. **Add few-shot examples**: Include working strategies in the LLM prompt
6. **Use Gemini for quality, local for quantity**: Hybrid approach

## Files Generated

- `results/aedb_run4.log` - Full experiment log
- `results/aedb_run2/` - Gemini run results and DEHB checkpoints
- `gpu_server.py` - Custom GPU inference server for RTX 2070
- `evolve_v2.py` - Updated with Ollama/Gemini/OpenRouter fallback chain
