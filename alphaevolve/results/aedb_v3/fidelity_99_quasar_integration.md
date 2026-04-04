# Roadmap to >99% Gate Fidelity: Integrating AEDB with the Full QUASAR Architecture

## 1. Context and Bottleneck Re-evaluation

The recent AEDB v3 experiment successfully demonstrated the automated discovery of noise-mitigating quantum gate sequences, achieving a peak fidelity of 0.7109 for a 3-qubit GHZ state under TLF-correlated charge noise. While a significant achievement for the AlphaEvolve + DEHB + BRFD subset, this falls short of the >99% fidelity threshold required for fault tolerance.

A critical review of the overarching `DYNAMO-Core` and `QUASAR` architectural specifications reveals that **the AEDB component was never intended to achieve >99% fidelity in isolation**. 

The AEDB v3 experiment ran only 3 of the 8 core components of the DYNAMO-Core framework (AlphaEvolve, DEHB, BRFD). It operated in an open-loop manner at the discrete gate level. The full QUASAR architecture is explicitly designed to solve the exact bottlenecks encountered in the AEDB experiment.

The question is not "how do we rewrite AEDB to get 99% fidelity?", but rather, **"how do we properly integrate the AEDB optimization loop into the full QUASAR stack to unlock its 99% potential?"**

## 2. How QUASAR Solves the Current Bottlenecks

The bottlenecks identified in the AEDB v3 experiment (linear noise accumulation, open-loop control, limited BRFD expressiveness) are directly addressed by the missing components of the QUASAR architecture:

| AEDB v3 Limitation | QUASAR Solution | Mechanism |
| :--- | :--- | :--- |
| **Open-Loop Gate Sequences** | **Soft Actor-Critic (SAC)** | Replaces static gate sequences with a closed-loop DRL policy that applies continuous, real-time pulse corrections based on the 45-dimensional observation space (including pairwise correlators) [1]. |
| **Unconstrained Search Space** | **CAMEL-Q** | Contextual Action Masking with Expert Prior restricts the SAC agent's exploration to physically viable pulse shapes, drastically accelerating convergence [2]. |
| **Noise Drift Vulnerability** | **ERL & SDFT** | Experiential Reinforcement Learning (ERL) and Self-Distillation Fine-Tuning (SDFT) provide reactive stability against non-stationary 1/f noise, preventing catastrophic forgetting [3] [4]. |
| **Tabular BRFD Policy** | **Fidelity-Based Reward** | The BRFD component in QUASAR shapes the reward for the deep neural network SAC agent, rather than relying on a simplistic tabular policy [5]. |

## 3. Strategic Recommendations for AEDB-QUASAR Integration

To achieve the target fidelity of >99% (F ≥ 0.99) as specified in the thesis proposal, we must transition from the isolated AEDB experiment to the full Multi-Qubit Control Framework (MQCF).

### 3.1. Shift AEDB's Target from "Gate Sequences" to "SAC Hyperparameters"

Currently, AlphaEvolve is trying to evolve the quantum control strategy directly. In the formal QUASAR architecture, **SAC is the controller, and AlphaEvolve/DEHB is the optimizer**.

*   **Action:** Modify the AlphaEvolve prompt templates and `skeleton.py`. Instead of evolving `Gate("rx", ...)` sequences, AlphaEvolve must evolve the architectural hyperparameters and reward shaping functions for the SAC agent.
*   **Rationale:** The continuous action space of pulse-level control is too vast for LLM-driven discrete code evolution to solve efficiently. SAC is purpose-built for continuous control. AlphaEvolve should focus on what it does best: meta-algorithmic discovery [6].

### 3.2. Integrate the SiliQun Gymnasium Environment

The AEDB v3 experiment used a simplified, standalone `fitness.py` evaluator. This must be replaced with the full physics engine.

*   **Action:** Replace `fitness.py` with `SiliQunEnv` from `siliqun.engine.gym_env`.
*   **Rationale:** `SiliQunEnv` provides the rigorous, MPO-based simulation of silicon spin qubits (Donor, SiMOS, GAA) with the full two-level noise model (T1, T2*, 1/f charge noise) required by the proposal. It exposes the critical 45-dimensional observation space needed by the SAC agent to detect correlated noise.

### 3.3. Implement the DRL Convergence Framework (Mandatory)

The project requirements mandate a dual-framework approach: the DRL Convergence Framework and the ScaleRL Framework.

*   **Action:** Ensure the SAC implementation integrates experience replay, target networks, entropy regularization, and gradient clipping. Deploy the **SWDFT (Sliding-Window DFT)** monitor to act as the arbitrator between proactive (DEHB/BRFD) and reactive (ERL/SDFT) modes [2].
*   **Rationale:** DRL for quantum control is notoriously unstable. The SWDFT monitor is the core architectural innovation of DYNAMO-Core that guarantees stability and is later repurposed in SeQurAIty as a security primitive against adversarial attacks.

## 4. Execution Roadmap

To reach >99% fidelity, the implementation must proceed in the following phases on the Aziz Supercomputer:

1.  **Phase 1: Environment Swap:** Integrate `SiliQunEnv` into the evaluation loop. Validate that the MPO-based simulator correctly models the 3-qubit GHZ state under the full two-level noise model.
2.  **Phase 2: SAC Integration:** Introduce the Stable-Baselines3 SAC agent. Configure it to accept the 45-dimensional observation space and output continuous pulse parameters.
3.  **Phase 3: AEDB Repurposing:** Rewire AlphaEvolve, DEHB, and BRFD to optimize the SAC agent's hyperparameters and reward function, rather than generating gate sequences directly.
4.  **Phase 4: Full MQCF Run:** Execute the complete pipeline (Phase 1: Digital Twin → Phase 2: AlphaEvolve HPO → Phase 3: SAC Training) on Aziz to achieve F ≥ 0.99.

By aligning the implementation with the formally defined QUASAR architecture, the path to fault-tolerant fidelities is clearly defined and computationally feasible.

## References

[1] R. Al-Shehri, "MASTER ARCHITECTURE AND PUBLICATION STRATEGY," *DYNAMO-Core · QUASAR · MOZAIQ · SeQurAIty*, 2026.
[2] R. Al-Shehri, "DYNAMO → QUASAR: Architecture Flexibility Gap Analysis," 2026.
[3] T. Shi et al., "Experiential Reinforcement Learning," *arXiv:2602.13949*, 2026.
[4] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," *arXiv:2601.19897*, 2026.
[5] R. Lu et al., "Discovery of the reward function for embodied reinforcement learning agents," *Nature Communications*, 2025.
[6] Z. Li et al., "Discovering Multiagent Learning Algorithms with Large Language Models," *arXiv:2602.16928*, 2026.
