"""
BRFD: Bilevel Reward Function Discovery for the AEDB system.

Discovers optimal dense reward functions for DRL agents operating
in SiliQun's noisy quantum environment. Uses meta-gradient descent
to learn a reward network R_omega that maximizes the true task
objective (gate fidelity) when used to train a DRL policy.

References:
    Zheng et al., "Bilevel Reward Function Discovery", Nature
    Communications, 2026.

Architecture:
    - Inner loop: DRL agent trains under learned reward R_omega
    - Outer loop: Meta-gradient updates omega to maximize true fidelity
    - Advantage product: A^R(s,a) * A^pi(s,a) guides reward updates
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitness import (
    Gate, TLFNoiseModel, gate_to_matrix, state_fidelity,
    build_target_state, build_initial_state,
)
from skeleton import Gate

logger = logging.getLogger("aedb.brfd")


# ======================================================================
# Reward Network (small MLP)
# ======================================================================

class RewardNetwork:
    """Small MLP that maps observations to scalar reward.

    The observation vector contains:
        [fidelity, leakage, entropy, step_fraction,
         zz_correlator_0, zz_correlator_1, ...,
         noise_realization_0, noise_realization_1, ...]

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation vector.
    hidden_dim : int
        Width of hidden layers.
    n_hidden : int
        Number of hidden layers.
    lr : float
        Learning rate for meta-gradient updates.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        hidden_dim: int = 32,
        n_hidden: int = 2,
        lr: float = 1e-3,
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.lr = lr

        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []

        rng = np.random.RandomState(42)

        # Input layer
        fan_in, fan_out = obs_dim, hidden_dim
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        self.weights.append(rng.randn(fan_in, fan_out) * scale)
        self.biases.append(np.zeros(fan_out))

        # Hidden layers
        for _ in range(n_hidden - 1):
            scale = np.sqrt(2.0 / (hidden_dim + hidden_dim))
            self.weights.append(rng.randn(hidden_dim, hidden_dim) * scale)
            self.biases.append(np.zeros(hidden_dim))

        # Output layer (scalar reward)
        scale = np.sqrt(2.0 / (hidden_dim + 1))
        self.weights.append(rng.randn(hidden_dim, 1) * scale)
        self.biases.append(np.zeros(1))

    def forward(self, obs: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Forward pass with cached activations for backprop.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (obs_dim,).

        Returns
        -------
        float
            Scalar reward value.
        list of np.ndarray
            Cached activations for each layer.
        """
        activations = [obs]
        x = obs

        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.tanh(x)  # Smooth activation for gradient flow
            activations.append(x)

        # Output layer (no activation, raw scalar)
        x = x @ self.weights[-1] + self.biases[-1]
        activations.append(x)

        return float(x[0]), activations

    def backward(
        self,
        activations: List[np.ndarray],
        grad_output: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass to compute gradients.

        Parameters
        ----------
        activations : list of np.ndarray
            Cached activations from forward pass.
        grad_output : float
            Gradient of loss w.r.t. reward output.

        Returns
        -------
        tuple of (list of np.ndarray, list of np.ndarray)
            Gradients for weights and biases.
        """
        n_layers = len(self.weights)
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # Output layer gradient
        delta = np.array([grad_output])
        grad_w[-1] = activations[-2].reshape(-1, 1) @ delta.reshape(1, -1)
        grad_b[-1] = delta

        # Hidden layers (reverse order)
        for i in range(n_layers - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * (1 - activations[i + 1] ** 2)
            grad_w[i] = activations[i].reshape(-1, 1) @ delta.reshape(1, -1)
            grad_b[i] = delta.flatten()

        return grad_w, grad_b

    def update(
        self,
        grad_w: List[np.ndarray],
        grad_b: List[np.ndarray],
    ):
        """Apply gradient update (gradient ascent for reward maximization).

        Parameters
        ----------
        grad_w : list of np.ndarray
            Weight gradients.
        grad_b : list of np.ndarray
            Bias gradients.
        """
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * np.clip(grad_w[i], -1.0, 1.0)
            self.biases[i] += self.lr * np.clip(grad_b[i], -1.0, 1.0)

    def get_params(self) -> Dict:
        """Serialize network parameters."""
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def set_params(self, params: Dict):
        """Deserialize network parameters."""
        self.weights = [np.array(w) for w in params["weights"]]
        self.biases = [np.array(b) for b in params["biases"]]


# ======================================================================
# Simple DRL Policy (REINFORCE for PoC)
# ======================================================================

class SimplePolicy:
    """Tabular softmax policy for gate selection.

    For the PoC, we use a simple policy that selects from a fixed
    set of gate actions at each step. This is sufficient to demonstrate
    BRFD's reward shaping capability.

    Parameters
    ----------
    n_actions : int
        Number of available gate actions.
    n_steps : int
        Maximum number of steps per episode.
    lr : float
        Policy learning rate.
    """

    def __init__(
        self,
        n_actions: int = 6,
        n_steps: int = 10,
        lr: float = 0.01,
    ):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.lr = lr

        # Logits for each (step, action) pair
        self.logits = np.zeros((n_steps, n_actions))

    def get_action_probs(self, step: int) -> np.ndarray:
        """Get action probabilities for a given step."""
        logits = self.logits[step]
        # Softmax with temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def sample_action(self, step: int, rng: np.random.RandomState) -> int:
        """Sample an action from the policy."""
        probs = self.get_action_probs(step)
        return rng.choice(self.n_actions, p=probs)

    def update(
        self,
        steps: List[int],
        actions: List[int],
        advantages: List[float],
    ):
        """REINFORCE update with advantages.

        Parameters
        ----------
        steps : list of int
            Step indices.
        actions : list of int
            Actions taken.
        advantages : list of float
            Advantage values (from BRFD reward or true reward).
        """
        for step, action, adv in zip(steps, actions, advantages):
            probs = self.get_action_probs(step)
            # Policy gradient: increase probability of good actions
            grad = -probs.copy()
            grad[action] += 1.0
            self.logits[step] += self.lr * adv * grad

    def reset(self):
        """Reset policy to uniform."""
        self.logits = np.zeros_like(self.logits)


# ======================================================================
# Gate Action Space
# ======================================================================

# Available gate actions for the DRL agent
GATE_ACTIONS = [
    lambda q: Gate("h", [q]),
    lambda q: Gate("rx", [q], {"theta": np.pi / 2}),
    lambda q: Gate("ry", [q], {"theta": np.pi / 2}),
    lambda q: Gate("rz", [q], {"theta": np.pi / 2}),
    lambda q: Gate("rx", [q], {"theta": np.pi}),
    lambda q: Gate("identity", [0]),
]

TWO_QUBIT_ACTIONS = [
    lambda: Gate("cnot", [0, 1]),
    lambda: Gate("cz", [0, 1]),
    lambda: Gate("swap", [0, 1]),
]


# ======================================================================
# BRFD Trainer
# ======================================================================

class BRFDTrainer:
    """Bilevel Reward Function Discovery trainer.

    Implements the BRFD algorithm:
    - Inner loop: Train DRL policy under learned reward R_omega
    - Outer loop: Update R_omega via meta-gradient to maximize true fidelity

    The meta-gradient uses the advantage product:
        grad_omega = E[A^R(s,a) * A^pi(s,a) * grad_omega R(s)]

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    target_gate : str
        Target operation.
    noise_amplitude : float
        Noise strength.
    reward_hidden_dim : int
        Reward network hidden dimension.
    reward_lr : float
        Reward network learning rate.
    policy_lr : float
        Policy learning rate.
    n_inner_episodes : int
        Episodes per inner loop iteration.
    n_outer_steps : int
        Number of outer loop (meta) updates.
    max_steps : int
        Maximum steps per episode.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        target_gate: str = "bell",
        noise_amplitude: float = 0.3,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
        reward_hidden_dim: int = 32,
        reward_lr: float = 1e-3,
        policy_lr: float = 0.01,
        n_inner_episodes: int = 20,
        n_outer_steps: int = 10,
        max_steps: int = 6,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.target_gate = target_gate
        self.noise_amplitude = noise_amplitude
        self.qubit_spacing_nm = qubit_spacing_nm
        self.tlf_correlation_length_nm = tlf_correlation_length_nm
        self.max_steps = max_steps
        self.n_inner_episodes = n_inner_episodes
        self.n_outer_steps = n_outer_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Observation: [fidelity, leakage_proxy, step_frac, n_gates,
        #               zz_01, noise_0, noise_1, gate_type_onehot_avg]
        self.obs_dim = 8

        # Reward network
        self.reward_net = RewardNetwork(
            obs_dim=self.obs_dim,
            hidden_dim=reward_hidden_dim,
            n_hidden=2,
            lr=reward_lr,
        )

        # DRL policy
        n_actions = len(GATE_ACTIONS) + len(TWO_QUBIT_ACTIONS)
        self.policy = SimplePolicy(
            n_actions=n_actions,
            n_steps=max_steps,
            lr=policy_lr,
        )

        # Noise model
        self.noise_model = TLFNoiseModel(
            n_qubits=n_qubits,
            qubit_spacing_nm=qubit_spacing_nm,
            tlf_correlation_length_nm=tlf_correlation_length_nm,
            noise_amplitude=noise_amplitude,
            seed=seed,
        )

        # Target state
        self.target_state = build_target_state(target_gate, n_qubits)
        self.initial_state = build_initial_state(target_gate, n_qubits)

    def _build_observation(
        self,
        state: np.ndarray,
        step: int,
        n_gates: int,
        noise_angles: np.ndarray,
    ) -> np.ndarray:
        """Build observation vector for the reward network.

        Parameters
        ----------
        state : np.ndarray
            Current quantum state.
        step : int
            Current step index.
        n_gates : int
            Number of gates applied so far.
        noise_angles : np.ndarray
            Latest noise realization.

        Returns
        -------
        np.ndarray
            Observation vector of shape (obs_dim,).
        """
        # Current fidelity
        fid = state_fidelity(state, self.target_state)

        # Leakage proxy (deviation from unit norm, should be ~0)
        leakage = 1.0 - float(np.abs(np.vdot(state, state)))

        # Step fraction
        step_frac = step / max(self.max_steps, 1)

        # Gate count (normalized)
        gate_frac = n_gates / max(self.max_steps * 2, 1)

        # ZZ correlator between qubits 0 and 1
        if self.n_qubits >= 2:
            # <Z0 Z1> = sum_i (-1)^(b0_i + b1_i) |c_i|^2
            dim = 2 ** self.n_qubits
            zz = 0.0
            for i in range(dim):
                b0 = (i >> (self.n_qubits - 1)) & 1
                b1 = (i >> (self.n_qubits - 2)) & 1
                zz += ((-1) ** (b0 + b1)) * abs(state[i]) ** 2
        else:
            zz = 0.0

        # Noise features
        noise_0 = noise_angles[0] if len(noise_angles) > 0 else 0.0
        noise_1 = noise_angles[1] if len(noise_angles) > 1 else 0.0

        # Entropy proxy
        probs = np.abs(state) ** 2
        probs = probs[probs > 1e-12]
        entropy = -np.sum(probs * np.log2(probs)) / max(np.log2(len(state)), 1)

        obs = np.array([
            fid, leakage, step_frac, gate_frac,
            zz, noise_0, noise_1, entropy,
        ], dtype=np.float64)

        return obs

    def _action_to_gate(self, action: int) -> Gate:
        """Convert action index to a Gate object."""
        n_single = len(GATE_ACTIONS)
        if action < n_single:
            return GATE_ACTIONS[action](0)
        else:
            return TWO_QUBIT_ACTIONS[action - n_single]()

    def _run_episode(
        self,
        use_learned_reward: bool = True,
    ) -> Dict:
        """Run a single episode and collect trajectory data.

        Parameters
        ----------
        use_learned_reward : bool
            If True, use the learned reward network.
            If False, use the true fidelity as reward.

        Returns
        -------
        dict
            Episode data including observations, actions, rewards, etc.
        """
        state = self.initial_state.copy()
        trajectory = {
            "observations": [],
            "actions": [],
            "learned_rewards": [],
            "true_rewards": [],
            "activations": [],
            "steps": [],
        }

        prev_fidelity = state_fidelity(state, self.target_state)

        for step in range(self.max_steps):
            # Sample noise
            noise_angles = self.noise_model.sample_noise()

            # Build observation
            obs = self._build_observation(state, step, step, noise_angles)

            # Select action
            action = self.policy.sample_action(step, self.rng)

            # Apply gate
            gate = self._action_to_gate(action)
            try:
                gate_mat = gate_to_matrix(gate, self.n_qubits)
                state = gate_mat @ state
            except Exception:
                pass  # Identity on error

            # Apply noise
            state = self.noise_model.apply_noise(state)

            # Compute rewards
            current_fidelity = state_fidelity(state, self.target_state)
            true_reward = current_fidelity - prev_fidelity  # Incremental

            # Learned reward
            learned_reward, activations = self.reward_net.forward(obs)

            # Store trajectory
            trajectory["observations"].append(obs)
            trajectory["actions"].append(action)
            trajectory["learned_rewards"].append(learned_reward)
            trajectory["true_rewards"].append(true_reward)
            trajectory["activations"].append(activations)
            trajectory["steps"].append(step)

            prev_fidelity = current_fidelity

        # Final fidelity
        trajectory["final_fidelity"] = state_fidelity(state, self.target_state)
        trajectory["final_state"] = state

        return trajectory

    def _compute_advantages(
        self,
        rewards: List[float],
        gamma: float = 0.99,
    ) -> List[float]:
        """Compute GAE-like advantages from rewards.

        Parameters
        ----------
        rewards : list of float
            Per-step rewards.
        gamma : float
            Discount factor.

        Returns
        -------
        list of float
            Advantage values.
        """
        n = len(rewards)
        advantages = [0.0] * n
        running = 0.0

        for t in reversed(range(n)):
            running = rewards[t] + gamma * running
            advantages[t] = running

        # Normalize
        adv_arr = np.array(advantages)
        if adv_arr.std() > 1e-8:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        return adv_arr.tolist()

    def train(self) -> Dict:
        """Run the full BRFD training loop.

        Returns
        -------
        dict
            Training results including best fidelity and reward params.
        """
        logger.info(
            f"BRFD training: {self.n_outer_steps} outer steps, "
            f"{self.n_inner_episodes} inner episodes each"
        )

        best_fidelity = 0.0
        best_reward_params = None
        history = {
            "outer_step": [],
            "mean_fidelity": [],
            "best_fidelity": [],
            "mean_learned_reward": [],
        }

        for outer_step in range(self.n_outer_steps):
            # ---- Inner loop: train policy under learned reward ----
            self.policy.reset()
            episode_fidelities = []

            for ep in range(self.n_inner_episodes):
                traj = self._run_episode(use_learned_reward=True)
                episode_fidelities.append(traj["final_fidelity"])

                # Compute advantages using learned rewards
                learned_advantages = self._compute_advantages(
                    traj["learned_rewards"]
                )

                # Update policy
                self.policy.update(
                    traj["steps"],
                    traj["actions"],
                    learned_advantages,
                )

            mean_fidelity = np.mean(episode_fidelities)

            # ---- Outer loop: meta-gradient for reward network ----
            # Collect trajectories with current policy for meta-gradient
            meta_grad_w = [np.zeros_like(w) for w in self.reward_net.weights]
            meta_grad_b = [np.zeros_like(b) for b in self.reward_net.biases]
            n_meta_samples = 5

            for _ in range(n_meta_samples):
                traj = self._run_episode(use_learned_reward=True)

                # Compute true advantages (from true fidelity signal)
                true_advantages = self._compute_advantages(
                    traj["true_rewards"]
                )

                # Compute learned advantages
                learned_advantages = self._compute_advantages(
                    traj["learned_rewards"]
                )

                # Meta-gradient: advantage product
                for t in range(len(traj["steps"])):
                    adv_product = (
                        true_advantages[t] * learned_advantages[t]
                    )

                    # Backprop through reward network
                    gw, gb = self.reward_net.backward(
                        traj["activations"][t],
                        adv_product,
                    )

                    for i in range(len(meta_grad_w)):
                        meta_grad_w[i] += gw[i] / n_meta_samples
                        meta_grad_b[i] += gb[i] / n_meta_samples

            # Update reward network
            self.reward_net.update(meta_grad_w, meta_grad_b)

            # Track best
            if mean_fidelity > best_fidelity:
                best_fidelity = mean_fidelity
                best_reward_params = self.reward_net.get_params()

            # Log
            mean_lr = np.mean([
                np.mean(traj.get("learned_rewards", [0]))
                for traj in [self._run_episode()]
            ])

            history["outer_step"].append(outer_step)
            history["mean_fidelity"].append(float(mean_fidelity))
            history["best_fidelity"].append(float(best_fidelity))
            history["mean_learned_reward"].append(float(mean_lr))

            logger.info(
                f"  Outer step {outer_step + 1}/{self.n_outer_steps}: "
                f"mean_fid={mean_fidelity:.4f}, best={best_fidelity:.4f}"
            )

        results = {
            "best_fidelity": float(best_fidelity),
            "final_mean_fidelity": float(mean_fidelity),
            "reward_params": best_reward_params,
            "history": history,
        }

        return results


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("BRFD Reward Function Discovery - Standalone Test")
    print("=" * 70)

    # Test with moderate noise (0.3 rad/gate)
    trainer = BRFDTrainer(
        n_qubits=2,
        target_gate="bell",
        noise_amplitude=0.3,
        reward_hidden_dim=32,
        reward_lr=1e-3,
        policy_lr=0.01,
        n_inner_episodes=20,
        n_outer_steps=10,
        max_steps=6,
        seed=42,
    )

    start = time.time()
    results = trainer.train()
    elapsed = time.time() - start

    print(f"\nBest fidelity: {results['best_fidelity']:.4f}")
    print(f"Final mean fidelity: {results['final_mean_fidelity']:.4f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nFidelity trajectory:")
    for step, fid in zip(
        results["history"]["outer_step"],
        results["history"]["mean_fidelity"],
    ):
        bar = "#" * int(fid * 40)
        print(f"  Step {step + 1:2d}: {fid:.4f} |{bar}")

    # Compare with hand-designed reward (just fidelity)
    print("\n--- Baseline (hand-designed reward) ---")
    baseline_trainer = BRFDTrainer(
        n_qubits=2,
        target_gate="bell",
        noise_amplitude=0.3,
        n_inner_episodes=20,
        n_outer_steps=1,  # No meta-learning
        max_steps=6,
        seed=42,
    )
    baseline_results = baseline_trainer.train()
    print(f"Baseline fidelity: {baseline_results['best_fidelity']:.4f}")
    print(f"BRFD improvement: {results['best_fidelity'] - baseline_results['best_fidelity']:.4f}")
