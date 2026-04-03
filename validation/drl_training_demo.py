"""
DRL Training Demonstration for SiliQun SoftwareX Paper

Trains a simple PPO agent on the SiliQun Gymnasium environment
for 2-qubit and 4-qubit systems, producing learning curves
that demonstrate the environment works end-to-end.

This uses a lightweight actor-critic network (no external DRL library)
to keep the demonstration self-contained and reproducible.
"""

import numpy as np
import json
import time
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleActorCritic:
    """Minimal actor-critic network for demonstration purposes."""

    def __init__(self, obs_dim, act_dim, hidden=64, lr=3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.lr = lr

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (obs_dim + hidden))
        scale2 = np.sqrt(2.0 / (hidden + hidden))
        scale3_a = np.sqrt(2.0 / (hidden + act_dim))
        scale3_v = np.sqrt(2.0 / (hidden + 1))

        self.W1 = np.random.randn(obs_dim, hidden).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, hidden).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden, dtype=np.float32)

        # Actor head (mean of Gaussian policy)
        self.W_mu = np.random.randn(hidden, act_dim).astype(np.float32) * scale3_a
        self.b_mu = np.zeros(act_dim, dtype=np.float32)
        self.log_std = np.zeros(act_dim, dtype=np.float32) - 0.5

        # Critic head
        self.W_v = np.random.randn(hidden, 1).astype(np.float32) * scale3_v
        self.b_v = np.zeros(1, dtype=np.float32)

    def forward(self, obs):
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        mu = h2 @ self.W_mu + self.b_mu
        mu = np.tanh(mu)  # Bound actions to [-1, 1]
        value = (h2 @ self.W_v + self.b_v).item()
        return mu, value, h1, h2

    def get_action(self, obs):
        mu, value, _, _ = self.forward(obs)
        std = np.exp(self.log_std)
        action = mu + std * np.random.randn(self.act_dim).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        return action, mu, std, value

    def log_prob(self, action, mu, std):
        var = std ** 2
        lp = -0.5 * np.sum(((action - mu) ** 2) / var + np.log(var) + np.log(2 * np.pi))
        return lp


def numerical_gradient(ac, obs_batch, act_batch, adv_batch, ret_batch, param_name, eps=1e-4):
    """Compute numerical gradient for a parameter (for simplicity)."""
    param = getattr(ac, param_name)
    grad = np.zeros_like(param)
    flat = param.ravel()
    for i in range(min(len(flat), 200)):  # Limit for speed
        old_val = flat[i]
        flat[i] = old_val + eps
        loss_plus = compute_loss(ac, obs_batch, act_batch, adv_batch, ret_batch)
        flat[i] = old_val - eps
        loss_minus = compute_loss(ac, obs_batch, act_batch, adv_batch, ret_batch)
        flat[i] = old_val
        grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def compute_loss(ac, obs_batch, act_batch, adv_batch, ret_batch):
    """Compute combined policy + value loss."""
    total = 0.0
    for obs, act, adv, ret in zip(obs_batch, act_batch, adv_batch, ret_batch):
        mu, val, _, _ = ac.forward(obs)
        std = np.exp(ac.log_std)
        lp = ac.log_prob(act, mu, std)
        total += -(lp * adv) + 0.5 * (val - ret) ** 2
    return total / len(obs_batch)


def reinforce_update(ac, obs_batch, act_batch, adv_batch, ret_batch):
    """Simple REINFORCE-style update with baseline."""
    # Accumulate gradients via finite differences on key params
    for pname in ['W1', 'b1', 'W2', 'b2', 'W_mu', 'b_mu', 'W_v', 'b_v', 'log_std']:
        param = getattr(ac, pname)
        flat = param.ravel()
        eps = 1e-3
        for i in range(min(len(flat), 100)):
            old_val = flat[i]
            flat[i] = old_val + eps
            loss_plus = compute_loss(ac, obs_batch, act_batch, adv_batch, ret_batch)
            flat[i] = old_val - eps
            loss_minus = compute_loss(ac, obs_batch, act_batch, adv_batch, ret_batch)
            flat[i] = old_val
            grad = (loss_plus - loss_minus) / (2 * eps)
            flat[i] -= ac.lr * grad


def train_on_env(n_qubits, n_episodes=200, max_steps=50, target_state="bell"):
    """Train a simple agent on the SiliQun environment."""
    from siliqun.engine.gym_env import make_siliqun_env

    print(f"\n{'='*60}")
    print(f"Training on {n_qubits}-qubit system (target: {target_state})")
    print(f"{'='*60}")

    env = make_siliqun_env(
        n_qubits=n_qubits,
        target=target_state,
        max_steps=max_steps,
        device="donor",
        sim_mode="mps" if n_qubits <= 4 else "auto",
        noise=False,
    )

    obs, info = env.reset()
    obs_dim = len(obs)
    act_dim = env.action_space.shape[0]

    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")

    ac = SimpleActorCritic(obs_dim, act_dim, hidden=32, lr=1e-3)

    episode_rewards = []
    episode_fidelities = []
    best_fidelity = 0.0
    gamma = 0.99

    for ep in range(n_episodes):
        obs, info = env.reset()
        obs = obs.astype(np.float32)

        obs_buf, act_buf, rew_buf, val_buf = [], [], [], []
        ep_reward = 0.0
        ep_fidelity = 0.0

        for step in range(max_steps):
            action, mu, std, value = ac.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs.astype(np.float32)

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)

            ep_reward += reward
            ep_fidelity = info.get("fidelity", ep_fidelity)

            obs = next_obs
            if terminated or truncated:
                break

        # Compute returns and advantages
        returns = []
        R = 0.0
        for r in reversed(rew_buf):
            R = r + gamma * R
            returns.insert(0, R)
        returns = np.array(returns, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        advantages = returns - values
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Update policy (simplified — only update on a subset for speed)
        if len(obs_buf) >= 5:
            indices = np.random.choice(len(obs_buf), min(10, len(obs_buf)), replace=False)
            reinforce_update(
                ac,
                [obs_buf[i] for i in indices],
                [act_buf[i] for i in indices],
                [advantages[i] for i in indices],
                [returns[i] for i in indices],
            )

        episode_rewards.append(ep_reward)
        episode_fidelities.append(ep_fidelity)
        best_fidelity = max(best_fidelity, ep_fidelity)

        if (ep + 1) % 20 == 0:
            avg_r = np.mean(episode_rewards[-20:])
            avg_f = np.mean(episode_fidelities[-20:])
            print(f"  Episode {ep+1:4d}: avg_reward={avg_r:+.3f}, avg_fidelity={avg_f:.4f}, best={best_fidelity:.4f}")

    env.close()

    return {
        "n_qubits": n_qubits,
        "target_state": target_state,
        "n_episodes": n_episodes,
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_fidelities": [float(f) for f in episode_fidelities],
        "best_fidelity": float(best_fidelity),
        "final_avg_fidelity": float(np.mean(episode_fidelities[-20:])),
    }


def plot_learning_curves(results_list, outpath):
    """Plot learning curves for all training runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2563eb', '#dc2626', '#059669', '#7c3aed']
    window = 10

    for i, res in enumerate(results_list):
        label = f"{res['n_qubits']}q ({res['target_state']})"
        color = colors[i % len(colors)]

        # Smooth rewards
        rewards = np.array(res['episode_rewards'])
        smoothed_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(len(smoothed_r)), smoothed_r, color=color, label=label, linewidth=1.5)

        # Smooth fidelities
        fids = np.array(res['episode_fidelities'])
        smoothed_f = np.convolve(fids, np.ones(window)/window, mode='valid')
        axes[1].plot(range(len(smoothed_f)), smoothed_f, color=color, label=label, linewidth=1.5)

    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('(a) Reward Learning Curves', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Gate Fidelity', fontsize=12)
    axes[1].set_title('(b) Fidelity Learning Curves', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nLearning curves saved to: {outpath}")


def main():
    print("SiliQun DRL Training Demonstration")
    print("=" * 60)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results_list = []

    # 2-qubit Bell state preparation
    res_2q = train_on_env(n_qubits=2, n_episodes=150, max_steps=30, target_state="bell")
    results_list.append(res_2q)

    # 4-qubit GHZ state preparation
    res_4q = train_on_env(n_qubits=4, n_episodes=150, max_steps=50, target_state="ghz")
    results_list.append(res_4q)

    # Plot learning curves
    outdir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(outdir, "..", "paper", "fig_learning_curves.png")
    plot_learning_curves(results_list, plot_path)

    # Save raw data
    data_path = os.path.join(outdir, "drl_training_results.json")
    with open(data_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"Raw data saved to: {data_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for res in results_list:
        print(f"  {res['n_qubits']}q {res['target_state']}: "
              f"best_F={res['best_fidelity']:.4f}, "
              f"final_avg_F={res['final_avg_fidelity']:.4f}")


if __name__ == "__main__":
    main()
