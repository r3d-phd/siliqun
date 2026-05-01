# -*- coding: utf-8 -*-
"""
QUASAR v3 Experiment Runner — VAPOR-lite + Elite-Preservation PBT — FIXED

Fixes applied:
  1. DEHB: min_budget → min_fidelity, max_budget → max_fidelity (v0.1.0+ API)
  2. Checkpoint: agent.critic → agent.critics (ModuleList)
  3. Agent: pass sigma_min=0.1, auto_alpha=True to VAPORSACAgent
  4. Results dir: separate v3_fixed results to avoid conflicts with old runs

Conditions (5 seeds each, 15 runs total):
  1. dehb_brfd          : DEHB+BRFD warm-start, standard SAC (control)
  2. dehb_brfd_vapor    : DEHB+BRFD warm-start, VAPOR-lite SAC (new)
  3. dehb_brfd_vapor_pbt: DEHB+BRFD warm-start, VAPOR-lite SAC + Elite PBT (full)
"""
import os
import sys
import json
import time
import logging
import copy
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SILIQUN_PATH = os.path.expanduser("~/siliqun")
QUASAR_PATH  = os.path.expanduser("~/quasar")
for p in [SILIQUN_PATH, QUASAR_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch

# SiliQun environment
import siliqun
from siliqun.engine.gym_env import SiliQunEnv

# QUASAR v3 modules (same directory as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from sac_vapor import VAPORSACAgent
from pbt_elite import ElitePBTScheduler

# BRFD reward shaping (from quasar package)
try:
    from quasar.training.brfd import BRFDRewardShaper
    HAS_BRFD = True
except ImportError:
    HAS_BRFD = False

# DEHB hyperparameter optimization
try:
    from dehb import DEHB
    HAS_DEHB = True
except ImportError:
    HAS_DEHB = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]

CONDITIONS = [
    "dehb_brfd",
    "dehb_brfd_vapor",
    "dehb_brfd_vapor_pbt",
]

# Training budget
TOTAL_STEPS  = 500_000   # FIX v4: 5× budget to allow SAC to escape zero-action local min
MAX_EP_STEPS = 20        # Short episodes → more episodes (5,000 total)
WARMUP_STEPS = 10_000    # FIX v4: 10× warmup for better initial exploration
EVAL_EPISODES = 20       # Episodes for evaluation

# PBT configuration
PBT_ROUNDS      = 10
PBT_POP_SIZE    = 8
STEPS_PER_ROUND = TOTAL_STEPS // PBT_ROUNDS  # 50,000 steps per round

# DEHB configuration
DEHB_BUDGET = 50         # Number of HP evaluations
DEHB_STEPS  = 5_000      # Steps per DEHB evaluation

# Default SAC hyperparameters (used when DEHB unavailable)
DEFAULT_HP = {
    "actor_lr":  3e-4,
    "critic_lr": 3e-4,
    "alpha":     1.0,    # FIX v4: High initial entropy to escape zero-action local min
    "tau":       0.005,
    "beta":      0.5,    # VAPOR uncertainty bonus weight
}

# ---------------------------------------------------------------------------
# Results directory — FIX: separate from old broken results
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.expanduser("~/quasar_v4_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Helper: create SiliQun environment
# ===========================================================================
def make_env(seed: int, reward_type: str = "shaped"):
    """Create a 3-qubit SiliQun GHZ environment."""
    env = SiliQunEnv(
        n_qubits=3,
        target_state="ghz",
        max_steps=MAX_EP_STEPS,
        reward_type=reward_type,
    )
    env.reset(seed=seed)
    return env


# ===========================================================================
# Helper: evaluate agent
# ===========================================================================
def evaluate_agent(agent, seed: int, n_episodes: int = EVAL_EPISODES):
    """
    Evaluate agent over n_episodes and return mean fidelity.
    Uses deterministic actions (no exploration noise).
    """
    env = make_env(seed + 10000)  # Different seed from training
    fidelities = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_fidelity = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_fidelity = info.get("best_fidelity", info.get("fidelity", 0.0))
        fidelities.append(ep_fidelity)
    env.close()
    return float(np.mean(fidelities))


# ===========================================================================
# DEHB hyperparameter optimization
# ===========================================================================
def run_dehb(seed: int, condition: str):
    """
    Run DEHB to find optimal SAC hyperparameters.
    Returns best HP dict.
    Falls back to DEFAULT_HP if DEHB unavailable.
    """
    if not HAS_DEHB:
        logger.warning("DEHB not available — using default HPs")
        return copy.deepcopy(DEFAULT_HP)

    logger.info(f"  Running DEHB ({DEHB_BUDGET} evals, {DEHB_STEPS} steps each)...")

    cs = _build_dehb_config_space()

    def dehb_objective(config, fidelity, **kwargs):
        hp = {
            "actor_lr":  float(config["actor_lr"]),
            "critic_lr": float(config["critic_lr"]),
            "alpha":     float(config["alpha"]),
            "tau":       float(config["tau"]),
            "beta":      float(config.get("beta", 0.5)),
        }
        use_vapor = "vapor" in condition
        agent = _make_agent(hp, seed, use_vapor=use_vapor)
        env = make_env(seed)

        obs, _ = env.reset()
        steps = 0
        budget_steps = int(fidelity)
        while steps < budget_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))
            if steps >= WARMUP_STEPS:
                agent.update()
            obs = next_obs if not done else env.reset()[0]
            steps += 1

        eval_fidelity = evaluate_agent(agent, seed, n_episodes=10)
        env.close()
        # FIX: DEHB v0.1.0+ expects dict with "fitness" and "cost"
        return {"fitness": -eval_fidelity, "cost": budget_steps}

    if cs is None:
        logger.warning("ConfigSpace unavailable — using default HPs")
        return copy.deepcopy(DEFAULT_HP)
    try:
        optimizer = DEHB(
            f=dehb_objective,
            cs=cs,
            # FIX: min_budget/max_budget renamed to min_fidelity/max_fidelity in v0.1.0+
            min_fidelity=DEHB_STEPS // 4,
            max_fidelity=DEHB_STEPS,
            n_workers=1,
            seed=seed,
        )
        trajectory, runtime, history = optimizer.run(
            fevals=DEHB_BUDGET,
            verbose=False,
        )
        incumbents = optimizer.get_incumbents()
        best_config = incumbents.get("config", incumbents.get("incumbent", None))
        if best_config is None:
            logger.warning("DEHB returned no incumbent — using default HPs")
            return copy.deepcopy(DEFAULT_HP)
        best_hp = {
            "actor_lr":  float(best_config["actor_lr"]),
            "critic_lr": float(best_config["critic_lr"]),
            "alpha":     float(best_config["alpha"]),
            "tau":       float(best_config["tau"]),
            "beta":      float(best_config.get("beta", 0.5)),
        }
        best_fidelity = -incumbents.get("fitness", incumbents.get("cost", 0.0))
        logger.info(f"  DEHB best HP: {best_hp}, fidelity={best_fidelity:.4f}")
        return best_hp
    except Exception as e:
        logger.warning(f"DEHB optimization failed ({e}) — using default HPs")
        return copy.deepcopy(DEFAULT_HP)


def _build_dehb_config_space():
    """Build ConfigSpace for DEHB."""
    try:
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add([
            CSH.UniformFloatHyperparameter("actor_lr",  lower=1e-5, upper=1e-2, log=True),
            CSH.UniformFloatHyperparameter("critic_lr", lower=1e-5, upper=1e-2, log=True),
            CSH.UniformFloatHyperparameter("alpha",     lower=0.01, upper=1.0,  log=True),
            CSH.UniformFloatHyperparameter("tau",       lower=0.001, upper=0.05),
            CSH.UniformFloatHyperparameter("beta",      lower=0.01, upper=2.0,  log=True),
        ])
        return cs
    except ImportError:
        logger.warning("ConfigSpace not available — using default HPs")
        return None


def _make_agent(hp: dict, seed: int, use_vapor: bool = True):
    """Instantiate the appropriate agent."""
    if use_vapor:
        return VAPORSACAgent(
            obs_dim=10, act_dim=10,
            hidden_dim=128,
            actor_lr=hp["actor_lr"],
            critic_lr=hp["critic_lr"],
            alpha=hp["alpha"],
            tau=hp["tau"],
            beta=hp.get("beta", 0.5),
            sigma_min=0.1,       # FIX: Sigma floor to prevent uncertainty collapse
            auto_alpha=True,     # FIX: Automatic entropy tuning
            n_critics=5,
            buffer_size=100_000,
            device="cuda",
            seed=seed,
        )
    else:
        # Standard SAC (twin critics, no VAPOR) — reuse VAPORSACAgent with n_critics=2, beta=0
        return VAPORSACAgent(
            obs_dim=10, act_dim=10,
            hidden_dim=128,
            actor_lr=hp["actor_lr"],
            critic_lr=hp["critic_lr"],
            alpha=hp["alpha"],
            tau=hp["tau"],
            beta=0.0,            # No uncertainty bonus
            sigma_min=0.0,       # No sigma floor for standard SAC
            auto_alpha=True,     # FIX: Still use auto-alpha for standard SAC
            n_critics=2,         # Standard twin critics
            buffer_size=100_000,
            device="cuda",
            seed=seed,
        )


# ===========================================================================
# BRFD reward shaping
# ===========================================================================
def apply_brfd(reward: float, obs, info: dict, brfd_shaper=None):
    """Apply BRFD reward shaping if available."""
    if brfd_shaper is None or not HAS_BRFD:
        return reward
    try:
        return brfd_shaper.shape(reward, obs, info)
    except Exception:
        return reward


# ===========================================================================
# Training functions
# ===========================================================================

def train_standard(condition: str, seed: int, best_hp: dict):
    """
    Train a single agent (no PBT) for TOTAL_STEPS steps.
    Used for: dehb_brfd, dehb_brfd_vapor
    """
    use_vapor = "vapor" in condition
    agent = _make_agent(best_hp, seed, use_vapor=use_vapor)
    env = make_env(seed)

    brfd_shaper = None
    if HAS_BRFD and "brfd" in condition:
        try:
            brfd_shaper = BRFDRewardShaper(seed=seed)
        except Exception:
            pass

    obs, _ = env.reset()
    total_steps = 0
    episode = 0
    episode_reward = 0.0
    fidelity_history = []
    eval_checkpoints = []
    best_eval_fidelity = 0.0
    best_agent_state = None  # best-model checkpoint

    logger.info(f"  Training {condition} seed={seed} for {TOTAL_STEPS} steps...")
    t0 = time.time()

    while total_steps < TOTAL_STEPS:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        shaped_reward = apply_brfd(reward, obs, info, brfd_shaper)
        agent.replay_buffer.push(obs, action, shaped_reward, next_obs, float(done))

        if total_steps >= WARMUP_STEPS and len(agent.replay_buffer) >= agent.batch_size:
            agent.update()

        episode_reward += reward
        obs = next_obs

        if done:
            fidelity = info.get("best_fidelity", info.get("fidelity", 0.0))
            fidelity_history.append(fidelity)
            agent.diagnostics["episode_reward"].append(episode_reward)
            agent.diagnostics["fidelity"].append(fidelity)
            episode += 1
            episode_reward = 0.0
            obs, _ = env.reset()

        # Evaluate every 10,000 steps
        if total_steps > 0 and total_steps % 10_000 == 0:
            eval_fidelity = evaluate_agent(agent, seed)
            eval_checkpoints.append({"step": total_steps, "fidelity": eval_fidelity})
            # FIX: Save best-model checkpoint using agent.critics (ModuleList)
            if eval_fidelity > best_eval_fidelity:
                best_eval_fidelity = eval_fidelity
                try:
                    best_agent_state = {
                        "actor": copy.deepcopy(agent.actor.state_dict()),
                        "critics": [copy.deepcopy(c.state_dict()) for c in agent.critics],
                        "target_critics": [copy.deepcopy(tc.state_dict()) for tc in agent.target_critics],
                        "step": total_steps,
                        "fidelity": eval_fidelity,
                    }
                    logger.info(f"    New best checkpoint at step {total_steps}: {eval_fidelity:.4f}")
                except Exception as e:
                    logger.warning(f"    Failed to save checkpoint: {e}")
            elapsed = time.time() - t0
            logger.info(
                f"    Step {total_steps:6d}/{TOTAL_STEPS} | "
                f"eval_fidelity={eval_fidelity:.4f} | "
                f"best_eval={best_eval_fidelity:.4f} | "
                f"alpha={agent.alpha:.4f} | "
                f"episodes={episode} | elapsed={elapsed:.0f}s"
            )

        total_steps += 1

    env.close()
    # FIX: Restore best-model checkpoint using agent.critics
    if best_agent_state is not None:
        try:
            agent.actor.load_state_dict(best_agent_state["actor"])
            for c, sd in zip(agent.critics, best_agent_state["critics"]):
                c.load_state_dict(sd)
            for tc, sd in zip(agent.target_critics, best_agent_state["target_critics"]):
                tc.load_state_dict(sd)
            logger.info(
                f"  Restored best checkpoint from step {best_agent_state['step']} "
                f"(eval_fidelity={best_agent_state['fidelity']:.4f})"
            )
        except Exception as e:
            logger.warning(f"  Could not restore best checkpoint: {e}")
    final_fidelity = evaluate_agent(agent, seed)
    elapsed = time.time() - t0

    logger.info(
        f"  Done: condition={condition} seed={seed} "
        f"final_fidelity={final_fidelity:.4f} "
        f"best_eval={best_eval_fidelity:.4f} time={elapsed:.0f}s"
    )

    return {
        "condition": condition,
        "seed": seed,
        "final_fidelity": final_fidelity,
        "best_eval_fidelity": best_eval_fidelity,
        "mean_fidelity": float(np.mean(fidelity_history[-100:])) if fidelity_history else 0.0,
        "max_fidelity": float(max(fidelity_history)) if fidelity_history else 0.0,
        "episodes": episode,
        "total_steps": total_steps,
        "time_seconds": elapsed,
        "eval_checkpoints": eval_checkpoints,
        "hp": best_hp,
    }


def train_pbt(condition: str, seed: int, best_hp: dict):
    """
    Train a population of agents with Elite-Preservation PBT.
    Used for: dehb_brfd_vapor_pbt
    """
    use_vapor = "vapor" in condition

    # Initialize population
    pbt = ElitePBTScheduler(
        population_size=PBT_POP_SIZE,
        exploit_top_k=2,
        exploit_bottom_k=2,
        perturb_factor=0.3,
        resample_prob=0.2,
        seed=seed,
    )
    population_hps = pbt.initialize_population(best_hp)
    agents = [_make_agent(hp, seed + i, use_vapor=use_vapor) for i, hp in enumerate(population_hps)]
    envs = [make_env(seed + i) for i in range(PBT_POP_SIZE)]

    brfd_shapers = []
    for i in range(PBT_POP_SIZE):
        shaper = None
        if HAS_BRFD and "brfd" in condition:
            try:
                shaper = BRFDRewardShaper(seed=seed + i)
            except Exception:
                pass
        brfd_shapers.append(shaper)

    # Initialize episode state for each member
    obs_list = [env.reset()[0] for env in envs]
    episode_rewards = [0.0] * PBT_POP_SIZE
    fidelity_histories = [[] for _ in range(PBT_POP_SIZE)]
    episode_counts = [0] * PBT_POP_SIZE

    round_results = []
    t0 = time.time()

    logger.info(
        f"  Training {condition} seed={seed} | "
        f"{PBT_ROUNDS} rounds × {STEPS_PER_ROUND} steps | "
        f"pop={PBT_POP_SIZE}"
    )

    for pbt_round in range(PBT_ROUNDS):
        logger.info(f"  --- PBT Round {pbt_round + 1}/{PBT_ROUNDS} ---")

        # Train each member for STEPS_PER_ROUND steps
        for step in range(STEPS_PER_ROUND):
            global_step = pbt_round * STEPS_PER_ROUND + step

            for i, (agent, env) in enumerate(zip(agents, envs)):
                action = agent.select_action(obs_list[i])
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                shaped_reward = apply_brfd(reward, obs_list[i], info, brfd_shapers[i])
                agent.replay_buffer.push(
                    obs_list[i], action, shaped_reward, next_obs, float(done)
                )

                if global_step >= WARMUP_STEPS and len(agent.replay_buffer) >= agent.batch_size:
                    agent.update()

                episode_rewards[i] += reward
                obs_list[i] = next_obs

                if done:
                    fidelity = info.get("best_fidelity", info.get("fidelity", 0.0))
                    fidelity_histories[i].append(fidelity)
                    agent.diagnostics["episode_reward"].append(episode_rewards[i])
                    agent.diagnostics["fidelity"].append(fidelity)
                    episode_counts[i] += 1
                    episode_rewards[i] = 0.0
                    obs_list[i] = env.reset()[0]

        # Evaluate all members after this round
        round_scores = []
        for i, agent in enumerate(agents):
            score = evaluate_agent(agent, seed + i, n_episodes=10)
            round_scores.append(score)
            hp = agent.get_hyperparameters()
            pbt.record_performance(i, score, hp, agent=agent)

        summary = pbt.get_summary()
        elapsed = time.time() - t0
        logger.info(
            f"  Round {pbt_round + 1} scores: {[f'{s:.4f}' for s in round_scores]} | "
            f"best={summary['best_score']:.4f} | "
            f"all_time_best={summary['all_time_best']:.4f} | "
            f"elapsed={elapsed:.0f}s"
        )

        round_results.append({
            "round": pbt_round + 1,
            "scores": round_scores,
            "best_score": summary["best_score"],
            "all_time_best": summary["all_time_best"],
            "summary": summary,
        })

        # PBT exploit/explore (elite is preserved)
        if pbt_round < PBT_ROUNDS - 1:
            pbt.exploit_and_explore(agents)

    # Restore all-time best to member 0
    best_agent = pbt.restore_all_time_best(agents)

    # Final evaluation with best agent
    final_fidelity = evaluate_agent(best_agent, seed, n_episodes=EVAL_EPISODES)
    elapsed = time.time() - t0

    logger.info(
        f"  Done: condition={condition} seed={seed} "
        f"final_fidelity={final_fidelity:.4f} "
        f"all_time_best={pbt.all_time_best_score:.4f} "
        f"time={elapsed:.0f}s"
    )

    # Collect all fidelity history from all members
    all_fidelities = []
    for fh in fidelity_histories:
        all_fidelities.extend(fh)

    for env in envs:
        env.close()

    return {
        "condition": condition,
        "seed": seed,
        "final_fidelity": final_fidelity,
        "all_time_best_fidelity": pbt.all_time_best_score,
        "mean_fidelity": float(np.mean(all_fidelities[-100:])) if all_fidelities else 0.0,
        "max_fidelity": float(max(all_fidelities)) if all_fidelities else 0.0,
        "episodes": sum(episode_counts),
        "total_steps": PBT_ROUNDS * STEPS_PER_ROUND * PBT_POP_SIZE,
        "time_seconds": elapsed,
        "round_results": round_results,
        "hp": best_hp,
        "pbt_summary": pbt.get_summary(),
    }


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_experiment(condition: str, seed: int):
    """Run a single condition × seed experiment."""
    result_path = os.path.join(
        RESULTS_DIR, f"{condition}_seed{seed}.json"
    )

    # Skip if already completed
    if os.path.exists(result_path):
        logger.info(f"  Skipping {condition} seed={seed} — already done")
        with open(result_path) as f:
            return json.load(f)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Condition: {condition} | Seed: {seed}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # Step 1: DEHB hyperparameter optimization
    best_hp = run_dehb(seed, condition)

    # Step 2: Train
    if "pbt" in condition:
        result = train_pbt(condition, seed, best_hp)
    else:
        result = train_standard(condition, seed, best_hp)

    result["wall_time"] = time.time() - t0

    # Save result
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="QUASAR v3 Experiment — FIXED")
    parser.add_argument("--condition", type=str, default=None,
                        help="Run only this condition (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed (default: all)")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else CONDITIONS
    seeds = [args.seed] if args.seed else SEEDS

    logger.info("=" * 60)
    logger.info("QUASAR v3 Experiment — VAPOR-lite + Elite PBT — FIXED")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total runs: {len(conditions) * len(seeds)}")
    logger.info(f"Steps per run: {TOTAL_STEPS:,}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    all_results = []
    for condition in conditions:
        for seed in seeds:
            try:
                result = run_experiment(condition, seed)
                all_results.append(result)
            except Exception as e:
                logger.error(f"FAILED: {condition} seed={seed}: {e}", exc_info=True)

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Condition':<30} {'Seed':>6} {'Final F':>8} {'Max F':>8} {'Time(min)':>10}")
    logger.info("-" * 60)
    for r in all_results:
        final_f = r.get("final_fidelity", r.get("all_time_best_fidelity", 0.0))
        max_f = r.get("max_fidelity", 0.0)
        t_min = r.get("time_seconds", 0) / 60
        logger.info(
            f"{r['condition']:<30} {r['seed']:>6} "
            f"{final_f:>8.4f} {max_f:>8.4f} {t_min:>10.1f}"
        )

    # Save master summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nMaster summary saved: {summary_path}")


if __name__ == "__main__":
    main()
