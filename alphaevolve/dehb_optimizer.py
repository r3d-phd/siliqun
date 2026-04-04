"""
DEHB optimizer for the AEDB system.

Wraps DEHB (Differential Evolution Hyperband) to optimize ALL
hyperparameters across every component of the AEDB pipeline:

  1. Noise model & gate sequence parameters
  2. AlphaEvolve LLM parameters (temperature, top_p, etc.)
  3. BRFD meta-learning parameters (reward_lr, policy_lr, etc.)
  4. Fitness evaluation parameters (n_samples, timeout, etc.)

DEHB is the master hyperparameter optimizer: it learns the best
configuration for the entire system, not just the gate sequence.

Uses SiliQun's multi-fidelity grid sizes (2q, 3q, 4q) as natural
budget levels for Hyperband's successive halving.

References:
    Awad et al., "DEHB: Evolutionary Hyperband for Scalable, Robust
    and Efficient Hyperparameter Optimization", IJCAI 2021.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple
from pathlib import Path

from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from dehb import DEHB

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitness import FitnessEvaluator, TLFNoiseModel, Gate
from skeleton import StandardDecomposition, Gate

logger = logging.getLogger("aedb.dehb")


# ======================================================================
# Full Cross-Component Configuration Space
# ======================================================================

def build_config_space() -> ConfigurationSpace:
    """Build the FULL hyperparameter search space for DEHB.

    This defines ALL tunable parameters across EVERY component:
      - Noise model & gate sequence
      - AlphaEvolve LLM generation
      - BRFD meta-learning
      - Fitness evaluation

    Returns
    -------
    ConfigurationSpace
        The full cross-component hyperparameter search space.
    """
    cs = ConfigurationSpace(seed=42)

    # ==================================================================
    # Group 1: Noise model parameters
    # ==================================================================
    cs.add(Float(
        "noise_amplitude",
        bounds=(0.01, 1.0),
        default=0.5,
        log=True,
    ))
    cs.add(Float(
        "tlf_correlation_length_nm",
        bounds=(20.0, 300.0),
        default=81.0,
    ))
    cs.add(Float(
        "qubit_spacing_nm",
        bounds=(50.0, 200.0),
        default=108.0,
    ))

    # ==================================================================
    # Group 2: Gate sequence parameters
    # ==================================================================
    cs.add(Float(
        "echo_angle",
        bounds=(0.01, 6.28),
        default=3.14159,
    ))
    cs.add(Integer(
        "n_echo_pairs",
        bounds=(0, 6),
        default=0,
    ))
    cs.add(Float(
        "pre_rotation_angle",
        bounds=(0.0, 6.28),
        default=0.0,
    ))
    cs.add(Categorical(
        "pre_rotation_axis",
        items=["none", "x", "y", "z"],
        default="none",
    ))
    cs.add(Float(
        "post_rotation_angle",
        bounds=(0.0, 6.28),
        default=0.0,
    ))
    cs.add(Categorical(
        "post_rotation_axis",
        items=["none", "x", "y", "z"],
        default="none",
    ))
    cs.add(Categorical(
        "dd_sequence",
        items=["none", "xy4", "cpmg", "uhrig", "knill"],
        default="none",
    ))
    cs.add(Integer(
        "dd_repetitions",
        bounds=(1, 6),
        default=1,
    ))

    # ==================================================================
    # Group 3: AlphaEvolve LLM parameters
    # ==================================================================
    cs.add(Float(
        "llm_temperature",
        bounds=(0.1, 1.5),
        default=0.8,
    ))
    cs.add(Float(
        "llm_top_p",
        bounds=(0.5, 1.0),
        default=0.95,
    ))
    cs.add(Integer(
        "llm_num_predict",
        bounds=(256, 2048),
        default=800,
    ))
    cs.add(Float(
        "llm_repeat_penalty",
        bounds=(1.0, 1.5),
        default=1.1,
    ))
    cs.add(Float(
        "llm_mutation_rate",
        bounds=(0.1, 0.9),
        default=0.5,
    ))
    cs.add(Integer(
        "llm_n_inspirations",
        bounds=(1, 5),
        default=2,
    ))

    # ==================================================================
    # Group 4: BRFD meta-learning parameters
    # ==================================================================
    cs.add(Float(
        "brfd_reward_lr",
        bounds=(1e-5, 1e-1),
        default=1e-3,
        log=True,
    ))
    cs.add(Float(
        "brfd_policy_lr",
        bounds=(1e-4, 0.5),
        default=0.01,
        log=True,
    ))
    cs.add(Integer(
        "brfd_hidden_dim",
        bounds=(16, 128),
        default=32,
    ))
    cs.add(Integer(
        "brfd_inner_episodes",
        bounds=(5, 50),
        default=20,
    ))
    cs.add(Integer(
        "brfd_outer_steps",
        bounds=(3, 30),
        default=10,
    ))
    cs.add(Integer(
        "brfd_max_steps",
        bounds=(4, 20),
        default=6,
    ))
    cs.add(Float(
        "brfd_gamma",
        bounds=(0.9, 1.0),
        default=0.99,
    ))

    # ==================================================================
    # Group 5: Fitness evaluation parameters
    # ==================================================================
    cs.add(Integer(
        "fitness_n_noise_samples",
        bounds=(20, 500),
        default=100,
    ))
    cs.add(Integer(
        "fitness_max_seq_length",
        bounds=(10, 200),
        default=50,
    ))

    return cs


# ======================================================================
# Config Extraction Helpers
# ======================================================================

def extract_noise_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract noise model parameters from a DEHB config."""
    return {
        "noise_amplitude": float(config.get("noise_amplitude", 0.5)),
        "qubit_spacing_nm": float(config.get("qubit_spacing_nm", 108.0)),
        "tlf_correlation_length_nm": float(config.get("tlf_correlation_length_nm", 81.0)),
    }


def extract_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract AlphaEvolve LLM parameters from a DEHB config."""
    return {
        "temperature": float(config.get("llm_temperature", 0.8)),
        "top_p": float(config.get("llm_top_p", 0.95)),
        "num_predict": int(config.get("llm_num_predict", 800)),
        "repeat_penalty": float(config.get("llm_repeat_penalty", 1.1)),
        "mutation_rate": float(config.get("llm_mutation_rate", 0.5)),
        "n_inspirations": int(config.get("llm_n_inspirations", 2)),
    }


def extract_brfd_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract BRFD meta-learning parameters from a DEHB config."""
    return {
        "reward_lr": float(config.get("brfd_reward_lr", 1e-3)),
        "policy_lr": float(config.get("brfd_policy_lr", 0.01)),
        "hidden_dim": int(config.get("brfd_hidden_dim", 32)),
        "inner_episodes": int(config.get("brfd_inner_episodes", 20)),
        "outer_steps": int(config.get("brfd_outer_steps", 10)),
        "max_steps": int(config.get("brfd_max_steps", 6)),
        "gamma": float(config.get("brfd_gamma", 0.99)),
    }


def extract_evolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract AlphaEvolve evolution parameters from a DEHB config.

    These control the evolutionary process itself (mutation rate,
    number of inspirations) as opposed to the LLM generation params.
    """
    return {
        "mutation_rate": float(config.get("llm_mutation_rate", 0.5)),
        "n_inspirations": int(config.get("llm_n_inspirations", 2)),
    }


def extract_fitness_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fitness evaluation parameters from a DEHB config."""
    return {
        "n_noise_samples": int(config.get("fitness_n_noise_samples", 100)),
        "max_seq_length": int(config.get("fitness_max_seq_length", 50)),
    }


# ======================================================================
# Parameterized Gate Sequence Generator
# ======================================================================

def config_to_gate_sequence(
    config: Dict[str, Any],
    target_gate: str,
    n_qubits: int,
) -> list:
    """Convert a DEHB configuration to a gate sequence.

    This is the bridge between DEHB's parameter space and SiliQun's
    gate-level simulation. The configuration specifies what gates
    to prepend/append and what DD sequences to interleave.

    Parameters
    ----------
    config : dict
        DEHB configuration dictionary.
    target_gate : str
        Target operation ("bell", "cnot", "ghz", etc.).
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list of Gate
        The parameterized gate sequence.
    """
    gates = []

    # --- Pre-rotation (optional noise-canceling rotation) ---
    pre_axis = config.get("pre_rotation_axis", "none")
    pre_angle = config.get("pre_rotation_angle", 0.0)
    if pre_axis != "none" and abs(pre_angle) > 0.01:
        gate_type = f"r{pre_axis}"
        for q in range(min(n_qubits, 2)):
            gates.append(Gate(gate_type, [q], {"theta": pre_angle}))

    # --- Core gate sequence (target-dependent) ---
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    elif target_gate == "swap":
        gates.append(Gate("swap", [0, 1]))
    elif target_gate == "h":
        gates.append(Gate("h", [0]))

    # --- Echo pairs (interleaved noise cancellation) ---
    n_echo = int(config.get("n_echo_pairs", 0))
    echo_angle = config.get("echo_angle", np.pi)
    for _ in range(n_echo):
        for q in range(min(n_qubits, 2)):
            gates.append(Gate("rx", [q], {"theta": echo_angle}))
            gates.append(Gate("rx", [q], {"theta": -echo_angle}))

    # --- Dynamical decoupling sequence ---
    dd_type = config.get("dd_sequence", "none")
    dd_reps = int(config.get("dd_repetitions", 1))

    if dd_type == "xy4":
        for _ in range(dd_reps):
            for q in range(min(n_qubits, 2)):
                gates.append(Gate("rx", [q], {"theta": np.pi}))
                gates.append(Gate("ry", [q], {"theta": np.pi}))
                gates.append(Gate("rx", [q], {"theta": np.pi}))
                gates.append(Gate("ry", [q], {"theta": np.pi}))
    elif dd_type == "cpmg":
        for _ in range(dd_reps):
            for q in range(min(n_qubits, 2)):
                gates.append(Gate("rx", [q], {"theta": np.pi}))
                gates.append(Gate("rx", [q], {"theta": np.pi}))
    elif dd_type == "uhrig":
        for k in range(1, dd_reps + 1):
            angle = np.pi * np.sin(np.pi * k / (2 * dd_reps + 2)) ** 2
            for q in range(min(n_qubits, 2)):
                gates.append(Gate("rz", [q], {"theta": angle}))
                gates.append(Gate("rx", [q], {"theta": np.pi}))
    elif dd_type == "knill":
        for _ in range(dd_reps):
            for q in range(min(n_qubits, 2)):
                gates.append(Gate("rx", [q], {"theta": np.pi}))
                gates.append(Gate("ry", [q], {"theta": np.pi}))

    # --- Post-rotation (optional correction) ---
    post_axis = config.get("post_rotation_axis", "none")
    post_angle = config.get("post_rotation_angle", 0.0)
    if post_axis != "none" and abs(post_angle) > 0.01:
        gate_type = f"r{post_axis}"
        for q in range(min(n_qubits, 2)):
            gates.append(Gate(gate_type, [q], {"theta": post_angle}))

    return gates


# ======================================================================
# Multi-Fidelity Objective Function
# ======================================================================

# Budget-to-grid mapping for multi-fidelity evaluation
BUDGET_GRID_MAP = {
    1: {"n_qubits": 2, "n_noise_samples": 30,  "label": "2q-fast"},
    3: {"n_qubits": 3, "n_noise_samples": 80,  "label": "3q-ghz"},
    9: {"n_qubits": 4, "n_noise_samples": 150, "label": "4q-ghz"},
}


def dehb_objective(
    config,
    fidelity: float = 1.0,
    **kwargs,
) -> Dict[str, float]:
    """DEHB objective function with multi-fidelity evaluation.

    This is called by DEHB for each configuration. The fidelity
    (budget) determines the evaluation quality (grid size and noise
    samples). The objective evaluates the FULL config including
    gate sequence, noise model, and implicitly the BRFD/LLM params
    (which are stored for later use by the orchestrator).

    Parameters
    ----------
    config : Configuration or dict
        Full hyperparameter configuration from DEHB.
    fidelity : float
        DEHB fidelity/budget level (1, 3, or 9).

    Returns
    -------
    dict
        Dictionary with "fitness" (to minimize, so we negate fidelity)
        and "cost" (wall time).
    """
    # Convert Configuration object to dict if needed
    if hasattr(config, 'get_dictionary'):
        config = config.get_dictionary()
    elif not isinstance(config, dict):
        config = dict(config)

    budget_int = int(round(fidelity))
    grid_config = BUDGET_GRID_MAP.get(budget_int, BUDGET_GRID_MAP[1])

    n_qubits = grid_config["n_qubits"]
    n_noise_samples = grid_config["n_noise_samples"]

    # Override n_noise_samples from config if budget allows
    cfg_samples = int(config.get("fitness_n_noise_samples", 100))
    # Use min of budget-scaled and config-specified
    n_noise_samples = min(n_noise_samples, cfg_samples)

    # Determine target gates based on grid size
    if n_qubits <= 2:
        target_gates = ["bell", "cnot"]
    else:
        target_gates = ["ghz"]

    # Extract noise parameters
    noise_cfg = extract_noise_config(config)

    # Build evaluator with config-specified noise parameters
    evaluator = FitnessEvaluator(
        n_qubits=n_qubits,
        target_gates=target_gates,
        n_noise_samples=n_noise_samples,
        noise_amplitude=noise_cfg["noise_amplitude"],
        qubit_spacing_nm=noise_cfg["qubit_spacing_nm"],
        tlf_correlation_length_nm=noise_cfg["tlf_correlation_length_nm"],
        max_sequence_length=int(config.get("fitness_max_seq_length", 50)),
        timeout_seconds=10.0,
    )

    # Create a strategy function from the config
    def strategy_fn(target, nq, nn_corr, spacing, corr_len):
        return config_to_gate_sequence(config, target, nq)

    # Evaluate
    start = time.time()
    result = evaluator.evaluate(strategy_fn, seed=42)
    cost = time.time() - start

    # DEHB minimizes, so negate fidelity
    fitness = 1.0 - result["fitness"] if result["valid"] else 1.0

    return {"fitness": float(fitness), "cost": float(cost)}


# ======================================================================
# DEHB Runner
# ======================================================================

class DEHBOptimizer:
    """Wrapper around DEHB for the AEDB system.

    Manages the DEHB instance, full cross-component configuration
    space, and result logging. DEHB learns hyperparameters for ALL
    components: noise model, gate sequence, LLM, BRFD, and fitness.

    Parameters
    ----------
    output_dir : str
        Directory for saving results and checkpoints.
    n_workers : int
        Number of parallel workers (1 for sequential).
    max_evaluations : int
        Maximum number of function evaluations.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        output_dir: str = "results/dehb",
        n_workers: int = 1,
        max_evaluations: int = 100,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_evaluations = max_evaluations
        self.seed = seed

        # Build full cross-component config space
        self.cs = build_config_space()

        self.dehb = DEHB(
            f=dehb_objective,
            cs=self.cs,
            dimensions=len(self.cs.get_hyperparameters()),
            min_fidelity=1,
            max_fidelity=9,
            eta=3,
            n_workers=n_workers,
            seed=seed,
            output_path=str(self.output_dir / "dehb_logs"),
        )

        self.history = []

    def run(self) -> Dict:
        """Run the DEHB optimization.

        Returns
        -------
        dict
            Best configuration and its fitness, including extracted
            sub-configs for each component.
        """
        n_dims = len(self.cs.get_hyperparameters())
        logger.info(
            f"Starting DEHB optimization: {self.max_evaluations} evaluations, "
            f"budgets [1, 3, 9], {n_dims} dimensions (full cross-component)"
        )

        start_time = time.time()

        # Run DEHB
        self.dehb.run(fevals=self.max_evaluations)

        total_time = time.time() - start_time

        # Extract best configuration
        try:
            incumbent_config, incumbent_fitness = self.dehb.get_incumbents()
            best_config = dict(incumbent_config) if incumbent_config is not None else {}
            best_fitness = float(incumbent_fitness) if incumbent_fitness is not None else 1.0
        except Exception:
            best_config = {}
            best_fitness = 1.0

        # Extract sub-configs for each component
        noise_cfg = extract_noise_config(best_config)
        llm_cfg = extract_llm_config(best_config)
        brfd_cfg = extract_brfd_config(best_config)
        fitness_cfg = extract_fitness_config(best_config)

        # Save results
        results = {
            "best_config": best_config,
            "best_fitness": best_fitness,
            "best_fidelity": 1.0 - best_fitness,
            "total_evaluations": self.max_evaluations,
            "total_time_seconds": total_time,
            # Extracted sub-configs for downstream use
            "noise_config": noise_cfg,
            "llm_config": llm_cfg,
            "brfd_config": brfd_cfg,
            "fitness_config": fitness_cfg,
        }

        # Save to file
        results_file = self.output_dir / "dehb_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(
            f"DEHB complete: best fidelity = {results['best_fidelity']:.6f}, "
            f"time = {total_time:.1f}s, dims = {n_dims}"
        )
        logger.info(f"  Noise:   amp={noise_cfg['noise_amplitude']:.4f}, "
                     f"spacing={noise_cfg['qubit_spacing_nm']:.1f}, "
                     f"corr_len={noise_cfg['tlf_correlation_length_nm']:.1f}")
        logger.info(f"  LLM:     temp={llm_cfg['temperature']:.2f}, "
                     f"top_p={llm_cfg['top_p']:.2f}, "
                     f"mut_rate={llm_cfg['mutation_rate']:.2f}")
        logger.info(f"  BRFD:    reward_lr={brfd_cfg['reward_lr']:.5f}, "
                     f"policy_lr={brfd_cfg['policy_lr']:.4f}, "
                     f"outer={brfd_cfg['outer_steps']}")
        logger.info(f"  Fitness: samples={fitness_cfg['n_noise_samples']}, "
                     f"max_seq={fitness_cfg['max_seq_length']}")

        return results

    def get_best_config(self) -> Dict:
        """Get the best configuration found so far.

        Returns
        -------
        dict
            Best hyperparameter configuration.
        """
        try:
            incumbent = self.dehb.get_incumbents()
            if incumbent:
                return dict(incumbent[0])
        except Exception:
            pass
        return {}

    def get_best_sub_configs(self) -> Dict[str, Dict]:
        """Get the best sub-configs for each component.

        Returns
        -------
        dict
            Dictionary with keys 'noise', 'llm', 'brfd', 'fitness',
            each containing the extracted sub-config.
        """
        best = self.get_best_config()
        return {
            "noise": extract_noise_config(best),
            "llm": extract_llm_config(best),
            "brfd": extract_brfd_config(best),
            "fitness": extract_fitness_config(best),
        }


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cs = build_config_space()
    hp_names = [hp.name for hp in cs.get_hyperparameters()]

    print("=" * 70)
    print("DEHB Optimizer - Full Cross-Component Config Space")
    print("=" * 70)
    print(f"Total dimensions: {len(hp_names)}")
    print()
    print("Group 1 - Noise model:")
    for n in hp_names:
        if n.startswith(("noise", "tlf", "qubit")):
            print(f"  {n}")
    print("Group 2 - Gate sequence:")
    for n in hp_names:
        if n.startswith(("echo", "n_echo", "pre_", "post_", "dd_")):
            print(f"  {n}")
    print("Group 3 - AlphaEvolve LLM:")
    for n in hp_names:
        if n.startswith("llm_"):
            print(f"  {n}")
    print("Group 4 - BRFD meta-learning:")
    for n in hp_names:
        if n.startswith("brfd_"):
            print(f"  {n}")
    print("Group 5 - Fitness evaluation:")
    for n in hp_names:
        if n.startswith("fitness_"):
            print(f"  {n}")
    print()

    # Quick test with small budget
    optimizer = DEHBOptimizer(
        output_dir="/home/ubuntu/siliqun/alphaevolve/results/dehb_test",
        max_evaluations=10,
        seed=42,
    )

    results = optimizer.run()

    print(f"\nBest fidelity: {results['best_fidelity']:.6f}")
    print(f"Sub-configs:")
    for group, cfg in [
        ("Noise", results["noise_config"]),
        ("LLM", results["llm_config"]),
        ("BRFD", results["brfd_config"]),
        ("Fitness", results["fitness_config"]),
    ]:
        print(f"  {group}: {cfg}")
    print(f"Total time: {results['total_time_seconds']:.1f}s")
