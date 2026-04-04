"""
DEHB optimizer for the AEDB system.

Wraps DEHB (Differential Evolution Hyperband) to optimize all
continuous and discrete hyperparameters of the noise mitigation
system. Uses SiliQun's multi-fidelity grid sizes (3x3, 4x4, 5x5)
as natural budget levels.

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
# Configuration Space Definition
# ======================================================================

def build_config_space() -> ConfigurationSpace:
    """Build the hyperparameter search space for DEHB.

    This defines all tunable parameters across the noise model,
    gate sequence, and evaluation settings. Following the project
    principle: DEHB learns ALL hyperparameters.

    Returns
    -------
    ConfigurationSpace
        The full hyperparameter search space.
    """
    cs = ConfigurationSpace(seed=42)

    # --- Noise model parameters ---
    cs.add(Float(
        "noise_amplitude",
        bounds=(0.01, 0.8),
        default=0.05,
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

    # --- Gate sequence parameters ---
    cs.add(Float(
        "echo_angle",
        bounds=(0.01, 6.28),
        default=3.14159,
    ))
    cs.add(Integer(
        "n_echo_pairs",
        bounds=(0, 8),
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

    # --- Dynamical decoupling parameters ---
    cs.add(Categorical(
        "dd_sequence",
        items=["none", "xy4", "cpmg", "uhrig", "knill"],
        default="none",
    ))
    cs.add(Integer(
        "dd_repetitions",
        bounds=(1, 10),
        default=1,
    ))

    return cs


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
    1: {"n_qubits": 2, "n_noise_samples": 50, "label": "2q-fast"},
    3: {"n_qubits": 2, "n_noise_samples": 200, "label": "2q-accurate"},
    9: {"n_qubits": 4, "n_noise_samples": 100, "label": "4q-ghz"},
}


def dehb_objective(
    config,
    fidelity: float = 1.0,
    **kwargs,
) -> Dict[str, float]:
    """DEHB objective function with multi-fidelity evaluation.

    This is called by DEHB for each configuration. The fidelity
    (budget) determines the evaluation quality (grid size and noise samples).

    Parameters
    ----------
    config : Configuration or dict
        Hyperparameter configuration from DEHB.
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

    # Determine target gates based on grid size
    if n_qubits <= 2:
        target_gates = ["bell", "cnot"]
    else:
        target_gates = ["ghz"]

    # Build evaluator with config-specified noise parameters
    evaluator = FitnessEvaluator(
        n_qubits=n_qubits,
        target_gates=target_gates,
        n_noise_samples=n_noise_samples,
        noise_amplitude=float(config.get("noise_amplitude", 0.05)),
        qubit_spacing_nm=float(config.get("qubit_spacing_nm", 108.0)),
        tlf_correlation_length_nm=float(config.get("tlf_correlation_length_nm", 81.0)),
        max_sequence_length=100,
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

    Manages the DEHB instance, configuration space, and result logging.

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

        # Build config space
        self.cs = build_config_space()

        # Initialize DEHB
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
            Best configuration and its fitness.
        """
        logger.info(
            f"Starting DEHB optimization: {self.max_evaluations} evaluations, "
            f"budgets [1, 3, 9], {len(self.cs.get_hyperparameters())} dimensions"
        )

        start_time = time.time()

        # Run DEHB (v0.1.2+ API)
        self.dehb.run(
            fevals=self.max_evaluations,
        )

        total_time = time.time() - start_time

        # Extract best configuration
        try:
            incumbent_config, incumbent_fitness = self.dehb.get_incumbents()
            best_config = dict(incumbent_config) if incumbent_config is not None else {}
            best_fitness = float(incumbent_fitness) if incumbent_fitness is not None else 1.0
        except Exception:
            best_config = {}
            best_fitness = 1.0

        # Save results
        results = {
            "best_config": best_config,
            "best_fitness": best_fitness,
            "best_fidelity": 1.0 - best_fitness,
            "total_evaluations": self.max_evaluations,
            "total_time_seconds": total_time,
        }

        # Save to file
        results_file = self.output_dir / "dehb_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(
            f"DEHB complete: best fidelity = {results['best_fidelity']:.6f}, "
            f"time = {total_time:.1f}s"
        )

        return results

    def get_best_config(self) -> Dict:
        """Get the best configuration found so far.

        Returns
        -------
        dict
            Best hyperparameter configuration.
        """
        incumbent = self.dehb.get_incumbents()
        if incumbent:
            return dict(incumbent[0])
        return {}


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("DEHB Optimizer - Standalone Test")
    print("=" * 70)
    print(f"Config space: {len(build_config_space().get_hyperparameters())} dimensions")
    print(f"Budget levels: 1 (2q-fast), 3 (2q-accurate), 9 (4q-ghz)")
    print()

    # Quick test with small budget
    optimizer = DEHBOptimizer(
        output_dir="/home/ubuntu/siliqun/alphaevolve/results/dehb_test",
        max_evaluations=30,
        seed=42,
    )

    results = optimizer.run()

    print(f"\nBest fidelity: {results['best_fidelity']:.6f}")
    print(f"Best config:")
    for k, v in results.get("best_config", {}).items():
        print(f"  {k}: {v}")
    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print(f"Total evaluations: {results['total_evaluations']}")
