"""
AlphaEvolve -- Full Implementation.

A faithful implementation of the AlphaEvolve architecture from
Novikov et al., "AlphaEvolve: A Gemini-Powered Coding Agent for
Designing Advanced Algorithms", 2025.

This engine integrates:
  1. **ProgramDatabase** -- MAP-Elites + island-based population model
  2. **PromptSampler** -- rich, stochastic prompts with inspirations
  3. **LLMEnsemble** -- flash/pro model ensemble
  4. **EvaluationCascade** -- multi-stage filtering with multi-metric scores

The controller loop follows the paper's pseudocode (Figure 2):
    parent, inspirations = database.sample()
    prompt = prompt_sampler.build(parent, inspirations)
    diff = llm.generate(prompt)
    child = apply_diff(parent, diff)
    results = evaluator.execute(child)
    database.add(child, results)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from program_database import Program, ProgramDatabase
from prompt_sampler import PromptSampler
from llm_ensemble import LLMEnsemble
from evaluation_cascade import EvaluationCascade, extract_function, compile_strategy

logger = logging.getLogger("aedb.alpha_evolve")


# ──────────────────────────────────────────────────────────────────────
# Seed strategies
# ──────────────────────────────────────────────────────────────────────

SEED_STRATEGIES = {
    "standard": '''\
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Standard textbook decomposition."""
    gates = []
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    return gates
''',

    "echo_cancel": '''\
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Echo-based noise cancellation with correlation-aware timing."""
    import numpy as np
    gates = []
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rx", [q], {"theta": np.pi}))
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rx", [q], {"theta": np.pi}))
    return gates
''',

    "correlation_aware": '''\
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Correlation-aware: use ZZ rotations to exploit spatial noise correlations."""
    import numpy as np
    gates = []
    theta = nn_correlation * np.pi / 4
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rz", [q], {"theta": theta}))
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rz", [q], {"theta": -theta}))
    return gates
''',

    "minimal": '''\
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Minimal gate count -- fewer gates means less accumulated noise."""
    gates = []
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    return gates
''',
}


# ──────────────────────────────────────────────────────────────────────
# AlphaEvolve Engine
# ──────────────────────────────────────────────────────────────────────

class AlphaEvolve:
    """Full AlphaEvolve implementation.

    Parameters
    ----------
    n_generations : int
        Number of evolutionary generations.
    population_per_gen : int
        Number of new candidates generated per generation.
    max_llm_calls : int
        Maximum total LLM API calls.
    noise_amplitude : float
        Noise strength per gate for evaluation.
    n_qubits : int
        Number of qubits.
    n_islands : int
        Number of MAP-Elites islands.
    n_inspirations : int
        Number of inspiration programs per prompt.
    flash_ratio : float
        Fraction of LLM calls routed to flash models.
    diff_probability : float
        Probability of requesting diff-based mutations.
    output_dir : str
        Directory for saving results.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_generations: int = 20,
        population_per_gen: int = 5,
        max_llm_calls: int = 100,
        noise_amplitude: float = 0.5,
        n_qubits: int = 2,
        n_islands: int = 3,
        n_inspirations: int = 2,
        flash_ratio: float = 0.7,
        diff_probability: float = 0.3,
        output_dir: str = "results/alpha_evolve",
        seed: int = 42,
    ):
        self.n_generations = n_generations
        self.population_per_gen = population_per_gen
        self.max_llm_calls = max_llm_calls
        self.noise_amplitude = noise_amplitude
        self.n_qubits = n_qubits
        self.n_inspirations = n_inspirations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Core components
        self.database = ProgramDatabase(
            n_islands=n_islands,
            cell_capacity=3,
            migration_interval=10,
            migration_count=2,
            seed=seed,
        )

        self.prompt_sampler = PromptSampler(
            noise_amplitude=noise_amplitude,
            diff_probability=diff_probability,
            seed=seed,
        )

        self.llm_ensemble = LLMEnsemble(
            flash_ratio=flash_ratio,
            seed=seed,
        )

        self.evaluator = EvaluationCascade(
            n_qubits=n_qubits,
            target_gates=["bell", "cnot"],
            noise_amplitude=noise_amplitude,
            stage_thresholds=[0.0, 0.3, 0.5],
            seed=seed,
        )

        # Tracking
        self.llm_calls = 0
        self.history: List[Dict] = []
        self.start_time = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Seed initialisation
    # ──────────────────────────────────────────────────────────────────

    def _seed_database(self):
        """Evaluate seed strategies and add them to the database."""
        logger.info("Seeding database with initial strategies...")

        for name, code in SEED_STRATEGIES.items():
            scores, fn = self.evaluator.evaluate(code)
            if scores is None:
                logger.warning(f"Seed '{name}' failed evaluation")
                scores = {"combined": 0.0, "base_fidelity": 0.0}

            program = Program(
                code=code,
                scores=scores,
                primary_score=scores.get("combined", 0.0),
                generation=0,
                source=f"seed:{name}",
            )
            self.database.add(program)

            logger.info(
                f"  Seed '{name}': combined={scores.get('combined', 0):.4f}, "
                f"fidelity={scores.get('base_fidelity', 0):.4f}, "
                f"gates={scores.get('gate_count', '?')}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Single evolution step
    # ──────────────────────────────────────────────────────────────────

    def _evolve_one(self, generation: int) -> Optional[Program]:
        """Generate and evaluate one new candidate program.

        This implements the core AlphaEvolve loop (Figure 2 of the paper):
          1. Sample parent + inspirations from database
          2. Build prompt with PromptSampler
          3. Generate code with LLMEnsemble
          4. Apply diff or extract function
          5. Evaluate through cascade
          6. Add to database

        Returns the new Program if successful, None otherwise.
        """
        if self.llm_calls >= self.max_llm_calls:
            return None

        # Step 1: Sample from database
        parent, inspirations, island_id = self.database.sample(
            n_inspirations=self.n_inspirations,
        )

        # Step 2: Build prompt
        system_prompt, user_prompt, is_diff = self.prompt_sampler.build_prompt(
            parent=parent,
            inspirations=inspirations,
            best_program=self.database.best_program,
            generation=generation,
        )

        # Step 3: Call LLM ensemble
        self.llm_calls += 1
        response, model_name = self.llm_ensemble.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
        )

        if response is None:
            logger.debug(f"LLM call {self.llm_calls} returned None")
            return None

        # Step 4: Extract or apply code
        child_code = None

        if is_diff:
            # Try to apply diff to parent code
            child_code = self.prompt_sampler.apply_diff(parent.code, response)
            if child_code is None:
                # Fallback: try extracting as full function
                child_code = extract_function(response)
        else:
            child_code = extract_function(response)

        if child_code is None:
            # Retry with error feedback
            logger.debug("First extraction failed, retrying with feedback")
            self.llm_calls += 1
            retry_prompt = (
                "Your previous response could not be parsed. "
                "Return ONLY the Python function starting with:\n"
                "def generate_gate_sequence(target_gate, n_qubits, "
                "nn_correlation, qubit_spacing_nm, corr_length_nm):\n\n"
                "Do NOT redefine Gate. Use 4-space indentation.\n"
                f"Previous response (truncated):\n{response[:300]}"
            )
            response2, model_name2 = self.llm_ensemble.call(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                max_tokens=1500,
            )
            if response2:
                child_code = extract_function(response2)

        if child_code is None:
            return None

        # Step 5: Evaluate through cascade
        scores, fn = self.evaluator.evaluate(child_code)
        if scores is None:
            return None

        # Step 6: Create program and add to database
        child = Program(
            code=child_code,
            scores=scores,
            primary_score=scores.get("combined", 0.0),
            generation=generation,
            parent_id=parent.uid,
            model=model_name,
            source="evolution",
        )

        self.database.add(child, island_id=island_id)

        return child

    # ──────────────────────────────────────────────────────────────────
    # Main evolution loop
    # ──────────────────────────────────────────────────────────────────

    def evolve(self) -> Dict[str, Any]:
        """Run the full AlphaEvolve evolutionary loop.

        Returns
        -------
        dict
            Results including best strategy, history, and statistics.
        """
        self.start_time = time.time()

        logger.info("=" * 70)
        logger.info("AlphaEvolve -- Full Implementation")
        logger.info("=" * 70)
        logger.info(
            f"Config: gens={self.n_generations}, pop/gen={self.population_per_gen}, "
            f"max_calls={self.max_llm_calls}, noise={self.noise_amplitude}, "
            f"qubits={self.n_qubits}"
        )

        # Seed the database
        self._seed_database()

        best_at_start = self.database.best_program
        logger.info(
            f"Initial best: {best_at_start.primary_score:.4f} "
            f"({best_at_start.source})"
        )

        # Evolution loop
        for gen in range(1, self.n_generations + 1):
            if self.llm_calls >= self.max_llm_calls:
                logger.info(f"LLM budget exhausted at generation {gen}")
                break

            gen_start = time.time()
            new_programs = []
            valid_count = 0

            n_candidates = min(
                self.population_per_gen,
                self.max_llm_calls - self.llm_calls,
            )

            for i in range(n_candidates):
                child = self._evolve_one(generation=gen)
                if child is not None:
                    new_programs.append(child)
                    if child.primary_score > 0:
                        valid_count += 1

            gen_time = time.time() - gen_start
            db_stats = self.database.get_stats()
            best = self.database.best_program

            # Check for new best
            is_new_best = (
                best and best_at_start
                and best.primary_score > best_at_start.primary_score
                and best.generation == gen
            )

            gen_record = {
                "generation": gen,
                "best_score": best.primary_score if best else 0.0,
                "best_fidelity": best.scores.get("base_fidelity", 0.0) if best else 0.0,
                "new_programs": len(new_programs),
                "valid_programs": valid_count,
                "llm_calls": self.llm_calls,
                "db_total": db_stats.get("total", 0),
                "db_cells": db_stats.get("cells", 0),
                "time_seconds": gen_time,
                "is_new_best": is_new_best,
            }
            self.history.append(gen_record)

            # Log
            new_best_marker = "  *** NEW BEST ***" if is_new_best else ""
            logger.info(
                f"Gen {gen}: best={best.primary_score:.4f}, "
                f"fid={best.scores.get('base_fidelity', 0):.4f}, "
                f"valid={valid_count}/{n_candidates}, "
                f"db={db_stats.get('total', 0)} programs "
                f"({db_stats.get('cells', 0)} cells), "
                f"calls={self.llm_calls}/{self.max_llm_calls}, "
                f"time={gen_time:.1f}s{new_best_marker}"
            )

            if is_new_best:
                best_at_start = best

        # Final results
        total_time = time.time() - self.start_time
        best = self.database.best_program
        cascade_stats = self.evaluator.get_stats()
        llm_stats = self.llm_ensemble.get_stats()

        results = {
            "best_score": best.primary_score if best else 0.0,
            "best_fidelity": best.scores.get("base_fidelity", 0.0) if best else 0.0,
            "best_code": best.code if best else "",
            "best_scores": best.scores if best else {},
            "best_model": best.model if best else "",
            "best_generation": best.generation if best else 0,
            "total_llm_calls": self.llm_calls,
            "total_time_seconds": total_time,
            "total_programs": self.database.total_programs,
            "history": self.history,
            "cascade_stats": cascade_stats,
            "llm_stats": llm_stats,
            "database_stats": self.database.get_stats(),
        }

        # Save results
        self._save_results(results)

        logger.info("=" * 70)
        logger.info("AlphaEvolve COMPLETE")
        logger.info(f"  Best score:    {results['best_score']:.4f}")
        logger.info(f"  Best fidelity: {results['best_fidelity']:.4f}")
        logger.info(f"  Best model:    {results['best_model']}")
        logger.info(f"  Best gen:      {results['best_generation']}")
        logger.info(f"  LLM calls:     {results['total_llm_calls']}")
        logger.info(f"  Total time:    {total_time:.1f}s")
        logger.info(f"  Programs:      {results['total_programs']}")
        logger.info("=" * 70)

        return results

    def _save_results(self, results: Dict):
        """Save results to disk."""
        # Best strategy
        best = self.database.best_program
        if best:
            best_file = self.output_dir / "best_strategy.py"
            with open(best_file, "w") as f:
                f.write(f"# Best strategy found by AlphaEvolve\n")
                f.write(f"# Score: {best.primary_score:.6f}\n")
                f.write(f"# Fidelity: {best.scores.get('base_fidelity', 0):.6f}\n")
                f.write(f"# Model: {best.model}\n")
                f.write(f"# Generation: {best.generation}\n\n")
                f.write(f"from fitness import Gate\n\n")
                f.write(best.code)

        # Full results JSON
        results_file = self.output_dir / "alpha_evolve_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # All programs
        all_progs = self.database.get_all_programs()
        programs_file = self.output_dir / "all_programs.json"
        with open(programs_file, "w") as f:
            json.dump(
                [p.to_dict() for p in all_progs],
                f, indent=2, default=str,
            )

        logger.info(f"Results saved to {self.output_dir}")

    # ──────────────────────────────────────────────────────────────────
    # External integration points (for AEDB orchestrator)
    # ──────────────────────────────────────────────────────────────────

    def get_best_strategy(self) -> Optional[Dict]:
        """Return the best strategy for AEDB integration."""
        best = self.database.best_program
        if best is None:
            return None
        return {
            "code": best.code,
            "fitness": best.scores.get("base_fidelity", 0.0),
            "combined_score": best.primary_score,
            "scores": best.scores,
            "model": best.model,
            "generation": best.generation,
        }

    def inject_strategy(
        self,
        code: str,
        scores: Dict[str, float],
        source: str = "external",
    ):
        """Inject an externally-optimised strategy into the database.

        Used by the AEDB orchestrator to feed DEHB/BRFD-optimised
        strategies back into the AlphaEvolve population.
        """
        program = Program(
            code=code,
            scores=scores,
            primary_score=scores.get("combined", scores.get("base_fidelity", 0.0)),
            generation=-1,
            source=source,
        )
        self.database.add(program)
        logger.info(
            f"Injected external strategy: score={program.primary_score:.4f} "
            f"(source={source})"
        )


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    engine = AlphaEvolve(
        n_generations=10,
        population_per_gen=3,
        max_llm_calls=30,
        noise_amplitude=0.5,
        n_qubits=2,
        n_islands=3,
        n_inspirations=2,
        flash_ratio=0.8,
        diff_probability=0.2,
        output_dir="results/alpha_evolve_test",
        seed=42,
    )

    results = engine.evolve()

    print(f"\nBest score: {results['best_score']:.4f}")
    print(f"Best fidelity: {results['best_fidelity']:.4f}")
    print(f"LLM calls: {results['total_llm_calls']}")
