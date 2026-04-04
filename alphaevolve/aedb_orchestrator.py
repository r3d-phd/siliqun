"""
AEDB Orchestrator: AlphaEvolve + DEHB + BRFD integrated tri-level optimisation.

Architecture (three nested optimisation layers):
    AlphaEvolve (outer) -- evolves gate-sequence ALGORITHMS (Python code)
        using MAP-Elites + island model + LLM ensemble + evaluation cascade
    DEHB (middle) -- optimises HYPERPARAMETERS for each algorithm
    BRFD (inner) -- discovers optimal REWARD FUNCTIONS for DRL training

Interaction protocol:
    1. AlphaEvolve generates a candidate gate-sequence algorithm.
    2. The evaluation cascade quickly filters obviously bad candidates.
    3. Promising candidates are refined by DEHB (hyperparameter tuning).
    4. Top candidates are further evaluated by BRFD (reward shaping).
    5. Multi-metric scores feed back into the AlphaEvolve Program Database.
    6. DEHB/BRFD-improved strategies are injected back into the database
       so the LLM can learn from them.

References:
    - AlphaEvolve: Novikov et al., 2025
    - DEHB: Awad et al., IJCAI 2021
    - BRFD: Nature Communications, 2025
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitness import FitnessEvaluator, Gate
from alpha_evolve import AlphaEvolve, SEED_STRATEGIES
from program_database import Program
from evaluation_cascade import compile_strategy
from dehb_optimizer import DEHBOptimizer
from brfd_reward import BRFDTrainer

logger = logging.getLogger("aedb.orchestrator")


class AEDBOrchestrator:
    """Tri-level optimisation orchestrator integrating AlphaEvolve, DEHB, and BRFD.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for evaluation.
    noise_amplitude : float
        Base noise amplitude (rad/gate).
    ae_generations : int
        Number of AlphaEvolve generations.
    ae_population_per_gen : int
        New candidates per AlphaEvolve generation.
    ae_max_llm_calls : int
        Maximum LLM calls for AlphaEvolve.
    ae_n_islands : int
        Number of MAP-Elites islands.
    ae_n_inspirations : int
        Number of inspiration programs per prompt.
    ae_flash_ratio : float
        Fraction of LLM calls routed to flash models.
    dehb_evaluations : int
        Number of DEHB evaluations per top candidate.
    brfd_outer_steps : int
        Number of BRFD outer optimisation steps.
    brfd_inner_episodes : int
        Number of BRFD inner training episodes.
    dehb_top_k : int
        Number of top candidates per generation to refine with DEHB.
    brfd_top_k : int
        Number of top candidates per generation to refine with BRFD.
    output_dir : str
        Directory for saving results.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        noise_amplitude: float = 0.5,
        ae_generations: int = 15,
        ae_population_per_gen: int = 5,
        ae_max_llm_calls: int = 80,
        ae_n_islands: int = 3,
        ae_n_inspirations: int = 2,
        ae_flash_ratio: float = 0.8,
        dehb_evaluations: int = 10,
        brfd_outer_steps: int = 5,
        brfd_inner_episodes: int = 10,
        dehb_top_k: int = 2,
        brfd_top_k: int = 1,
        output_dir: str = "results/aedb",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.noise_amplitude = noise_amplitude
        self.dehb_evaluations = dehb_evaluations
        self.brfd_outer_steps = brfd_outer_steps
        self.brfd_inner_episodes = brfd_inner_episodes
        self.dehb_top_k = dehb_top_k
        self.brfd_top_k = brfd_top_k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Core AlphaEvolve engine (full architecture)
        self.alpha_evolve = AlphaEvolve(
            n_generations=ae_generations,
            population_per_gen=ae_population_per_gen,
            max_llm_calls=ae_max_llm_calls,
            noise_amplitude=noise_amplitude,
            n_qubits=n_qubits,
            n_islands=ae_n_islands,
            n_inspirations=ae_n_inspirations,
            flash_ratio=ae_flash_ratio,
            diff_probability=0.25,
            output_dir=str(self.output_dir / "alpha_evolve"),
            seed=seed,
        )

        # High-fidelity evaluator for final assessment
        self.final_evaluator = FitnessEvaluator(
            n_qubits=n_qubits,
            target_gates=["bell", "cnot"],
            n_noise_samples=500,
            noise_amplitude=noise_amplitude,
            qubit_spacing_nm=108.0,
            tlf_correlation_length_nm=81.0,
            max_sequence_length=50,
            timeout_seconds=10.0,
        )

        # Tracking
        self.total_dehb_evals = 0
        self.total_brfd_steps = 0
        self.history: list = []

    # ──────────────────────────────────────────────────────────────────
    # DEHB refinement
    # ──────────────────────────────────────────────────────────────────

    def _refine_with_dehb(self, program: Program) -> Dict:
        """Refine a strategy with DEHB hyperparameter optimisation.

        Parameters
        ----------
        program : Program
            The program to refine.

        Returns
        -------
        dict
            DEHB results including best fitness and config.
        """
        import hashlib
        code_hash = hashlib.md5(program.code.encode()).hexdigest()[:8]
        dehb_dir = str(
            self.output_dir / f"dehb_{code_hash}_{int(time.time())}"
        )

        try:
            dehb = DEHBOptimizer(
                output_dir=dehb_dir,
                max_evaluations=self.dehb_evaluations,
                seed=self.seed,
            )
            results = dehb.run()
            self.total_dehb_evals += self.dehb_evaluations

            return {
                "dehb_fitness": results["best_fidelity"],
                "dehb_config": results["best_config"],
                "dehb_evals": results["total_evaluations"],
            }
        except Exception as e:
            logger.warning(f"DEHB failed for {program.uid}: {e}")
            return {"dehb_fitness": 0.0, "dehb_config": None, "dehb_evals": 0}

    # ──────────────────────────────────────────────────────────────────
    # BRFD refinement
    # ──────────────────────────────────────────────────────────────────

    def _refine_with_brfd(self, program: Program) -> Dict:
        """Refine a strategy with BRFD reward function discovery.

        Parameters
        ----------
        program : Program
            The program to refine.

        Returns
        -------
        dict
            BRFD results including best fidelity and improvement.
        """
        try:
            brfd = BRFDTrainer(
                n_qubits=self.n_qubits,
                noise_amplitude=self.noise_amplitude,
                n_outer_steps=self.brfd_outer_steps,
                n_inner_episodes=self.brfd_inner_episodes,
                seed=self.seed,
            )
            result = brfd.train()
            self.total_brfd_steps += self.brfd_outer_steps

            return {
                "brfd_fidelity": result["best_fidelity"],
                "brfd_improvement": result.get("improvement", 0.0),
            }
        except Exception as e:
            logger.warning(f"BRFD failed for {program.uid}: {e}")
            return {"brfd_fidelity": 0.0, "brfd_improvement": 0.0}

    # ──────────────────────────────────────────────────────────────────
    # Combined scoring
    # ──────────────────────────────────────────────────────────────────

    def _compute_combined_score(
        self,
        base_scores: Dict[str, float],
        dehb_fitness: float,
        brfd_fidelity: float,
    ) -> Dict[str, float]:
        """Compute combined tri-level scores.

        The combined score weights:
          - 50% AlphaEvolve base fidelity
          - 30% DEHB-optimised fitness
          - 20% BRFD reward fidelity
        """
        base_fid = base_scores.get("base_fidelity", 0.0)
        combined = (
            0.50 * base_fid
            + 0.30 * dehb_fitness
            + 0.20 * brfd_fidelity
        )

        scores = dict(base_scores)
        scores["dehb_fitness"] = dehb_fitness
        scores["brfd_fidelity"] = brfd_fidelity
        scores["combined"] = combined
        return scores

    # ──────────────────────────────────────────────────────────────────
    # Main orchestration loop
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Run the full AEDB tri-level optimisation.

        The orchestration proceeds in four phases:

        Phase 1 -- Seed evaluation with full AEDB pipeline.
        Phase 2 -- AlphaEvolve evolution (MAP-Elites + LLM ensemble).
        Phase 3 -- DEHB/BRFD refinement of top candidates.
        Phase 4 -- Final high-fidelity evaluation.

        Returns
        -------
        dict
            Complete results from all three optimisation levels.
        """
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("AEDB Orchestrator v2: AlphaEvolve + DEHB + BRFD")
        logger.info("=" * 70)
        logger.info(f"  Qubits:        {self.n_qubits}")
        logger.info(f"  Noise:         {self.noise_amplitude} rad/gate")
        logger.info(f"  AE gens:       {self.alpha_evolve.n_generations}")
        logger.info(f"  AE pop/gen:    {self.alpha_evolve.population_per_gen}")
        logger.info(f"  AE max calls:  {self.alpha_evolve.max_llm_calls}")
        logger.info(f"  AE islands:    {len(self.alpha_evolve.database.islands)}")
        logger.info(f"  DEHB evals:    {self.dehb_evaluations} per top-{self.dehb_top_k}")
        logger.info(f"  BRFD steps:    {self.brfd_outer_steps} per top-{self.brfd_top_k}")

        # ============================================================
        # Phase 1: Seed evaluation with DEHB + BRFD
        # ============================================================
        logger.info("\n--- Phase 1: Seed Evaluation (DEHB + BRFD) ---")

        for name, code in SEED_STRATEGIES.items():
            fn = compile_strategy(code)
            if fn is None:
                logger.warning(f"Seed '{name}' failed to compile")
                continue

            # Base evaluation through cascade
            scores, _ = self.alpha_evolve.evaluator.evaluate(code)
            if scores is None:
                scores = {"combined": 0.0, "base_fidelity": 0.0}

            # DEHB refinement
            seed_prog = Program(code=code, scores=scores, source=f"seed:{name}")
            dehb_result = self._refine_with_dehb(seed_prog)

            # BRFD refinement
            brfd_result = self._refine_with_brfd(seed_prog)

            # Combined scoring
            combined_scores = self._compute_combined_score(
                scores,
                dehb_result["dehb_fitness"],
                brfd_result["brfd_fidelity"],
            )

            # Add to AlphaEvolve database with enriched scores
            program = Program(
                code=code,
                scores=combined_scores,
                primary_score=combined_scores["combined"],
                generation=0,
                source=f"seed:{name}",
            )
            self.alpha_evolve.database.add(program)

            logger.info(
                f"  Seed '{name}': fid={scores.get('base_fidelity', 0):.4f}, "
                f"dehb={dehb_result['dehb_fitness']:.4f}, "
                f"brfd={brfd_result['brfd_fidelity']:.4f}, "
                f"combined={combined_scores['combined']:.4f}"
            )

        db_stats = self.alpha_evolve.database.get_stats()
        best = self.alpha_evolve.database.best_program
        logger.info(
            f"  Database: {db_stats['total']} programs, "
            f"best={best.primary_score:.4f}"
        )

        # ============================================================
        # Phase 2: AlphaEvolve evolution
        # ============================================================
        logger.info("\n--- Phase 2: AlphaEvolve Evolution ---")
        logger.info("  (MAP-Elites + LLM ensemble + evaluation cascade)")

        # The AlphaEvolve engine already has the seeded database.
        # We skip its internal seeding and run the evolution loop directly.
        ae_best_before = best.primary_score if best else 0.0

        for gen in range(1, self.alpha_evolve.n_generations + 1):
            if self.alpha_evolve.llm_calls >= self.alpha_evolve.max_llm_calls:
                logger.info(f"  LLM budget exhausted at gen {gen}")
                break

            gen_start = time.time()
            new_programs = []
            valid_count = 0

            n_candidates = min(
                self.alpha_evolve.population_per_gen,
                self.alpha_evolve.max_llm_calls - self.alpha_evolve.llm_calls,
            )

            for _ in range(n_candidates):
                child = self.alpha_evolve._evolve_one(generation=gen)
                if child is not None:
                    new_programs.append(child)
                    if child.primary_score > 0:
                        valid_count += 1

            # DEHB/BRFD refinement for top-k new candidates
            if new_programs:
                new_programs.sort(
                    key=lambda p: p.primary_score, reverse=True
                )

                for prog in new_programs[:self.dehb_top_k]:
                    dehb_result = self._refine_with_dehb(prog)
                    brfd_result = self._refine_with_brfd(prog)

                    combined_scores = self._compute_combined_score(
                        prog.scores,
                        dehb_result["dehb_fitness"],
                        brfd_result["brfd_fidelity"],
                    )
                    prog.scores = combined_scores
                    prog.primary_score = combined_scores["combined"]

                    # Re-add with updated scores
                    self.alpha_evolve.database.add(prog)

            gen_time = time.time() - gen_start
            db_stats = self.alpha_evolve.database.get_stats()
            best = self.alpha_evolve.database.best_program

            is_new_best = (
                best and best.primary_score > ae_best_before
            )
            if is_new_best:
                ae_best_before = best.primary_score

            gen_record = {
                "generation": gen,
                "best_score": best.primary_score if best else 0.0,
                "best_fidelity": best.scores.get("base_fidelity", 0) if best else 0.0,
                "valid": valid_count,
                "total": n_candidates,
                "llm_calls": self.alpha_evolve.llm_calls,
                "dehb_evals": self.total_dehb_evals,
                "brfd_steps": self.total_brfd_steps,
                "db_programs": db_stats.get("total", 0),
                "db_cells": db_stats.get("cells", 0),
                "time_seconds": gen_time,
            }
            self.history.append(gen_record)

            new_best_marker = "  *** NEW BEST ***" if is_new_best else ""
            logger.info(
                f"  Gen {gen}: best={best.primary_score:.4f}, "
                f"fid={best.scores.get('base_fidelity', 0):.4f}, "
                f"valid={valid_count}/{n_candidates}, "
                f"db={db_stats.get('total', 0)} ({db_stats.get('cells', 0)} cells), "
                f"LLM={self.alpha_evolve.llm_calls}, "
                f"DEHB={self.total_dehb_evals}, "
                f"BRFD={self.total_brfd_steps}, "
                f"time={gen_time:.1f}s{new_best_marker}"
            )

        # ============================================================
        # Phase 3: Final DEHB+BRFD refinement of global top candidates
        # ============================================================
        logger.info("\n--- Phase 3: Final Refinement ---")

        all_programs = self.alpha_evolve.database.get_all_programs()
        all_programs.sort(key=lambda p: p.primary_score, reverse=True)

        top_final = min(3, len(all_programs))
        for i, prog in enumerate(all_programs[:top_final]):
            logger.info(
                f"  Refining top-{i+1}: score={prog.primary_score:.4f} "
                f"(source={prog.source})"
            )
            dehb_result = self._refine_with_dehb(prog)
            brfd_result = self._refine_with_brfd(prog)

            combined_scores = self._compute_combined_score(
                prog.scores,
                dehb_result["dehb_fitness"],
                brfd_result["brfd_fidelity"],
            )
            prog.scores = combined_scores
            prog.primary_score = combined_scores["combined"]
            self.alpha_evolve.database.add(prog)

            logger.info(
                f"    -> refined score={prog.primary_score:.4f} "
                f"(dehb={dehb_result['dehb_fitness']:.4f}, "
                f"brfd={brfd_result['brfd_fidelity']:.4f})"
            )

        # ============================================================
        # Phase 4: Final high-fidelity evaluation
        # ============================================================
        logger.info("\n--- Phase 4: Final Evaluation (500 samples) ---")

        best = self.alpha_evolve.database.best_program
        fn = compile_strategy(best.code) if best else None
        if fn:
            final_result = self.final_evaluator.evaluate(fn, seed=self.seed)
            final_fitness = (
                final_result["fitness"] if final_result["valid"] else 0.0
            )
        else:
            final_fitness = best.primary_score if best else 0.0

        total_time = time.time() - start_time

        # ============================================================
        # Save results
        # ============================================================
        results = {
            "best_combined_score": float(best.primary_score) if best else 0.0,
            "best_base_fidelity": float(
                best.scores.get("base_fidelity", 0.0)
            ) if best else 0.0,
            "final_fidelity_500": float(final_fitness),
            "best_code": best.code if best else "",
            "best_scores": best.scores if best else {},
            "best_model": best.model if best else "",
            "best_generation": best.generation if best else 0,
            "best_source": best.source if best else "",
            "total_llm_calls": self.alpha_evolve.llm_calls,
            "total_dehb_evals": self.total_dehb_evals,
            "total_brfd_steps": self.total_brfd_steps,
            "total_time_seconds": total_time,
            "total_programs": self.alpha_evolve.database.total_programs,
            "noise_amplitude": self.noise_amplitude,
            "n_qubits": self.n_qubits,
            "history": self.history,
            "cascade_stats": self.alpha_evolve.evaluator.get_stats(),
            "llm_stats": self.alpha_evolve.llm_ensemble.get_stats(),
            "database_stats": self.alpha_evolve.database.get_stats(),
        }

        # Save best strategy
        if best:
            best_file = self.output_dir / "aedb_best_strategy.py"
            with open(best_file, "w") as f:
                f.write("# Best strategy found by AEDB Orchestrator v2\n")
                f.write(f"# Combined score: {best.primary_score:.6f}\n")
                f.write(f"# Base fidelity:  {best.scores.get('base_fidelity', 0):.6f}\n")
                f.write(f"# Final (500):    {final_fitness:.6f}\n")
                f.write(f"# Model:          {best.model}\n")
                f.write(f"# Generation:     {best.generation}\n")
                f.write(f"# Source:         {best.source}\n")
                f.write(f"# LLM calls:     {self.alpha_evolve.llm_calls}\n")
                f.write(f"# DEHB evals:    {self.total_dehb_evals}\n")
                f.write(f"# BRFD steps:    {self.total_brfd_steps}\n")
                f.write(f"# Time:          {total_time:.1f}s\n\n")
                f.write("from fitness import Gate\n\n")
                f.write(best.code)

        # Save full results
        results_file = self.output_dir / "aedb_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save history
        history_file = self.output_dir / "aedb_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"\n{'=' * 70}")
        logger.info("AEDB v2 COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Best combined:   {best.primary_score:.4f}" if best else "  No best")
        logger.info(f"  Best fidelity:   {best.scores.get('base_fidelity', 0):.4f}" if best else "")
        logger.info(f"  Final (500):     {final_fitness:.4f}")
        logger.info(f"  Total programs:  {self.alpha_evolve.database.total_programs}")
        logger.info(f"  LLM calls:       {self.alpha_evolve.llm_calls}")
        logger.info(f"  DEHB evals:      {self.total_dehb_evals}")
        logger.info(f"  BRFD steps:      {self.total_brfd_steps}")
        logger.info(f"  Total time:      {total_time:.1f}s")
        logger.info(f"{'=' * 70}")

        return results


# ======================================================================
# Standalone execution
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("AEDB v2: AlphaEvolve + DEHB + BRFD Orchestrator")
    print("=" * 70)
    print(f"Architecture: MAP-Elites + LLM Ensemble + Evaluation Cascade")
    print(f"Noise: 0.5 rad/gate (hard regime)")
    print(f"Target: Discover novel noise mitigation strategies")
    print()

    orchestrator = AEDBOrchestrator(
        n_qubits=2,
        noise_amplitude=0.5,
        ae_generations=10,
        ae_population_per_gen=3,
        ae_max_llm_calls=40,
        ae_n_islands=3,
        ae_n_inspirations=2,
        ae_flash_ratio=0.8,
        dehb_evaluations=8,
        brfd_outer_steps=4,
        brfd_inner_episodes=8,
        dehb_top_k=2,
        brfd_top_k=1,
        output_dir="/home/ubuntu/siliqun/alphaevolve/results/aedb_v2",
        seed=42,
    )

    results = orchestrator.run()

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Best combined score: {results['best_combined_score']:.4f}")
    print(f"Best base fidelity:  {results['best_base_fidelity']:.4f}")
    print(f"Final fidelity (500): {results['final_fidelity_500']:.4f}")
    print(f"Total programs:      {results['total_programs']}")
    print(f"LLM calls:           {results['total_llm_calls']}")
    print(f"DEHB evals:          {results['total_dehb_evals']}")
    print(f"BRFD steps:          {results['total_brfd_steps']}")
    print(f"Time:                {results['total_time_seconds']:.1f}s")
    print(f"\nBest strategy:")
    print(results['best_code'])
