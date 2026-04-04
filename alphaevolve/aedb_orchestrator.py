"""
AEDB Orchestrator v3: AlphaEvolve + DEHB + BRFD integrated tri-level optimisation.

Architecture (three nested optimisation layers):
    DEHB (meta-level) -- learns ALL hyperparameters for every component
    AlphaEvolve (outer) -- evolves gate-sequence ALGORITHMS (Python code)
        using MAP-Elites + island model + LLM ensemble + evaluation cascade
    BRFD (inner) -- discovers optimal REWARD FUNCTIONS for DRL training

Key design principle: DEHB is the MASTER hyperparameter optimizer.
It learns configurations for:
  - Noise model parameters (amplitude, spacing, correlation)
  - Gate sequence parameters (echo, DD, rotations)
  - AlphaEvolve LLM parameters (temperature, top_p, etc.)
  - BRFD meta-learning parameters (learning rates, architecture, etc.)
  - Fitness evaluation parameters (samples, timeout)

Interaction protocol:
    1. DEHB runs first to discover optimal hyperparameters for all components.
    2. AlphaEvolve uses DEHB-learned LLM configs (temperature, etc.).
    3. Each evolved strategy is refined by BRFD with DEHB-learned BRFD configs.
    4. DEHB-optimised gate sequences are injected back into AlphaEvolve's database.
    5. Periodic DEHB re-optimisation adapts configs as the population evolves.

References:
    - AlphaEvolve: Novikov et al., 2025
    - DEHB: Awad et al., IJCAI 2021
    - BRFD: Zheng et al., Nature Communications, 2026
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitness import FitnessEvaluator, Gate
from alpha_evolve import AlphaEvolve, SEED_STRATEGIES
from program_database import Program
from evaluation_cascade import compile_strategy
from dehb_optimizer import (
    DEHBOptimizer,
    extract_noise_config,
    extract_llm_config,
    extract_brfd_config,
    extract_fitness_config,
    config_to_gate_sequence,
)
from brfd_reward import BRFDTrainer

logger = logging.getLogger("aedb.orchestrator")


class AEDBOrchestrator:
    """Tri-level optimisation orchestrator: DEHB (meta) + AlphaEvolve (outer) + BRFD (inner).

    DEHB is the master hyperparameter optimizer that learns configurations
    for ALL components. AlphaEvolve evolves gate-sequence algorithms using
    LLM-guided code generation. BRFD discovers optimal reward functions
    for DRL-based gate selection.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for evaluation (3 for GHZ).
    noise_amplitude : float
        Base noise amplitude (rad/gate).
    target_gates : list of str
        Target quantum operations to evaluate.
    ae_generations : int
        Number of AlphaEvolve generations.
    ae_population_per_gen : int
        New candidates per AlphaEvolve generation.
    ae_max_llm_calls : int
        Maximum LLM calls for AlphaEvolve.
    ae_n_islands : int
        Number of MAP-Elites islands.
    dehb_evaluations : int
        Number of DEHB evaluations (meta-level).
    dehb_reoptimize_interval : int
        Re-run DEHB every N generations to adapt configs.
    brfd_top_k : int
        Number of top candidates per generation to refine with BRFD.
    output_dir : str
        Directory for saving results.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        noise_amplitude: float = 0.5,
        target_gates: Optional[List[str]] = None,
        ae_generations: int = 20,
        ae_population_per_gen: int = 5,
        ae_max_llm_calls: int = 120,
        ae_n_islands: int = 3,
        dehb_evaluations: int = 30,
        dehb_reoptimize_interval: int = 10,
        brfd_top_k: int = 2,
        output_dir: str = "results/aedb_v3",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.noise_amplitude = noise_amplitude
        self.target_gates = target_gates or (
            ["ghz"] if n_qubits >= 3 else ["bell", "cnot"]
        )
        self.ae_generations = ae_generations
        self.ae_population_per_gen = ae_population_per_gen
        self.ae_max_llm_calls = ae_max_llm_calls
        self.ae_n_islands = ae_n_islands
        self.dehb_evaluations = dehb_evaluations
        self.dehb_reoptimize_interval = dehb_reoptimize_interval
        self.brfd_top_k = brfd_top_k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # DEHB-learned configs (populated in Phase 1)
        self.dehb_config: Dict[str, Any] = {}
        self.noise_config: Dict[str, Any] = {}
        self.llm_config: Dict[str, Any] = {}
        self.brfd_config: Dict[str, Any] = {}
        self.fitness_config: Dict[str, Any] = {}

        # AlphaEvolve engine (created after DEHB provides configs)
        self.alpha_evolve: Optional[AlphaEvolve] = None

        # Tracking
        self.total_dehb_evals = 0
        self.total_brfd_steps = 0
        self.total_brfd_outer = 0
        self.history: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────
    # Phase 1: DEHB meta-level optimisation
    # ──────────────────────────────────────────────────────────────────

    def _run_dehb(self, label: str = "initial") -> Dict:
        """Run DEHB to learn optimal hyperparameters for ALL components.

        Returns
        -------
        dict
            DEHB results with extracted sub-configs.
        """
        dehb_dir = str(self.output_dir / f"dehb_{label}_{int(time.time())}")

        logger.info(f"  Running DEHB ({label}): {self.dehb_evaluations} evaluations...")

        dehb = DEHBOptimizer(
            output_dir=dehb_dir,
            max_evaluations=self.dehb_evaluations,
            seed=self.seed,
        )

        results = dehb.run()
        self.total_dehb_evals += self.dehb_evaluations

        # Extract and store sub-configs
        self.dehb_config = results.get("best_config", {})
        self.noise_config = results.get("noise_config", extract_noise_config(self.dehb_config))
        self.llm_config = results.get("llm_config", extract_llm_config(self.dehb_config))
        self.brfd_config = results.get("brfd_config", extract_brfd_config(self.dehb_config))
        self.fitness_config = results.get("fitness_config", extract_fitness_config(self.dehb_config))

        logger.info(f"  DEHB best fidelity: {results.get('best_fidelity', 0):.4f}")
        logger.info(f"    Noise:  amp={self.noise_config.get('noise_amplitude', 0):.4f}, "
                     f"spacing={self.noise_config.get('qubit_spacing_nm', 0):.1f}")
        logger.info(f"    LLM:    temp={self.llm_config.get('temperature', 0):.2f}, "
                     f"top_p={self.llm_config.get('top_p', 0):.2f}, "
                     f"mut_rate={self.llm_config.get('mutation_rate', 0):.2f}")
        logger.info(f"    BRFD:   reward_lr={self.brfd_config.get('reward_lr', 0):.5f}, "
                     f"policy_lr={self.brfd_config.get('policy_lr', 0):.4f}, "
                     f"outer={self.brfd_config.get('outer_steps', 0)}, "
                     f"hidden={self.brfd_config.get('hidden_dim', 0)}")
        logger.info(f"    Fitness: samples={self.fitness_config.get('n_noise_samples', 0)}, "
                     f"max_seq={self.fitness_config.get('max_seq_length', 0)}")

        return results

    # ──────────────────────────────────────────────────────────────────
    # Create AlphaEvolve with DEHB-learned configs
    # ──────────────────────────────────────────────────────────────────

    def _create_alpha_evolve(self):
        """Create AlphaEvolve engine using DEHB-learned LLM configs."""
        self.alpha_evolve = AlphaEvolve(
            n_generations=self.ae_generations,
            population_per_gen=self.ae_population_per_gen,
            max_llm_calls=self.ae_max_llm_calls,
            noise_amplitude=self.noise_amplitude,
            n_qubits=self.n_qubits,
            n_islands=self.ae_n_islands,
            n_inspirations=self.llm_config.get("n_inspirations", 2),
            flash_ratio=0.8,
            diff_probability=0.0,  # full rewrite for local models
            output_dir=str(self.output_dir / "alpha_evolve"),
            seed=self.seed,
        )

        # Update evaluator target gates for GHZ
        self.alpha_evolve.evaluator = self._create_evaluator(
            n_samples=self.fitness_config.get("n_noise_samples", 100),
        )

        # Update LLM ensemble with DEHB-learned params
        self.alpha_evolve.llm_ensemble.default_temperature = self.llm_config.get("temperature", 0.8)
        self.alpha_evolve.llm_ensemble.default_top_p = self.llm_config.get("top_p", 0.95)
        self.alpha_evolve.llm_ensemble.default_num_predict = self.llm_config.get("num_predict", 800)
        self.alpha_evolve.llm_ensemble.default_repeat_penalty = self.llm_config.get("repeat_penalty", 1.1)

    def _create_evaluator(self, n_samples: int = 100):
        """Create a fitness evaluator with current configs."""
        from evaluation_cascade import EvaluationCascade
        return EvaluationCascade(
            n_qubits=self.n_qubits,
            target_gates=self.target_gates,
            noise_amplitude=self.noise_amplitude,
            stage_thresholds=[0.0, 0.2, 0.4],
            seed=self.seed,
        )

    # ──────────────────────────────────────────────────────────────────
    # BRFD refinement with DEHB-learned configs
    # ──────────────────────────────────────────────────────────────────

    def _refine_with_brfd(self, program: Program) -> Dict:
        """Refine a strategy with BRFD using DEHB-learned hyperparameters.

        Parameters
        ----------
        program : Program
            The program to refine.

        Returns
        -------
        dict
            BRFD results including best fidelity.
        """
        try:
            target = self.target_gates[0] if self.target_gates else "ghz"

            brfd = BRFDTrainer(
                n_qubits=self.n_qubits,
                target_gate=target,
                noise_amplitude=self.noise_amplitude,
                dehb_config=self.dehb_config,  # DEHB drives ALL BRFD params
                seed=self.seed,
            )

            result = brfd.train()
            outer_steps = self.brfd_config.get("outer_steps", 10)
            self.total_brfd_steps += outer_steps * self.brfd_config.get("inner_episodes", 20)
            self.total_brfd_outer += outer_steps

            return {
                "brfd_fidelity": result["best_fidelity"],
                "brfd_mean_fidelity": result.get("final_mean_fidelity", 0.0),
                "brfd_config": result.get("config", {}),
            }
        except Exception as e:
            logger.warning(f"BRFD failed for {program.uid}: {e}")
            return {"brfd_fidelity": 0.0, "brfd_mean_fidelity": 0.0}

    # ──────────────────────────────────────────────────────────────────
    # Inject DEHB-optimised gate sequence into AlphaEvolve
    # ──────────────────────────────────────────────────────────────────

    def _inject_dehb_strategy(self):
        """Convert DEHB's best gate sequence config into a strategy and
        inject it into AlphaEvolve's database."""
        if not self.dehb_config or self.alpha_evolve is None:
            return

        # Generate the gate sequence code from DEHB config
        target = self.target_gates[0] if self.target_gates else "ghz"
        gates = config_to_gate_sequence(self.dehb_config, target, self.n_qubits)

        # Convert to Python code
        gate_lines = []
        for g in gates:
            if g.params:
                gate_lines.append(
                    f'    gates.append(Gate("{g.gate_type}", {g.qubits}, {g.params}))'
                )
            else:
                gate_lines.append(
                    f'    gates.append(Gate("{g.gate_type}", {g.qubits}))'
                )

        code = (
            'def generate_gate_sequence(target_gate, n_qubits, nn_correlation, '
            'qubit_spacing_nm, corr_length_nm):\n'
            '    """DEHB-optimised gate sequence."""\n'
            '    gates = []\n'
        )
        if gate_lines:
            code += '\n'.join(gate_lines) + '\n'
        code += '    return gates\n'

        # Evaluate and inject
        scores, fn = self.alpha_evolve.evaluator.evaluate(code)
        if scores is not None:
            scores["source"] = "dehb_optimised"
            self.alpha_evolve.inject_strategy(
                code=code,
                scores=scores,
                source="dehb_optimised",
            )
            logger.info(
                f"  Injected DEHB strategy: fid={scores.get('base_fidelity', 0):.4f}, "
                f"combined={scores.get('combined', 0):.4f}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Combined scoring
    # ──────────────────────────────────────────────────────────────────

    def _compute_combined_score(
        self,
        base_scores: Dict[str, float],
        dehb_fidelity: float,
        brfd_fidelity: float,
    ) -> Dict[str, float]:
        """Compute combined tri-level scores.

        Weights:
          - 50% AlphaEvolve base fidelity
          - 30% DEHB-optimised fitness
          - 20% BRFD reward fidelity
        """
        base_fid = base_scores.get("base_fidelity", 0.0)
        combined = (
            0.50 * base_fid
            + 0.30 * dehb_fidelity
            + 0.20 * brfd_fidelity
        )

        scores = dict(base_scores)
        scores["dehb_fidelity"] = dehb_fidelity
        scores["brfd_fidelity"] = brfd_fidelity
        scores["combined"] = combined
        return scores

    # ──────────────────────────────────────────────────────────────────
    # Main orchestration loop
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Run the full AEDB tri-level optimisation.

        Phase 1 -- DEHB meta-optimisation (learn ALL hyperparameters).
        Phase 2 -- Seed evaluation with DEHB configs + BRFD refinement.
        Phase 3 -- AlphaEvolve evolution with periodic DEHB re-optimisation.
        Phase 4 -- Final BRFD refinement of top candidates.
        Phase 5 -- Final high-fidelity evaluation.

        Returns
        -------
        dict
            Complete results from all three optimisation levels.
        """
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("AEDB Orchestrator v3: DEHB (meta) + AlphaEvolve (outer) + BRFD (inner)")
        logger.info("=" * 70)
        logger.info(f"  Qubits:          {self.n_qubits}")
        logger.info(f"  Target gates:    {self.target_gates}")
        logger.info(f"  Noise:           {self.noise_amplitude} rad/gate")
        logger.info(f"  AE generations:  {self.ae_generations}")
        logger.info(f"  AE pop/gen:      {self.ae_population_per_gen}")
        logger.info(f"  AE max LLM:      {self.ae_max_llm_calls}")
        logger.info(f"  DEHB evals:      {self.dehb_evaluations}")
        logger.info(f"  DEHB re-opt:     every {self.dehb_reoptimize_interval} gens")
        logger.info(f"  BRFD top-k:      {self.brfd_top_k}")

        # ============================================================
        # Phase 1: DEHB meta-level optimisation
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1: DEHB Meta-Level Optimisation")
        logger.info("  Learning optimal hyperparameters for ALL components...")
        logger.info("=" * 70)

        phase1_start = time.time()
        dehb_results = self._run_dehb(label="initial")
        phase1_time = time.time() - phase1_start

        logger.info(f"  Phase 1 complete: {phase1_time:.1f}s, "
                     f"{self.total_dehb_evals} DEHB evals")

        # ============================================================
        # Phase 2: Create AlphaEvolve + seed evaluation with BRFD
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 2: Seed Evaluation (DEHB configs + BRFD)")
        logger.info("=" * 70)

        phase2_start = time.time()

        # Create AlphaEvolve with DEHB-learned configs
        self._create_alpha_evolve()

        # Inject DEHB-optimised gate sequence
        self._inject_dehb_strategy()

        # Evaluate seeds with BRFD refinement
        dehb_fidelity = dehb_results.get("best_fidelity", 0.0)

        for name, code in SEED_STRATEGIES.items():
            fn = compile_strategy(code, n_qubits=self.n_qubits, target_gates=self.target_gates)
            if fn is None:
                logger.warning(f"  Seed '{name}' failed to compile")
                continue

            # Base evaluation through cascade
            scores, _ = self.alpha_evolve.evaluator.evaluate(code)
            if scores is None:
                scores = {"combined": 0.0, "base_fidelity": 0.0}

            # BRFD refinement with DEHB-learned configs
            seed_prog = Program(code=code, scores=scores, source=f"seed:{name}")
            brfd_result = self._refine_with_brfd(seed_prog)

            # Combined scoring
            combined_scores = self._compute_combined_score(
                scores,
                dehb_fidelity,
                brfd_result["brfd_fidelity"],
            )

            # Add to AlphaEvolve database
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
                f"dehb={dehb_fidelity:.4f}, "
                f"brfd={brfd_result['brfd_fidelity']:.4f}, "
                f"combined={combined_scores['combined']:.4f}"
            )

        phase2_time = time.time() - phase2_start
        db_stats = self.alpha_evolve.database.get_stats()
        best = self.alpha_evolve.database.best_program
        logger.info(
            f"  Phase 2 complete: {phase2_time:.1f}s, "
            f"db={db_stats['total']} programs, "
            f"best={best.primary_score:.4f}"
        )

        # ============================================================
        # Phase 3: AlphaEvolve evolution with periodic DEHB re-opt
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 3: AlphaEvolve Evolution (MAP-Elites + LLM + BRFD)")
        logger.info("=" * 70)

        phase3_start = time.time()
        ae_best_before = best.primary_score if best else 0.0

        for gen in range(1, self.ae_generations + 1):
            if self.alpha_evolve.llm_calls >= self.alpha_evolve.max_llm_calls:
                logger.info(f"  LLM budget exhausted at gen {gen}")
                break

            gen_start = time.time()

            # --- Periodic DEHB re-optimisation ---
            if gen > 1 and gen % self.dehb_reoptimize_interval == 0:
                logger.info(f"\n  --- DEHB re-optimisation at gen {gen} ---")
                self._run_dehb(label=f"reopt_gen{gen}")
                self._inject_dehb_strategy()

                # Update AlphaEvolve LLM configs
                self.alpha_evolve.llm_ensemble.default_temperature = self.llm_config.get("temperature", 0.8)
                self.alpha_evolve.llm_ensemble.default_top_p = self.llm_config.get("top_p", 0.95)

                dehb_fidelity = self.dehb_config.get("best_fidelity", dehb_fidelity)

            # --- Generate new candidates ---
            new_programs = []
            valid_count = 0

            n_candidates = min(
                self.ae_population_per_gen,
                self.alpha_evolve.max_llm_calls - self.alpha_evolve.llm_calls,
            )

            for _ in range(n_candidates):
                child = self.alpha_evolve._evolve_one(generation=gen)
                if child is not None:
                    new_programs.append(child)
                    if child.primary_score > 0:
                        valid_count += 1

            # --- BRFD refinement for top-k new candidates ---
            if new_programs:
                new_programs.sort(key=lambda p: p.primary_score, reverse=True)

                for prog in new_programs[:self.brfd_top_k]:
                    brfd_result = self._refine_with_brfd(prog)

                    combined_scores = self._compute_combined_score(
                        prog.scores,
                        dehb_fidelity,
                        brfd_result["brfd_fidelity"],
                    )
                    prog.scores = combined_scores
                    prog.primary_score = combined_scores["combined"]

                    # Re-add with updated scores
                    self.alpha_evolve.database.add(prog)

            gen_time = time.time() - gen_start
            db_stats = self.alpha_evolve.database.get_stats()
            best = self.alpha_evolve.database.best_program

            is_new_best = best and best.primary_score > ae_best_before
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
                "brfd_outer_steps": self.total_brfd_outer,
                "brfd_total_steps": self.total_brfd_steps,
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
                f"BRFD={self.total_brfd_outer}, "
                f"time={gen_time:.1f}s{new_best_marker}"
            )

        phase3_time = time.time() - phase3_start
        logger.info(f"  Phase 3 complete: {phase3_time:.1f}s")

        # ============================================================
        # Phase 4: Final BRFD refinement of global top candidates
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 4: Final BRFD Refinement of Top Candidates")
        logger.info("=" * 70)

        phase4_start = time.time()
        all_programs = self.alpha_evolve.database.get_all_programs()
        all_programs.sort(key=lambda p: p.primary_score, reverse=True)

        top_final = min(3, len(all_programs))
        for i, prog in enumerate(all_programs[:top_final]):
            logger.info(
                f"  Refining top-{i+1}: score={prog.primary_score:.4f} "
                f"(source={prog.source})"
            )
            brfd_result = self._refine_with_brfd(prog)

            combined_scores = self._compute_combined_score(
                prog.scores,
                dehb_fidelity,
                brfd_result["brfd_fidelity"],
            )
            prog.scores = combined_scores
            prog.primary_score = combined_scores["combined"]
            self.alpha_evolve.database.add(prog)

            logger.info(
                f"    -> refined score={prog.primary_score:.4f} "
                f"(brfd={brfd_result['brfd_fidelity']:.4f})"
            )

        phase4_time = time.time() - phase4_start

        # ============================================================
        # Phase 5: Final high-fidelity evaluation
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 5: Final Evaluation (500 samples)")
        logger.info("=" * 70)

        final_evaluator = FitnessEvaluator(
            n_qubits=self.n_qubits,
            target_gates=self.target_gates,
            n_noise_samples=500,
            noise_amplitude=self.noise_amplitude,
            qubit_spacing_nm=self.noise_config.get("qubit_spacing_nm", 108.0),
            tlf_correlation_length_nm=self.noise_config.get("tlf_correlation_length_nm", 81.0),
            max_sequence_length=self.fitness_config.get("max_seq_length", 50),
            timeout_seconds=10.0,
        )

        best = self.alpha_evolve.database.best_program
        fn = compile_strategy(best.code, n_qubits=self.n_qubits, target_gates=self.target_gates) if best else None
        if fn:
            final_result = final_evaluator.evaluate(fn, seed=self.seed)
            final_fitness = final_result["fitness"] if final_result["valid"] else 0.0
        else:
            final_fitness = best.primary_score if best else 0.0

        total_time = time.time() - start_time

        # ============================================================
        # Save results
        # ============================================================
        results = {
            "best_combined_score": float(best.primary_score) if best else 0.0,
            "best_base_fidelity": float(best.scores.get("base_fidelity", 0.0)) if best else 0.0,
            "final_fidelity_500": float(final_fitness),
            "best_code": best.code if best else "",
            "best_scores": best.scores if best else {},
            "best_model": best.model if best else "",
            "best_generation": best.generation if best else 0,
            "best_source": best.source if best else "",
            "total_llm_calls": self.alpha_evolve.llm_calls if self.alpha_evolve else 0,
            "total_dehb_evals": self.total_dehb_evals,
            "total_brfd_outer_steps": self.total_brfd_outer,
            "total_brfd_total_steps": self.total_brfd_steps,
            "total_time_seconds": total_time,
            "total_programs": self.alpha_evolve.database.total_programs if self.alpha_evolve else 0,
            "noise_amplitude": self.noise_amplitude,
            "n_qubits": self.n_qubits,
            "target_gates": self.target_gates,
            "dehb_config": self.dehb_config,
            "llm_config": self.llm_config,
            "brfd_config": self.brfd_config,
            "noise_config": self.noise_config,
            "fitness_config": self.fitness_config,
            "phase_times": {
                "phase1_dehb": phase1_time,
                "phase2_seeds": phase2_time,
                "phase3_evolution": phase3_time,
                "phase4_refinement": phase4_time,
                "total": total_time,
            },
            "history": self.history,
        }

        # Save best strategy
        if best:
            best_file = self.output_dir / "aedb_best_strategy.py"
            with open(best_file, "w") as f:
                f.write("# Best strategy found by AEDB Orchestrator v3\n")
                f.write(f"# Combined score: {best.primary_score:.6f}\n")
                f.write(f"# Base fidelity:  {best.scores.get('base_fidelity', 0):.6f}\n")
                f.write(f"# Final (500):    {final_fitness:.6f}\n")
                f.write(f"# Model:          {best.model}\n")
                f.write(f"# Generation:     {best.generation}\n")
                f.write(f"# Source:         {best.source}\n")
                f.write(f"# N qubits:       {self.n_qubits}\n")
                f.write(f"# Target gates:   {self.target_gates}\n")
                f.write(f"# LLM calls:      {self.alpha_evolve.llm_calls}\n")
                f.write(f"# DEHB evals:     {self.total_dehb_evals}\n")
                f.write(f"# BRFD outer:     {self.total_brfd_outer}\n")
                f.write(f"# Time:           {total_time:.1f}s\n\n")
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

        # Save DEHB configs
        configs_file = self.output_dir / "dehb_learned_configs.json"
        with open(configs_file, "w") as f:
            json.dump({
                "dehb_config": self.dehb_config,
                "noise_config": self.noise_config,
                "llm_config": self.llm_config,
                "brfd_config": self.brfd_config,
                "fitness_config": self.fitness_config,
            }, f, indent=2, default=str)

        logger.info(f"\n{'=' * 70}")
        logger.info("AEDB v3 COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Best combined:     {best.primary_score:.4f}" if best else "  No best")
        logger.info(f"  Best fidelity:     {best.scores.get('base_fidelity', 0):.4f}" if best else "")
        logger.info(f"  Final (500):       {final_fitness:.4f}")
        logger.info(f"  Total programs:    {self.alpha_evolve.database.total_programs}")
        logger.info(f"  LLM calls:         {self.alpha_evolve.llm_calls}")
        logger.info(f"  DEHB evals:        {self.total_dehb_evals}")
        logger.info(f"  BRFD outer steps:  {self.total_brfd_outer}")
        logger.info(f"  BRFD total steps:  {self.total_brfd_steps}")
        logger.info(f"  Phase 1 (DEHB):    {phase1_time:.1f}s")
        logger.info(f"  Phase 2 (Seeds):   {phase2_time:.1f}s")
        logger.info(f"  Phase 3 (Evolve):  {phase3_time:.1f}s")
        logger.info(f"  Phase 4 (Refine):  {phase4_time:.1f}s")
        logger.info(f"  Total time:        {total_time:.1f}s")
        logger.info(f"{'=' * 70}")

        return results


# ======================================================================
# Standalone execution
# ======================================================================

if __name__ == "__main__":
    # Set up logging to both file and console
    log_dir = Path("/home/ubuntu/siliqun/alphaevolve/results/aedb_v3")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "aedb_run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    print("=" * 70)
    print("AEDB v3: DEHB (meta) + AlphaEvolve (outer) + BRFD (inner)")
    print("=" * 70)
    print(f"Architecture: DEHB learns ALL hyperparameters")
    print(f"  -> AlphaEvolve: MAP-Elites + LLM Ensemble + Evaluation Cascade")
    print(f"  -> BRFD: Bilevel Reward Function Discovery")
    print(f"Qubits: 3 (GHZ state)")
    print(f"Noise: 0.5 rad/gate")
    print()

    orchestrator = AEDBOrchestrator(
        n_qubits=3,
        noise_amplitude=0.5,
        target_gates=["ghz"],
        ae_generations=20,
        ae_population_per_gen=5,
        ae_max_llm_calls=120,
        ae_n_islands=3,
        dehb_evaluations=30,
        dehb_reoptimize_interval=10,
        brfd_top_k=2,
        output_dir=str(log_dir),
        seed=42,
    )

    results = orchestrator.run()

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Best combined score:  {results['best_combined_score']:.4f}")
    print(f"Best base fidelity:   {results['best_base_fidelity']:.4f}")
    print(f"Final fidelity (500): {results['final_fidelity_500']:.4f}")
    print(f"Total programs:       {results['total_programs']}")
    print(f"LLM calls:            {results['total_llm_calls']}")
    print(f"DEHB evals:           {results['total_dehb_evals']}")
    print(f"BRFD outer steps:     {results['total_brfd_outer_steps']}")
    print(f"BRFD total steps:     {results['total_brfd_total_steps']}")
    print(f"Time:                 {results['total_time_seconds']:.1f}s")
    print(f"\nDEHB-learned configs:")
    print(f"  LLM:    {results['llm_config']}")
    print(f"  BRFD:   {results['brfd_config']}")
    print(f"  Noise:  {results['noise_config']}")
    print(f"\nBest strategy:")
    print(results['best_code'][:500])
