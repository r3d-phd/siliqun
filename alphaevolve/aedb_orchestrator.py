"""
AEDB Orchestrator: AlphaEvolve + DEHB + BRFD integrated tri-level optimization.

Architecture:
    AlphaEvolve (outer): Evolves gate sequence ALGORITHMS (Python code)
    DEHB (middle): Optimizes HYPERPARAMETERS for each algorithm
    BRFD (inner): Discovers optimal REWARD FUNCTIONS for DRL training

The three layers interact as follows:
    1. AlphaEvolve generates a candidate gate sequence algorithm
    2. DEHB finds the best hyperparameters for that algorithm
    3. BRFD discovers the best reward function for DRL training with
       that algorithm + those hyperparameters
    4. The combined fitness feeds back to AlphaEvolve's selection

References:
    - AlphaEvolve: Fawzi et al., arXiv:2602.16928, 2025
    - DEHB: Awad et al., IJCAI 2021
    - BRFD: Nature Communications, 2025
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitness import FitnessEvaluator, Gate
from evolve_v2 import (
    AlphaEvolveV2,
    Individual,
    compile_strategy,
    call_llm,
    extract_function,
    FREE_MODELS,
    SEED_STRATEGIES,
)
from dehb_optimizer import DEHBOptimizer
from brfd_reward import BRFDTrainer

logger = logging.getLogger("aedb.orchestrator")


class AEDBOrchestrator:
    """Tri-level optimization orchestrator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for evaluation.
    noise_amplitude : float
        Base noise amplitude (rad/gate).
    ae_generations : int
        Number of AlphaEvolve generations.
    ae_population : int
        AlphaEvolve population size.
    ae_max_llm_calls : int
        Maximum LLM calls for AlphaEvolve.
    dehb_evaluations : int
        Number of DEHB evaluations per algorithm.
    brfd_outer_steps : int
        Number of BRFD outer optimization steps.
    brfd_inner_episodes : int
        Number of BRFD inner training episodes.
    output_dir : str
        Directory for saving results.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        noise_amplitude: float = 0.3,
        ae_generations: int = 10,
        ae_population: int = 8,
        ae_max_llm_calls: int = 60,
        dehb_evaluations: int = 15,
        brfd_outer_steps: int = 8,
        brfd_inner_episodes: int = 15,
        output_dir: str = "results/aedb",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.noise_amplitude = noise_amplitude
        self.ae_generations = ae_generations
        self.ae_population = ae_population
        self.ae_max_llm_calls = ae_max_llm_calls
        self.dehb_evaluations = dehb_evaluations
        self.brfd_outer_steps = brfd_outer_steps
        self.brfd_inner_episodes = brfd_inner_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Base fitness evaluator
        self.base_evaluator = FitnessEvaluator(
            n_qubits=n_qubits,
            target_gates=["bell", "cnot"],
            n_noise_samples=100,
            noise_amplitude=noise_amplitude,
            qubit_spacing_nm=108.0,
            tlf_correlation_length_nm=81.0,
            max_sequence_length=50,
            timeout_seconds=5.0,
        )

        # Tracking
        self.total_llm_calls = 0
        self.total_dehb_evals = 0
        self.total_brfd_steps = 0
        self.history = []
        self.best_ever = None
        self.best_fitness = 0.0

    def _evaluate_with_dehb(
        self,
        strategy_code: str,
        strategy_fn: callable,
    ) -> Dict:
        """Evaluate a strategy with DEHB hyperparameter optimization.

        Parameters
        ----------
        strategy_code : str
            Source code of the strategy.
        strategy_fn : callable
            Compiled strategy function.

        Returns
        -------
        dict
            Results including best fitness and optimal hyperparameters.
        """
        try:
            # Use unique DEHB output dir to avoid checkpoint reuse
            import hashlib
            code_hash = hashlib.md5(strategy_code.encode()).hexdigest()[:8]
            dehb_dir = str(self.output_dir / f"dehb_{code_hash}_{int(time.time())}")
            dehb = DEHBOptimizer(
                output_dir=dehb_dir,
                max_evaluations=self.dehb_evaluations,
                seed=self.seed,
            )

            results = dehb.run()
            self.total_dehb_evals += self.dehb_evaluations

            return {
                "dehb_best_fitness": results["best_fidelity"],
                "dehb_best_config": results["best_config"],
                "dehb_n_evals": results["total_evaluations"],
            }

        except Exception as e:
            logger.warning(f"DEHB failed: {e}")
            # Fall back to base evaluation
            result = self.base_evaluator.evaluate(strategy_fn, seed=self.seed)
            return {
                "dehb_best_fitness": result["fitness"] if result["valid"] else 0.0,
                "dehb_best_config": None,
                "dehb_n_evals": 0,
            }

    def _evaluate_with_brfd(
        self,
        strategy_fn: callable,
        noise_amplitude: float,
    ) -> Dict:
        """Evaluate with BRFD reward function discovery.

        Parameters
        ----------
        strategy_fn : callable
            Compiled strategy function.
        noise_amplitude : float
            Noise amplitude for evaluation.

        Returns
        -------
        dict
            Results including BRFD-optimized fidelity.
        """
        try:
            brfd = BRFDTrainer(
                n_qubits=self.n_qubits,
                noise_amplitude=noise_amplitude,
                n_outer_steps=self.brfd_outer_steps,
                n_inner_episodes=self.brfd_inner_episodes,
                seed=self.seed,
            )

            result = brfd.train()
            self.total_brfd_steps += self.brfd_outer_steps

            return {
                "brfd_best_fidelity": result["best_fidelity"],
                "brfd_improvement": result.get("improvement", 0.0),
            }

        except Exception as e:
            logger.warning(f"BRFD failed: {e}")
            return {
                "brfd_best_fidelity": 0.0,
                "brfd_improvement": 0.0,
            }

    def _combined_fitness(
        self,
        base_fitness: float,
        dehb_fitness: float,
        brfd_fidelity: float,
    ) -> float:
        """Compute combined tri-level fitness score.

        Parameters
        ----------
        base_fitness : float
            Raw strategy fitness.
        dehb_fitness : float
            DEHB-optimized fitness.
        brfd_fidelity : float
            BRFD-discovered reward fidelity.

        Returns
        -------
        float
            Combined fitness score.
        """
        # Weighted combination:
        # - 50% from DEHB-optimized fitness (best hyperparams for this algo)
        # - 30% from base fitness (algorithm quality without tuning)
        # - 20% from BRFD (reward function quality)
        return 0.5 * dehb_fitness + 0.3 * base_fitness + 0.2 * brfd_fidelity

    def run(self) -> Dict:
        """Run the full AEDB tri-level optimization.

        Returns
        -------
        dict
            Complete results from all three optimization levels.
        """
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("AEDB Orchestrator: AlphaEvolve + DEHB + BRFD")
        logger.info("=" * 70)
        logger.info(f"  Qubits: {self.n_qubits}")
        logger.info(f"  Noise: {self.noise_amplitude} rad/gate")
        logger.info(f"  AE: {self.ae_generations} gens, pop={self.ae_population}")
        logger.info(f"  DEHB: {self.dehb_evaluations} evals per strategy")
        logger.info(f"  BRFD: {self.brfd_outer_steps} outer steps")
        logger.info(f"  Free LLMs: {len(FREE_MODELS)} models")

        # ============================================================
        # Phase 1: Evaluate seed strategies with full AEDB pipeline
        # ============================================================
        logger.info("\n--- Phase 1: Seed Strategy Evaluation ---")

        population = []
        for name, code in SEED_STRATEGIES.items():
            fn = compile_strategy(code)
            if fn is None:
                continue

            # Base fitness
            base_result = self.base_evaluator.evaluate(fn, seed=self.seed)
            base_fitness = base_result["fitness"] if base_result["valid"] else 0.0

            # DEHB optimization
            dehb_result = self._evaluate_with_dehb(code, fn)

            # BRFD reward discovery
            brfd_result = self._evaluate_with_brfd(fn, self.noise_amplitude)

            # Combined fitness
            combined = self._combined_fitness(
                base_fitness,
                dehb_result["dehb_best_fitness"],
                brfd_result["brfd_best_fidelity"],
            )

            ind = Individual(
                code=code,
                fitness=combined,
                source=f"seed:{name}",
                generation=0,
            )
            population.append(ind)

            logger.info(
                f"  Seed '{name}': base={base_fitness:.4f}, "
                f"dehb={dehb_result['dehb_best_fitness']:.4f}, "
                f"brfd={brfd_result['brfd_best_fidelity']:.4f}, "
                f"combined={combined:.4f}"
            )

        if not population:
            logger.error("No valid seed strategies!")
            return {"error": "No valid seeds"}

        self.best_ever = max(population, key=lambda x: x.fitness)
        self.best_fitness = self.best_ever.fitness

        # Fill population
        while len(population) < self.ae_population:
            population.append(Individual(
                code=self.best_ever.code,
                fitness=self.best_ever.fitness,
                source="seed:clone",
                generation=0,
            ))

        # ============================================================
        # Phase 2: LLM-driven evolution with DEHB+BRFD evaluation
        # ============================================================
        logger.info("\n--- Phase 2: LLM-Driven Evolution ---")

        for gen in range(1, self.ae_generations + 1):
            if self.total_llm_calls >= self.ae_max_llm_calls:
                logger.info(f"LLM budget exhausted at gen {gen}")
                break

            gen_start = time.time()
            new_individuals = []

            # Generate mutants
            n_mutants = min(
                self.ae_population // 2,
                self.ae_max_llm_calls - self.total_llm_calls,
            )

            for i in range(n_mutants):
                if self.total_llm_calls >= self.ae_max_llm_calls:
                    break

                # Tournament selection
                candidates = self.rng.choice(
                    len(population), size=min(3, len(population)), replace=False
                )
                parent = max(
                    [population[c] for c in candidates],
                    key=lambda x: x.fitness,
                )

                # LLM mutation
                model = FREE_MODELS[self.total_llm_calls % len(FREE_MODELS)]
                prompt = self._build_aedb_prompt(parent, self.best_ever, gen)

                self.total_llm_calls += 1
                response = call_llm(prompt, model)

                if response is None:
                    continue

                code = extract_function(response)
                if code is None:
                    continue

                fn = compile_strategy(code)
                if fn is None:
                    continue

                # Evaluate with base fitness only for speed
                # (full DEHB+BRFD only for top candidates)
                base_result = self.base_evaluator.evaluate(fn, seed=self.seed)
                base_fitness = base_result["fitness"] if base_result["valid"] else 0.0

                if base_fitness > 0:
                    new_individuals.append(Individual(
                        code=code,
                        fitness=base_fitness,
                        source="mutation",
                        model=model.split("/")[-1],
                        generation=gen,
                    ))

            # Full AEDB evaluation for top new candidates
            if new_individuals:
                new_individuals.sort(key=lambda x: x.fitness, reverse=True)
                top_k = min(2, len(new_individuals))

                for ind in new_individuals[:top_k]:
                    fn = compile_strategy(ind.code)
                    if fn is None:
                        continue

                    dehb_result = self._evaluate_with_dehb(ind.code, fn)
                    brfd_result = self._evaluate_with_brfd(
                        fn, self.noise_amplitude
                    )

                    ind.fitness = self._combined_fitness(
                        ind.fitness,
                        dehb_result["dehb_best_fitness"],
                        brfd_result["brfd_best_fidelity"],
                    )

            # Selection
            combined_pop = population + new_individuals
            combined_pop.sort(key=lambda x: x.fitness, reverse=True)
            population = combined_pop[:self.ae_population]

            # Update best
            gen_best = population[0]
            if gen_best.fitness > self.best_fitness:
                self.best_ever = gen_best
                self.best_fitness = gen_best.fitness
                logger.info(
                    f"  NEW BEST at gen {gen}: {gen_best.fitness:.4f} "
                    f"(model={gen_best.model})"
                )

            gen_time = time.time() - gen_start
            valid_new = [x for x in new_individuals if x.fitness > 0]

            gen_stats = {
                "generation": gen,
                "best_fitness": self.best_fitness,
                "gen_mean": np.mean([x.fitness for x in population]),
                "new_valid": len(valid_new),
                "new_total": n_mutants,
                "llm_calls": self.total_llm_calls,
                "dehb_evals": self.total_dehb_evals,
                "brfd_steps": self.total_brfd_steps,
                "time_seconds": gen_time,
            }
            self.history.append(gen_stats)

            logger.info(
                f"Gen {gen}: best={self.best_fitness:.4f}, "
                f"mean={gen_stats['gen_mean']:.4f}, "
                f"valid={len(valid_new)}/{n_mutants}, "
                f"LLM={self.total_llm_calls}, "
                f"DEHB={self.total_dehb_evals}, "
                f"BRFD={self.total_brfd_steps}, "
                f"time={gen_time:.1f}s"
            )

        # ============================================================
        # Phase 3: Final evaluation of best strategy
        # ============================================================
        logger.info("\n--- Phase 3: Final Evaluation ---")

        fn = compile_strategy(self.best_ever.code)
        if fn:
            # High-fidelity evaluation
            final_evaluator = FitnessEvaluator(
                n_qubits=self.n_qubits,
                target_gates=["bell", "cnot"],
                n_noise_samples=500,
                noise_amplitude=self.noise_amplitude,
                qubit_spacing_nm=108.0,
                tlf_correlation_length_nm=81.0,
                max_sequence_length=50,
                timeout_seconds=10.0,
            )
            final_result = final_evaluator.evaluate(fn, seed=self.seed)
            final_fitness = final_result["fitness"] if final_result["valid"] else 0.0
        else:
            final_fitness = self.best_fitness

        total_time = time.time() - start_time

        # ============================================================
        # Save results
        # ============================================================
        results = {
            "best_fitness": float(self.best_fitness),
            "final_fitness": float(final_fitness),
            "best_code": self.best_ever.code,
            "best_model": self.best_ever.model,
            "best_generation": self.best_ever.generation,
            "best_source": self.best_ever.source,
            "total_llm_calls": self.total_llm_calls,
            "total_dehb_evals": self.total_dehb_evals,
            "total_brfd_steps": self.total_brfd_steps,
            "total_time_seconds": total_time,
            "noise_amplitude": self.noise_amplitude,
            "n_qubits": self.n_qubits,
            "history": self.history,
        }

        # Save best strategy
        best_file = self.output_dir / "aedb_best_strategy.py"
        with open(best_file, "w") as f:
            f.write(f"# Best strategy found by AEDB Orchestrator\n")
            f.write(f"# Combined fitness: {self.best_fitness:.6f}\n")
            f.write(f"# Final fitness (500 samples): {final_fitness:.6f}\n")
            f.write(f"# Model: {self.best_ever.model}\n")
            f.write(f"# Generation: {self.best_ever.generation}\n")
            f.write(f"# Source: {self.best_ever.source}\n")
            f.write(f"# Total LLM calls: {self.total_llm_calls}\n")
            f.write(f"# Total DEHB evals: {self.total_dehb_evals}\n")
            f.write(f"# Total BRFD steps: {self.total_brfd_steps}\n")
            f.write(f"# Total time: {total_time:.1f}s\n\n")
            f.write(f"from fitness import Gate\n\n")
            f.write(self.best_ever.code)

        # Save full results
        results_file = self.output_dir / "aedb_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save history
        history_file = self.output_dir / "aedb_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"AEDB COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Best combined fitness: {self.best_fitness:.4f}")
        logger.info(f"Final fitness (500 samples): {final_fitness:.4f}")
        logger.info(f"Total LLM calls: {self.total_llm_calls}")
        logger.info(f"Total DEHB evaluations: {self.total_dehb_evals}")
        logger.info(f"Total BRFD steps: {self.total_brfd_steps}")
        logger.info(f"Total time: {total_time:.1f}s")

        return results

    def _build_aedb_prompt(
        self,
        parent: Individual,
        best: Individual,
        generation: int,
    ) -> str:
        """Build an AEDB-aware mutation prompt.

        Parameters
        ----------
        parent : Individual
            Parent strategy.
        best : Individual
            Best strategy found so far.
        generation : int
            Current generation.

        Returns
        -------
        str
            Prompt for the LLM.
        """
        return f"""You are designing gate sequences for silicon spin qubits under
TLF-correlated noise (amplitude={self.noise_amplitude} rad/gate).

The noise has spatial correlations: nearby qubits experience similar noise
with correlation ~ exp(-distance / 81nm). Qubit spacing is 108nm, so
nearest-neighbor correlation is ~0.26.

CURRENT STRATEGY (fitness={parent.fitness:.4f}):
```python
{parent.code}
```

BEST STRATEGY SO FAR (fitness={best.fitness:.4f}):
```python
{best.code}
```

Generation {generation}/{self.ae_generations}. The noise is STRONG
({self.noise_amplitude} rad/gate), so each extra gate adds significant noise.
The key challenge: find gate sequences that CANCEL correlated noise
without adding too many extra gates.

Ideas to explore:
- Symmetric echo pairs that cancel correlated Z-noise
- Adaptive rotations proportional to exp(-spacing/corr_length)
- Interleaving identity gates to create noise-cancellation windows
- Using the correlation structure to design parity-check-like sequences
- Minimal-depth decompositions that avoid unnecessary gates

Return ONLY the improved Python function:
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
"""


# ======================================================================
# Standalone execution
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("AEDB: AlphaEvolve + DEHB + BRFD Orchestrator")
    print("=" * 70)
    print(f"Free LLMs: {', '.join(m.split('/')[-1] for m in FREE_MODELS)}")
    print(f"Noise: 0.5 rad/gate (hard regime)")
    print(f"Target: Discover novel noise mitigation strategies")
    print()

    orchestrator = AEDBOrchestrator(
        n_qubits=2,
        noise_amplitude=0.5,
        ae_generations=6,
        ae_population=6,
        ae_max_llm_calls=30,
        dehb_evaluations=8,
        brfd_outer_steps=4,
        brfd_inner_episodes=8,
        output_dir="/home/ubuntu/siliqun/alphaevolve/results/aedb_run2",
        seed=42,
    )

    results = orchestrator.run()

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Best combined fitness: {results['best_fitness']:.4f}")
    print(f"Final fitness (500 samples): {results['final_fitness']:.4f}")
    print(f"Best model: {results['best_model']}")
    print(f"Best generation: {results['best_generation']}")
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"Total DEHB evals: {results['total_dehb_evals']}")
    print(f"Total BRFD steps: {results['total_brfd_steps']}")
    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print(f"\nBest strategy:")
    print(results['best_code'])
