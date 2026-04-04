"""
AlphaEvolve-style LLM-driven code evolution for noise mitigation discovery.

Uses OpenRouter API to access diverse LLMs as intelligent mutation operators
in an evolutionary loop. SiliQun's physics engine serves as the fitness evaluator.

v2: Improved prompts with complete working examples, error-feedback retry,
    and robust code extraction.
"""

from __future__ import annotations
import os
import json
import time
import copy
import hashlib
import logging
import traceback
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import requests

from skeleton import Gate, STRATEGY_TEMPLATE, DEFAULT_METHOD_BODY
from fitness import FitnessEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class EvolveConfig:
    population_size: int = 10
    n_generations: int = 20
    n_elites: int = 2
    tournament_size: int = 3
    models: List[str] = field(default_factory=lambda: [
        "deepseek/deepseek-chat",
        "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4",
    ])
    temperature: float = 0.7
    max_tokens: int = 2000
    n_qubits: int = 2
    target_gates: List[str] = field(default_factory=lambda: ["bell", "cnot"])
    n_noise_samples: int = 200
    noise_amplitude: float = 0.05
    qubit_spacing_nm: float = 108.0
    tlf_correlation_length_nm: float = 81.0
    max_sequence_length: int = 40
    parsimony_weight: float = 0.001
    max_llm_calls: int = 150
    max_wall_time: float = 3600.0
    max_retries: int = 1  # Retry with error feedback
    output_dir: str = "results"
    save_every: int = 5


# ======================================================================
# Individual
# ======================================================================

@dataclass
class Individual:
    method_body: str
    fitness: float = 0.0
    raw_fitness: float = 0.0
    per_target: Dict[str, float] = field(default_factory=dict)
    sequence_lengths: Dict[str, int] = field(default_factory=dict)
    generation: int = 0
    parent_hash: str = ""
    model_used: str = ""
    valid: bool = False
    error: Optional[str] = None

    @property
    def hash(self) -> str:
        return hashlib.md5(self.method_body.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "hash": self.hash, "fitness": self.fitness,
            "raw_fitness": self.raw_fitness, "per_target": self.per_target,
            "sequence_lengths": self.sequence_lengths, "generation": self.generation,
            "parent_hash": self.parent_hash, "model_used": self.model_used,
            "valid": self.valid, "error": self.error,
            "method_body": self.method_body,
        }


# ======================================================================
# Working example for prompts
# ======================================================================

WORKING_EXAMPLE = '''        # This is a WORKING example. Your code MUST follow this exact pattern.
        # Use Gate("type", (qubit_indices,), {"theta": angle}) for rotations.
        # Use Gate("type", (q1, q2)) for two-qubit gates (no params needed).
        # MUST return a list of Gate objects for EVERY target_gate case.
        
        if target_gate == "bell":
            return [
                Gate("h", (0,)),
                Gate("cnot", (0, 1)),
            ]
        elif target_gate == "cnot":
            return [Gate("cnot", (0, 1))]
        elif target_gate == "swap":
            return [
                Gate("cnot", (0, 1)),
                Gate("cnot", (1, 0)),
                Gate("cnot", (0, 1)),
            ]
        elif target_gate == "ghz":
            seq = [Gate("h", (0,))]
            for q in range(1, n_qubits):
                seq.append(Gate("cnot", (0, q)))
            return seq
        else:
            return [Gate("identity", (0,))]'''


# ======================================================================
# LLM Mutation Engine
# ======================================================================

class LLMMutationEngine:
    
    SYSTEM_MSG = """You are an expert quantum physicist writing Python code.
CRITICAL RULES:
1. Output ONLY the method body - no class, no def, no imports, no markdown
2. Use 8-space indentation for the top level
3. Gate syntax: Gate("rx", (0,), {"theta": 1.57}) or Gate("cnot", (0, 1))
4. Available gates: rx, ry, rz, h, cnot, cz, swap, sqrt_swap, identity
5. MUST handle at least "bell" and "cnot" target_gate cases
6. MUST return a list of Gate objects in every code path
7. Variables available: target_gate, n_qubits, noise_correlation, qubit_spacing_nm, tlf_correlation_length_nm, np, Gate"""

    IMPROVE_PROMPT = """Improve this gate sequence strategy to achieve HIGHER fidelity under TLF-correlated noise.

PHYSICS:
- TLF noise causes correlated Z-dephasing on nearby qubits
- Correlation: C(d) = exp(-d/{tlf_lc}nm), nearest-neighbor = {nn_corr:.3f}
- Each gate in the sequence adds one round of noise exposure
- Shorter sequences = less noise exposure = generally better
- But clever echo/decoupling sequences can cancel correlated errors

CURRENT STRATEGY (fitness = {fitness:.6f}):
```
{current_code}
```

WORKING EXAMPLE (for reference - your output must follow this exact pattern):
```
{example}
```

Write ONLY the improved method body (8-space indent, no def/class):"""

    CREATIVE_PROMPT = """Invent a NOVEL gate sequence strategy for mitigating correlated noise in silicon spin qubits.

PHYSICS:
- TLF charge noise creates spatially correlated Z-dephasing
- Correlation: C(d) = exp(-d/{tlf_lc}nm), nearest-neighbor = {nn_corr:.3f}
- DFS encoding protects against global noise; we fight RESIDUAL correlated noise
- Ideas: dynamical decoupling, composite pulses, correlation-aware scheduling, sqrt_swap native gates

BEST KNOWN (fitness = {fitness:.6f}):
```
{current_code}
```

WORKING EXAMPLE (your output MUST follow this exact pattern):
```
{example}
```

Write ONLY a creative new method body (8-space indent, no def/class):"""

    CROSSOVER_PROMPT = """Combine the best ideas from two parent strategies.

PARENT A (fitness = {fitness_a:.6f}):
```
{code_a}
```

PARENT B (fitness = {fitness_b:.6f}):
```
{code_b}
```

WORKING EXAMPLE (your output MUST follow this exact pattern):
```
{example}
```

Write ONLY the combined method body (8-space indent, no def/class):"""

    REPAIR_PROMPT = """Your previous code had an error. Fix it.

YOUR CODE:
```
{broken_code}
```

ERROR: {error}

WORKING EXAMPLE (follow this exact pattern):
```
{example}
```

Write ONLY the fixed method body (8-space indent, no def/class):"""

    def __init__(self, config: EvolveConfig):
        self.config = config
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.call_count = 0
        self.total_tokens = 0

    def mutate(self, parent: Individual, mutation_type: str = "improve",
               parent_b: Optional[Individual] = None,
               model: Optional[str] = None) -> str:
        if model is None:
            model = np.random.choice(self.config.models)

        nn_corr = np.exp(-self.config.qubit_spacing_nm / self.config.tlf_correlation_length_nm)

        if mutation_type == "crossover" and parent_b is not None:
            prompt = self.CROSSOVER_PROMPT.format(
                fitness_a=parent.fitness, code_a=parent.method_body,
                fitness_b=parent_b.fitness, code_b=parent_b.method_body,
                example=WORKING_EXAMPLE,
            )
        elif mutation_type == "creative":
            prompt = self.CREATIVE_PROMPT.format(
                fitness=parent.fitness, current_code=parent.method_body,
                tlf_lc=self.config.tlf_correlation_length_nm,
                nn_corr=nn_corr, example=WORKING_EXAMPLE,
            )
        else:
            prompt = self.IMPROVE_PROMPT.format(
                fitness=parent.fitness, current_code=parent.method_body,
                tlf_lc=self.config.tlf_correlation_length_nm,
                nn_corr=nn_corr, example=WORKING_EXAMPLE,
            )

        try:
            response = self._call_llm(prompt, model)
            return self._extract_method_body(response)
        except Exception as e:
            logger.warning(f"LLM call failed ({model}): {e}")
            return parent.method_body

    def repair(self, broken_code: str, error: str, model: str) -> str:
        prompt = self.REPAIR_PROMPT.format(
            broken_code=broken_code, error=error, example=WORKING_EXAMPLE,
        )
        try:
            response = self._call_llm(prompt, model)
            return self._extract_method_body(response)
        except Exception as e:
            logger.warning(f"Repair call failed ({model}): {e}")
            return broken_code

    def _call_llm(self, prompt: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/r3d-phd/siliqun",
            "X-Title": "SiliQun AlphaEvolve",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        self.call_count += 1
        if "usage" in data:
            self.total_tokens += data["usage"].get("total_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def _extract_method_body(self, response: str) -> str:
        """Robustly extract the method body from LLM response."""
        text = response.strip()

        # 1. Remove markdown code fences
        # Handle ```python ... ``` and ``` ... ```
        fence_pattern = r'```(?:python)?\s*\n?(.*?)```'
        matches = re.findall(fence_pattern, text, re.DOTALL)
        if matches:
            # Take the longest match (most likely the actual code)
            text = max(matches, key=len)

        # 2. Remove any class/def/import lines
        lines = text.strip().split("\n")
        filtered = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            # Skip imports
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            # Skip class/def declarations
            if stripped.startswith("class ") or stripped.startswith("def "):
                continue
            # Skip decorator lines
            if stripped.startswith("@"):
                continue
            # Handle docstrings
            if '"""' in stripped:
                count = stripped.count('"""')
                if count == 1:
                    in_docstring = not in_docstring
                    continue
                elif count >= 2:
                    # Single-line docstring
                    continue
            if in_docstring:
                continue
            # Skip lines with just "return" type hints
            if stripped.startswith("-> "):
                continue
            filtered.append(line)

        text = "\n".join(filtered).strip()
        if not text:
            text = WORKING_EXAMPLE

        # 3. Re-indent to 8 spaces
        lines = text.split("\n")
        min_indent = float("inf")
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        if min_indent == float("inf"):
            min_indent = 0

        reindented = []
        for line in lines:
            if line.strip():
                reindented.append("        " + line[min_indent:])
            else:
                reindented.append("")

        return "\n".join(reindented)


# ======================================================================
# Evolutionary Loop
# ======================================================================

class AlphaEvolveLoop:

    def __init__(self, config: EvolveConfig):
        self.config = config
        self.mutation_engine = LLMMutationEngine(config)
        self.evaluator = FitnessEvaluator(
            n_qubits=config.n_qubits,
            target_gates=config.target_gates,
            n_noise_samples=config.n_noise_samples,
            noise_amplitude=config.noise_amplitude,
            qubit_spacing_nm=config.qubit_spacing_nm,
            tlf_correlation_length_nm=config.tlf_correlation_length_nm,
            max_sequence_length=config.max_sequence_length,
        )
        self.population: List[Individual] = []
        self.best_ever: Optional[Individual] = None
        self.history: List[Dict] = []
        self.start_time = 0.0
        os.makedirs(config.output_dir, exist_ok=True)

    def _compile_and_evaluate(self, individual: Individual) -> Individual:
        """Compile and evaluate, with optional retry on failure."""
        for attempt in range(1 + self.config.max_retries):
            try:
                full_source = STRATEGY_TEMPLATE.replace("{METHOD_BODY}", individual.method_body)
                exec_globals = {"np": np, "Gate": Gate, "math": __import__("math")}
                exec(full_source, exec_globals)
                strategy_class = exec_globals["EvolvedStrategy"]
                strategy_fn = strategy_class.generate_gate_sequence
                result = self.evaluator.evaluate(strategy_fn, seed=42)

                individual.raw_fitness = result["fitness"]
                individual.per_target = result["per_target"]
                individual.sequence_lengths = result["sequence_lengths"]
                individual.valid = result["valid"]
                individual.error = result["error"]

                if result["valid"] and result["sequence_lengths"]:
                    avg_len = np.mean(list(result["sequence_lengths"].values()))
                    individual.fitness = result["fitness"] - self.config.parsimony_weight * avg_len
                else:
                    individual.fitness = 0.0

                if individual.valid:
                    return individual  # Success!

                # Invalid result - try repair if we have retries left
                if attempt < self.config.max_retries and individual.model_used:
                    error_msg = individual.error or "Unknown evaluation error"
                    logger.info(f"  Attempting repair (error: {error_msg[:80]}...)")
                    individual.method_body = self.mutation_engine.repair(
                        individual.method_body, error_msg, individual.model_used
                    )
                    continue

            except Exception as e:
                error_msg = f"Compilation: {str(e)}"
                individual.valid = False
                individual.fitness = 0.0
                individual.error = error_msg

                # Try repair
                if attempt < self.config.max_retries and individual.model_used:
                    logger.info(f"  Attempting repair (error: {error_msg[:80]}...)")
                    individual.method_body = self.mutation_engine.repair(
                        individual.method_body, error_msg, individual.model_used
                    )
                    continue

        return individual

    def _initialize_population(self):
        seeds = [
            ("standard", DEFAULT_METHOD_BODY),
            ("correlation_aware", '''        echo_depth = max(1, int(noise_correlation * 3))
        if target_gate == "bell":
            seq = [Gate("h", (0,))]
            for _ in range(echo_depth):
                seq.append(Gate("rz", (0,), {"theta": np.pi}))
                seq.append(Gate("rz", (0,), {"theta": -np.pi}))
            seq.append(Gate("cnot", (0, 1)))
            return seq
        elif target_gate == "cnot":
            return [Gate("cnot", (0, 1))]
        elif target_gate == "swap":
            return [Gate("cnot", (0, 1)), Gate("cnot", (1, 0)), Gate("cnot", (0, 1))]
        elif target_gate == "ghz":
            seq = [Gate("h", (0,))]
            for q in range(1, n_qubits):
                seq.append(Gate("cnot", (0, q)))
            return seq
        else:
            return [Gate("identity", (0,))]
'''),
            ("sqrt_swap_native", '''        if target_gate == "bell":
            return [
                Gate("h", (0,)),
                Gate("sqrt_swap", (0, 1)),
                Gate("sqrt_swap", (0, 1)),
            ]
        elif target_gate == "cnot":
            return [Gate("h", (1,)), Gate("cz", (0, 1)), Gate("h", (1,))]
        elif target_gate == "swap":
            return [Gate("swap", (0, 1))]
        elif target_gate == "ghz":
            seq = [Gate("h", (0,))]
            for q in range(1, n_qubits):
                seq.append(Gate("cnot", (0, q)))
            return seq
        else:
            return [Gate("identity", (0,))]
'''),
        ]

        for name, body in seeds:
            ind = Individual(method_body=body, generation=0, model_used=f"seed:{name}")
            ind = self._compile_and_evaluate(ind)
            self.population.append(ind)
            logger.info(f"Seed '{name}': fitness={ind.fitness:.6f}, valid={ind.valid}")

        # Fill remaining with LLM mutations
        while len(self.population) < self.config.population_size:
            if self.mutation_engine.call_count >= self.config.max_llm_calls:
                break
            parent = max(self.population, key=lambda x: x.fitness)
            model = self.config.models[len(self.population) % len(self.config.models)]
            mutation_type = np.random.choice(["improve", "creative"])

            new_body = self.mutation_engine.mutate(parent, mutation_type, model=model)
            ind = Individual(method_body=new_body, generation=0,
                             parent_hash=parent.hash, model_used=model)
            ind = self._compile_and_evaluate(ind)
            self.population.append(ind)
            status = "VALID" if ind.valid else f"INVALID ({ind.error[:60] if ind.error else '?'})"
            logger.info(f"Init ({model.split('/')[-1]}): {status}, fitness={ind.fitness:.6f}")

    def _tournament_select(self) -> Individual:
        candidates = np.random.choice(
            len(self.population),
            size=min(self.config.tournament_size, len(self.population)),
            replace=False,
        )
        return self.population[max(candidates, key=lambda i: self.population[i].fitness)]

    def _evolve_generation(self, gen: int):
        new_population = []

        # Elitism
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for e in sorted_pop[:self.config.n_elites]:
            elite = copy.deepcopy(e)
            elite.generation = gen
            new_population.append(elite)

        while len(new_population) < self.config.population_size:
            if self.mutation_engine.call_count >= self.config.max_llm_calls:
                logger.warning("LLM budget exhausted")
                break
            if time.time() - self.start_time > self.config.max_wall_time:
                logger.warning("Wall time budget exhausted")
                break

            r = np.random.random()
            if r < 0.5:
                mutation_type = "improve"
            elif r < 0.8:
                mutation_type = "creative"
            else:
                mutation_type = "crossover"

            parent = self._tournament_select()
            parent_b = self._tournament_select() if mutation_type == "crossover" else None
            model = self.config.models[len(new_population) % len(self.config.models)]

            new_body = self.mutation_engine.mutate(parent, mutation_type, parent_b=parent_b, model=model)
            offspring = Individual(method_body=new_body, generation=gen,
                                   parent_hash=parent.hash, model_used=model)
            offspring = self._compile_and_evaluate(offspring)
            new_population.append(offspring)

            if offspring.valid:
                logger.info(
                    f"Gen {gen} | VALID | {model.split('/')[-1]} | "
                    f"fitness={offspring.fitness:.6f} | {mutation_type} | "
                    f"seqs={offspring.sequence_lengths}"
                )
            else:
                logger.info(
                    f"Gen {gen} | INVALID | {model.split('/')[-1]} | "
                    f"{mutation_type} | err={offspring.error[:70] if offspring.error else '?'}"
                )

        self.population = new_population

    def _update_best(self):
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_ever is None or current_best.fitness > self.best_ever.fitness:
            self.best_ever = copy.deepcopy(current_best)
            logger.info(f"*** NEW BEST: fitness={self.best_ever.fitness:.6f} "
                        f"(hash={self.best_ever.hash}, model={self.best_ever.model_used})")

    def _log_generation(self, gen: int):
        fitnesses = [ind.fitness for ind in self.population]
        valid_count = sum(1 for ind in self.population if ind.valid)
        stats = {
            "generation": gen,
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "mean_fitness": np.mean(fitnesses) if fitnesses else 0,
            "std_fitness": np.std(fitnesses) if fitnesses else 0,
            "valid_count": valid_count,
            "total_count": len(self.population),
            "llm_calls": self.mutation_engine.call_count,
            "total_tokens": self.mutation_engine.total_tokens,
            "wall_time": time.time() - self.start_time,
            "best_ever_fitness": self.best_ever.fitness if self.best_ever else 0,
        }
        self.history.append(stats)
        logger.info(
            f"=== Gen {gen} === best={stats['best_fitness']:.6f} | "
            f"mean={stats['mean_fitness']:.6f} | valid={valid_count}/{len(self.population)} | "
            f"LLM={stats['llm_calls']} | {stats['wall_time']:.0f}s"
        )

    def _save_checkpoint(self, gen: int):
        checkpoint = {
            "generation": gen,
            "best_ever": self.best_ever.to_dict() if self.best_ever else None,
            "population": [ind.to_dict() for ind in self.population],
            "history": self.history,
        }
        path = os.path.join(self.config.output_dir, f"checkpoint_gen{gen:03d}.json")
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        if self.best_ever:
            best_path = os.path.join(self.config.output_dir, "best_strategy.py")
            full_source = STRATEGY_TEMPLATE.replace("{METHOD_BODY}", self.best_ever.method_body)
            with open(best_path, "w") as f:
                f.write(f"# Best strategy found by AlphaEvolve\n")
                f.write(f"# Generation: {self.best_ever.generation}\n")
                f.write(f"# Fitness: {self.best_ever.fitness:.6f}\n")
                f.write(f"# Raw fidelity: {self.best_ever.raw_fitness:.6f}\n")
                f.write(f"# Model: {self.best_ever.model_used}\n")
                f.write(f"# Per-target: {self.best_ever.per_target}\n")
                f.write(f"# Seq lengths: {self.best_ever.sequence_lengths}\n\n")
                f.write("import numpy as np\n")
                f.write("from skeleton import Gate\n\n")
                f.write(full_source)

    def run(self):
        self.start_time = time.time()
        nn_corr = np.exp(-self.config.qubit_spacing_nm / self.config.tlf_correlation_length_nm)

        logger.info("=" * 60)
        logger.info("AlphaEvolve Noise Mitigation Discovery v2")
        logger.info("=" * 60)
        logger.info(f"Pop={self.config.population_size} | Gens={self.config.n_generations} | "
                     f"Noise={self.config.noise_amplitude} | NN_corr={nn_corr:.3f}")
        logger.info(f"Models: {[m.split('/')[-1] for m in self.config.models]}")
        logger.info(f"Max LLM calls: {self.config.max_llm_calls} | Retries: {self.config.max_retries}")
        logger.info("=" * 60)

        self._initialize_population()
        self._update_best()
        self._log_generation(0)
        self._save_checkpoint(0)

        for gen in range(1, self.config.n_generations + 1):
            if self.mutation_engine.call_count >= self.config.max_llm_calls:
                logger.warning("LLM budget exhausted. Stopping.")
                break
            if time.time() - self.start_time > self.config.max_wall_time:
                logger.warning("Wall time exhausted. Stopping.")
                break

            self._evolve_generation(gen)
            self._update_best()
            self._log_generation(gen)
            if gen % self.config.save_every == 0:
                self._save_checkpoint(gen)

        self._save_checkpoint(gen)

        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info("EVOLUTION COMPLETE")
        logger.info(f"Time: {elapsed:.0f}s | LLM calls: {self.mutation_engine.call_count} | "
                     f"Tokens: {self.mutation_engine.total_tokens}")
        logger.info(f"Best fitness: {self.best_ever.fitness:.6f} | "
                     f"Raw fidelity: {self.best_ever.raw_fitness:.6f}")
        logger.info(f"Per-target: {self.best_ever.per_target}")
        logger.info(f"Seq lengths: {self.best_ever.sequence_lengths}")
        logger.info(f"Model: {self.best_ever.model_used}")
        logger.info("=" * 60)
        logger.info("Best strategy code:")
        logger.info(self.best_ever.method_body)

        return self.best_ever


if __name__ == "__main__":
    config = EvolveConfig(
        population_size=10,
        n_generations=20,
        n_elites=2,
        tournament_size=3,
        models=[
            "deepseek/deepseek-chat",
            "google/gemini-2.5-flash",
            "anthropic/claude-sonnet-4",
        ],
        temperature=0.7,
        max_tokens=2000,
        n_qubits=2,
        target_gates=["bell", "cnot"],
        n_noise_samples=200,
        noise_amplitude=0.05,
        qubit_spacing_nm=108.0,
        tlf_correlation_length_nm=81.0,
        max_sequence_length=40,
        parsimony_weight=0.001,
        max_llm_calls=150,
        max_retries=1,
        max_wall_time=3600.0,
        output_dir="/home/ubuntu/siliqun/alphaevolve/results",
        save_every=5,
    )

    loop = AlphaEvolveLoop(config)
    best = loop.run()
