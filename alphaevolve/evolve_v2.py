"""
AlphaEvolve v2: LLM-driven code evolution for quantum noise mitigation.

Upgrades from v1:
    - Uses FREE LLMs via OpenRouter (Qwen3-Coder, Llama 3.3 70B, etc.)
    - Harder noise (0.3 rad/gate) where novel strategies can shine
    - Better prompt engineering with complete working examples
    - Error-feedback retry loop
    - Multi-model ensemble for diversity

References:
    Fawzi et al., "AlphaEvolve: A Gemini-Powered Coding Agent for
    Designing Advanced Algorithms", arXiv:2602.16928, 2025.
"""

from __future__ import annotations
import os
import sys
import json
import time
import copy
import logging
import textwrap
import traceback
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitness import FitnessEvaluator, Gate

logger = logging.getLogger("aedb.evolve_v2")

# ======================================================================
# LLM Configuration: Local Ollama (primary) + Gemini (secondary) + OpenRouter (fallback)
# ======================================================================

# Local Ollama (primary - zero rate limits, runs on user's RTX 2070)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-gpu")
USE_OLLAMA = os.environ.get("USE_OLLAMA", "1") == "1"  # Enable by default

# Gemini API (secondary - reliable, fast, high quality)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
USE_GEMINI = bool(GEMINI_API_KEY)  # Use Gemini if key available

# OpenRouter free models (fallback)
_PRIMARY_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "arcee-ai/trinity-large-preview:free",
]
_FALLBACK_MODELS = [
    "qwen/qwen3-coder:free",
    "qwen/qwen3.6-plus:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]
FREE_MODELS = [OLLAMA_MODEL, GEMINI_MODEL] + _PRIMARY_MODELS + _FALLBACK_MODELS

# Delay between LLM calls (seconds)
LLM_CALL_DELAY = 1  # Local Ollama has zero rate limits

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ======================================================================
# Seed Strategies (complete working examples)
# ======================================================================

SEED_STRATEGIES = {
    "standard": '''def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
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

    "echo_cancel": '''def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Echo-based noise cancellation: insert X-X echo pairs around core gates."""
    import numpy as np
    gates = []
    # Pre-echo on all qubits
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rx", [q], {"theta": np.pi}))
    # Core operation
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    # Post-echo (undo pre-echo)
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rx", [q], {"theta": np.pi}))
    return gates
''',

    "correlation_aware": '''def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Correlation-aware: use ZZ rotations to exploit spatial noise correlations."""
    import numpy as np
    gates = []
    # Correlation-dependent pre-rotation
    theta = nn_correlation * np.pi / 4
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rz", [q], {"theta": theta}))
    # Core operation
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        gates.append(Gate("h", [0]))
        for i in range(n_qubits - 1):
            gates.append(Gate("cnot", [i, i + 1]))
    # Undo pre-rotation
    for q in range(min(n_qubits, 2)):
        gates.append(Gate("rz", [q], {"theta": -theta}))
    return gates
''',
}


# ======================================================================
# LLM Mutation Engine
# ======================================================================

SYSTEM_PROMPT = """You are an expert quantum physicist and algorithm designer.
Your task is to write a Python function that generates a gate sequence for
quantum noise mitigation in silicon spin qubits.

IMPORTANT RULES:
1. You MUST return ONLY the function code, nothing else
2. The function signature MUST be exactly:
   def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
3. The function MUST return a list of Gate objects
4. Gate constructor: Gate(gate_type, qubits, params_dict)
   - gate_type: "h", "rx", "ry", "rz", "cnot", "cz", "swap", "identity"
   - qubits: list of qubit indices (e.g., [0] for single, [0, 1] for two-qubit)
   - params_dict: optional, e.g., {"theta": 3.14159}
5. You can import numpy as np inside the function
6. The Gate class is already imported, do NOT redefine it
7. nn_correlation is the nearest-neighbor noise correlation (0 to 1)
8. qubit_spacing_nm is the physical distance between qubits in nanometers
9. corr_length_nm is the TLF noise correlation length in nanometers
10. target_gate can be "bell", "cnot", or "ghz"
11. Keep the sequence SHORT - each gate adds noise, so fewer gates = less noise
12. Use the noise correlation parameters to design adaptive strategies

PHYSICS CONTEXT:
- Silicon spin qubits suffer from TLF (two-level fluctuator) charge noise
- Noise on nearby qubits is CORRELATED with correlation ~ exp(-d/l_c)
- Correlated noise can be partially cancelled by symmetric operations
- The DFS (decoherence-free subspace) encoding protects against global noise
- Dynamical decoupling (DD) sequences can refocus quasi-static noise
"""


def _call_ollama(
    prompt: str,
    max_retries: int = 2,
) -> Optional[str]:
    """Call local Ollama LLM (zero rate limits).

    Parameters
    ----------
    prompt : str
        User prompt.
    max_retries : int
        Number of retries on failure.

    Returns
    -------
    str or None
        LLM response text, or None on failure.
    """
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 1500,
                        "repeat_penalty": 1.1,
                    },
                },
                timeout=300,  # Local LLM can be slow on CPU
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("response", "")
                if text:
                    # Log performance metrics
                    eval_count = data.get("eval_count", 0)
                    eval_dur = data.get("eval_duration", 1) / 1e9
                    tok_per_s = eval_count / max(eval_dur, 0.001)
                    logger.info(
                        f"Ollama: {eval_count} tokens in {eval_dur:.1f}s "
                        f"({tok_per_s:.1f} tok/s)"
                    )
                    return text
                else:
                    logger.warning(f"Ollama returned empty (attempt {attempt})")
            else:
                logger.warning(
                    f"Ollama HTTP {resp.status_code} (attempt {attempt})"
                )
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable, skipping")
            return None  # Don't retry if server is down
        except Exception as e:
            logger.warning(f"Ollama error (attempt {attempt}): {e}")
            time.sleep(2)

    return None


def _call_gemini(
    prompt: str,
    max_retries: int = 2,
) -> Optional[str]:
    """Call Gemini API directly.

    Parameters
    ----------
    prompt : str
        User prompt.
    max_retries : int
        Number of retries on failure.

    Returns
    -------
    str or None
        LLM response text, or None on failure.
    """
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed, falling back to OpenRouter")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    for attempt in range(max_retries + 1):
        try:
            time.sleep(LLM_CALL_DELAY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
            )
            if response.text:
                return response.text
            else:
                logger.warning(f"Gemini returned empty response (attempt {attempt})")
                time.sleep(3)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = 10 * (attempt + 1)
                logger.info(f"Gemini rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Gemini error (attempt {attempt}): {e}")
                time.sleep(3)

    return None


def _call_openrouter(
    prompt: str,
    model: str,
    max_retries: int = 2,
) -> Optional[str]:
    """Call an LLM via OpenRouter API.

    Parameters
    ----------
    prompt : str
        User prompt.
    model : str
        Model identifier.
    max_retries : int
        Number of retries on failure.

    Returns
    -------
    str or None
        LLM response text, or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.8,
    }

    for attempt in range(max_retries + 1):
        try:
            time.sleep(max(LLM_CALL_DELAY, 8))  # Longer delay for OpenRouter
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=90,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait_time = 20 * (attempt + 1)
                logger.info(f"OpenRouter rate limited on {model}, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif resp.status_code == 404:
                return None
            else:
                logger.warning(f"OpenRouter failed ({model}): {resp.status_code}")
                time.sleep(5)
        except Exception as e:
            logger.warning(f"OpenRouter error ({model}): {e}")
            time.sleep(5)

    return None


def call_llm(
    prompt: str,
    model: str = "",
    max_retries: int = 2,
) -> Optional[str]:
    """Call an LLM - uses Ollama (primary), Gemini (secondary), OpenRouter (fallback).

    Parameters
    ----------
    prompt : str
        User prompt.
    model : str
        Model identifier (used for OpenRouter fallback).
    max_retries : int
        Number of retries on failure.

    Returns
    -------
    str or None
        LLM response text, or None on failure.
    """
    # Try local Ollama first (primary - zero rate limits)
    if USE_OLLAMA:
        result = _call_ollama(prompt, max_retries=max_retries)
        if result is not None:
            return result
        logger.info("Ollama failed, falling back to Gemini")

    # Try Gemini (secondary)
    if USE_GEMINI:
        result = _call_gemini(prompt, max_retries=max_retries)
        if result is not None:
            return result
        logger.info("Gemini failed, falling back to OpenRouter")

    # Fallback to OpenRouter
    if model and model != GEMINI_MODEL and OPENROUTER_KEY:
        return _call_openrouter(prompt, model, max_retries=max_retries)

    # Try primary OpenRouter models
    for or_model in _PRIMARY_MODELS[:2]:
        result = _call_openrouter(prompt, or_model, max_retries=1)
        if result is not None:
            return result

    return None


def extract_function(response: str) -> Optional[str]:
    """Extract the generate_gate_sequence function from LLM response.

    Parameters
    ----------
    response : str
        Raw LLM response text.

    Returns
    -------
    str or None
        Extracted function code, or None if extraction fails.
    """
    if response is None:
        return None

    # Try to find code block
    code = response
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    # Find the function definition
    lines = code.split("\n")
    func_start = None
    for i, line in enumerate(lines):
        if "def generate_gate_sequence" in line:
            func_start = i
            break

    if func_start is None:
        return None

    # Extract function body (everything indented after def)
    func_lines = [lines[func_start]]
    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if stripped == "":
            func_lines.append("")
            continue
        # Check if still inside function (indented or empty)
        if line.startswith("    ") or line.startswith("\t") or stripped == "":
            func_lines.append(line)
        elif stripped.startswith("def ") or stripped.startswith("class "):
            break  # New function/class = end of our function
        else:
            # Could be a continuation or end
            if i == func_start + 1:
                func_lines.append("    " + line)  # Fix missing indent
            else:
                break

    # Remove trailing empty lines
    while func_lines and func_lines[-1].strip() == "":
        func_lines.pop()

    if len(func_lines) < 3:
        return None

    return "\n".join(func_lines)


def compile_strategy(code: str) -> Optional[callable]:
    """Compile a strategy function from source code.

    Parameters
    ----------
    code : str
        Python source code containing generate_gate_sequence.

    Returns
    -------
    callable or None
        The compiled function, or None on error.
    """
    try:
        # Create execution namespace with Gate and numpy available
        import numpy as _np
        import math as _math
        namespace = {
            "Gate": Gate,
            "np": _np,
            "numpy": _np,
            "math": _math,
            "__builtins__": __builtins__,
        }
        exec(code, namespace)

        if "generate_gate_sequence" not in namespace:
            return None

        fn = namespace["generate_gate_sequence"]

        # Quick smoke test
        result = fn("bell", 2, 0.3, 108.0, 81.0)
        if not isinstance(result, list):
            return None
        if len(result) == 0:
            return None
        for g in result:
            if not hasattr(g, "gate_type"):
                return None

        return fn

    except Exception as e:
        logger.debug(f"Compile error: {e}")
        return None


# ======================================================================
# Individual (population member)
# ======================================================================

class Individual:
    """A single member of the evolutionary population."""

    def __init__(
        self,
        code: str,
        fitness: float = 0.0,
        source: str = "seed",
        model: str = "",
        generation: int = 0,
    ):
        self.code = code
        self.fitness = fitness
        self.source = source
        self.model = model
        self.generation = generation

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "fitness": self.fitness,
            "source": self.source,
            "model": self.model,
            "generation": self.generation,
        }


# ======================================================================
# AlphaEvolve v2 Engine
# ======================================================================

class AlphaEvolveV2:
    """LLM-driven code evolution engine using free OpenRouter models.

    Parameters
    ----------
    population_size : int
        Size of the population.
    n_generations : int
        Number of generations to evolve.
    max_llm_calls : int
        Maximum total LLM API calls.
    noise_amplitude : float
        Noise strength for fitness evaluation.
    n_qubits : int
        Number of qubits for evaluation.
    output_dir : str
        Directory for saving results.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        population_size: int = 10,
        n_generations: int = 15,
        max_llm_calls: int = 100,
        noise_amplitude: float = 0.3,
        n_qubits: int = 2,
        output_dir: str = "results/evolve_v2",
        seed: int = 42,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_llm_calls = max_llm_calls
        self.noise_amplitude = noise_amplitude
        self.n_qubits = n_qubits
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Fitness evaluator with harder noise
        self.evaluator = FitnessEvaluator(
            n_qubits=n_qubits,
            target_gates=["bell", "cnot"],
            n_noise_samples=100,
            noise_amplitude=noise_amplitude,
            qubit_spacing_nm=108.0,
            tlf_correlation_length_nm=81.0,
            max_sequence_length=50,
            timeout_seconds=5.0,
        )

        self.population: List[Individual] = []
        self.best_ever: Optional[Individual] = None
        self.llm_calls = 0
        self.history = []

    def _evaluate(self, code: str) -> float:
        """Evaluate a strategy's fitness.

        Parameters
        ----------
        code : str
            Python source code of the strategy.

        Returns
        -------
        float
            Fitness score (0 to 1).
        """
        fn = compile_strategy(code)
        if fn is None:
            return 0.0

        try:
            result = self.evaluator.evaluate(fn, seed=self.seed)
            return result["fitness"] if result["valid"] else 0.0
        except Exception:
            return 0.0

    def _initialize_population(self):
        """Initialize population with seed strategies."""
        for name, code in SEED_STRATEGIES.items():
            fitness = self._evaluate(code)
            ind = Individual(
                code=code,
                fitness=fitness,
                source=f"seed:{name}",
                generation=0,
            )
            self.population.append(ind)
            logger.info(f"Seed '{name}': fitness={fitness:.4f}")

        # Fill remaining slots with copies of best seed
        best_seed = max(self.population, key=lambda x: x.fitness)
        while len(self.population) < self.population_size:
            ind = Individual(
                code=best_seed.code,
                fitness=best_seed.fitness,
                source="seed:clone",
                generation=0,
            )
            self.population.append(ind)

    def _build_mutation_prompt(
        self,
        parent: Individual,
        best: Individual,
    ) -> str:
        """Build the mutation prompt for the LLM.

        Parameters
        ----------
        parent : Individual
            Parent strategy to mutate.
        best : Individual
            Best strategy found so far.

        Returns
        -------
        str
            Prompt for the LLM.
        """
        prompt = f"""Here is a gate sequence strategy for quantum noise mitigation.
It achieves a fidelity of {parent.fitness:.4f} under TLF-correlated noise
(amplitude={self.noise_amplitude}, correlation_length=81nm, qubit_spacing=108nm).

CURRENT STRATEGY:
```python
{parent.code}
```

The BEST strategy so far achieves fidelity {best.fitness:.4f}:
```python
{best.code}
```

Your task: Create an IMPROVED version of the current strategy that achieves
HIGHER fidelity. Think about:
- Can you exploit the noise correlation (nn_correlation parameter)?
- Would dynamical decoupling help? (pairs of pi-pulses that refocus noise)
- Can you reduce the total gate count while preserving the target operation?
- Would adaptive rotations based on qubit_spacing_nm / corr_length_nm help?
- Can you use symmetry to cancel correlated errors?

IMPORTANT: Each additional gate adds ~{self.noise_amplitude:.2f} rad of noise,
so the strategy must balance noise cancellation vs. added noise.

Return ONLY the improved function code, starting with:
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
"""
        return prompt

    def _mutate(
        self,
        parent: Individual,
        best: Individual,
        generation: int,
    ) -> Optional[Individual]:
        """Generate a mutant using an LLM.

        Parameters
        ----------
        parent : Individual
            Parent to mutate.
        best : Individual
            Best individual for reference.
        generation : int
            Current generation number.

        Returns
        -------
        Individual or None
            New individual, or None on failure.
        """
        if self.llm_calls >= self.max_llm_calls:
            return None

        # Select model (round-robin across free models)
        model = FREE_MODELS[self.llm_calls % len(FREE_MODELS)]

        prompt = self._build_mutation_prompt(parent, best)

        self.llm_calls += 1
        response = call_llm(prompt, model)

        if response is None:
            return None

        code = extract_function(response)
        if code is None:
            # Retry with error feedback
            retry_prompt = f"""Your previous response could not be parsed.
Please return ONLY the Python function, starting with:
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):

Make sure to:
1. Use proper 4-space indentation
2. Return a list of Gate objects
3. Gate constructor: Gate(gate_type, qubits, params_dict)
4. Do NOT redefine the Gate class

Previous response (failed to parse):
{response[:500]}
"""
            self.llm_calls += 1
            response = call_llm(retry_prompt, model)
            if response:
                code = extract_function(response)

        if code is None:
            return None

        fitness = self._evaluate(code)

        return Individual(
            code=code,
            fitness=fitness,
            source="mutation",
            model=model.split("/")[-1],
            generation=generation,
        )

    def evolve(self) -> Dict:
        """Run the evolutionary loop.

        Returns
        -------
        dict
            Results including best strategy and history.
        """
        logger.info(
            f"AlphaEvolve v2: pop={self.population_size}, "
            f"gens={self.n_generations}, noise={self.noise_amplitude}, "
            f"max_llm_calls={self.max_llm_calls}"
        )

        # Initialize
        self._initialize_population()
        self.best_ever = max(self.population, key=lambda x: x.fitness)

        for gen in range(1, self.n_generations + 1):
            if self.llm_calls >= self.max_llm_calls:
                logger.info(f"LLM budget exhausted at gen {gen}")
                break

            gen_start = time.time()
            new_individuals = []

            # Generate mutants
            n_mutants = min(
                self.population_size,
                self.max_llm_calls - self.llm_calls,
            )

            for i in range(n_mutants):
                # Tournament selection for parent
                candidates = self.rng.choice(
                    len(self.population), size=3, replace=False
                )
                parent = max(
                    [self.population[c] for c in candidates],
                    key=lambda x: x.fitness,
                )

                mutant = self._mutate(parent, self.best_ever, gen)
                if mutant is not None:
                    new_individuals.append(mutant)

            # Selection: keep best from combined pool
            combined = self.population + new_individuals
            combined.sort(key=lambda x: x.fitness, reverse=True)
            self.population = combined[:self.population_size]

            # Update best
            gen_best = self.population[0]
            if gen_best.fitness > self.best_ever.fitness:
                self.best_ever = gen_best
                logger.info(
                    f"  NEW BEST at gen {gen}: {gen_best.fitness:.4f} "
                    f"(model={gen_best.model})"
                )

            # Log generation stats
            valid_new = [x for x in new_individuals if x.fitness > 0]
            gen_time = time.time() - gen_start

            gen_stats = {
                "generation": gen,
                "best_fitness": self.best_ever.fitness,
                "gen_best": gen_best.fitness,
                "gen_mean": np.mean([x.fitness for x in self.population]),
                "new_valid": len(valid_new),
                "new_total": len(new_individuals),
                "llm_calls": self.llm_calls,
                "time_seconds": gen_time,
            }
            self.history.append(gen_stats)

            logger.info(
                f"Gen {gen}: best={self.best_ever.fitness:.4f}, "
                f"mean={gen_stats['gen_mean']:.4f}, "
                f"valid={len(valid_new)}/{len(new_individuals)}, "
                f"calls={self.llm_calls}/{self.max_llm_calls}, "
                f"time={gen_time:.1f}s"
            )

        # Save results
        results = {
            "best_fitness": float(self.best_ever.fitness),
            "best_code": self.best_ever.code,
            "best_model": self.best_ever.model,
            "best_generation": self.best_ever.generation,
            "total_llm_calls": self.llm_calls,
            "history": self.history,
        }

        # Save best strategy
        best_file = self.output_dir / "best_strategy_v2.py"
        with open(best_file, "w") as f:
            f.write(f"# Best strategy found by AlphaEvolve v2\n")
            f.write(f"# Fitness: {self.best_ever.fitness:.6f}\n")
            f.write(f"# Model: {self.best_ever.model}\n")
            f.write(f"# Generation: {self.best_ever.generation}\n\n")
            f.write(f"from fitness import Gate\n\n")
            f.write(self.best_ever.code)

        # Save full results
        results_file = self.output_dir / "evolve_v2_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

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
    print("AlphaEvolve v2 - Free LLMs + Harder Noise")
    print("=" * 70)
    print(f"Models: {', '.join(m.split('/')[-1] for m in FREE_MODELS)}")
    print(f"Noise amplitude: 0.3 rad/gate (6x harder than v1)")
    print()

    engine = AlphaEvolveV2(
        population_size=8,
        n_generations=10,
        max_llm_calls=50,
        noise_amplitude=0.3,
        n_qubits=2,
        output_dir="/home/ubuntu/siliqun/alphaevolve/results/evolve_v2",
        seed=42,
    )

    results = engine.evolve()

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Best model: {results['best_model']}")
    print(f"Best generation: {results['best_generation']}")
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"\nBest strategy code:")
    print(results['best_code'])
