"""
Evaluation Cascade for AlphaEvolve.

Implements multi-stage evaluation with early pruning, following the
AlphaEvolve paper (Novikov et al., 2025, Section 2.4).

Stages:
  1. **Syntax check** -- compile and validate the function signature.
  2. **Quick fidelity** -- 10 noise samples, prune if below threshold.
  3. **Medium fidelity** -- 50 noise samples, prune if below threshold.
  4. **Full fidelity** -- 200 noise samples, definitive score.

Multi-metric scoring:
  - base_fidelity   : average fidelity across target gates
  - gate_efficiency  : penalty for excessive gate count
  - correlation_use  : bonus for using correlation parameters
  - combined         : weighted combination (primary score)

Supports both 2-qubit (Bell/CNOT) and 3+-qubit (GHZ) targets.
DEHB-learned parameters (noise_amplitude, n_samples, etc.) can be
injected at construction time.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fitness import FitnessEvaluator, Gate

logger = logging.getLogger("aedb.eval_cascade")


# ──────────────────────────────────────────────────────────────────────
# Code compilation and validation
# ──────────────────────────────────────────────────────────────────────

def compile_strategy(
    code: str,
    n_qubits: int = 3,
    target_gates: Optional[List[str]] = None,
) -> Optional[Callable]:
    """Compile a strategy string into a callable function.

    Validates that the function:
      - has the correct signature
      - returns a list of Gate objects
      - does not crash on the specified target gates

    Parameters
    ----------
    code : str
        Python source code containing generate_gate_sequence.
    n_qubits : int
        Number of qubits for smoke tests.
    target_gates : list of str, optional
        Gates to smoke-test. Defaults to ["ghz"] for n_qubits>=3,
        ["bell"] for n_qubits==2.
    """
    if not code or "def generate_gate_sequence" not in code:
        return None

    namespace = {"Gate": Gate, "np": np, "numpy": np}
    try:
        exec(code, namespace)
    except Exception:
        return None

    fn = namespace.get("generate_gate_sequence")
    if fn is None or not callable(fn):
        return None

    # Determine which gates to smoke-test
    if target_gates is None:
        target_gates = ["ghz"] if n_qubits >= 3 else ["bell"]

    # Smoke test each target gate
    for gate_name in target_gates:
        test_nq = n_qubits if gate_name == "ghz" else 2
        try:
            result = fn(gate_name, test_nq, 0.5, 108.0, 81.0)
            if not isinstance(result, list) or len(result) == 0:
                return None
            for g in result:
                if not hasattr(g, "gate_type"):
                    return None
        except Exception:
            return None

    return fn


def extract_function(response: str) -> Optional[str]:
    """Extract the generate_gate_sequence function from an LLM response."""
    if not response:
        return None

    # Try to find function in code blocks
    for marker in ["```python", "```"]:
        if marker in response:
            blocks = response.split(marker)
            for block in blocks[1:]:
                end = block.find("```")
                code = block[:end] if end != -1 else block
                if "def generate_gate_sequence" in code:
                    # Extract just the function
                    lines = code.split("\n")
                    func_lines = []
                    in_func = False
                    for line in lines:
                        if line.strip().startswith("def generate_gate_sequence"):
                            in_func = True
                        if in_func:
                            if (
                                func_lines
                                and line.strip()
                                and not line.startswith(" ")
                                and not line.startswith("\t")
                                and not line.strip().startswith("#")
                                and not line.strip().startswith("def generate_gate_sequence")
                            ):
                                break
                            func_lines.append(line)
                    if func_lines:
                        return "\n".join(func_lines)

    # Try raw extraction (no code blocks)
    if "def generate_gate_sequence" in response:
        lines = response.split("\n")
        func_lines = []
        in_func = False
        for line in lines:
            if line.strip().startswith("def generate_gate_sequence"):
                in_func = True
            if in_func:
                if (
                    func_lines
                    and line.strip()
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and not line.strip().startswith("#")
                ):
                    break
                func_lines.append(line)
        if func_lines:
            return "\n".join(func_lines)

    return None


# ──────────────────────────────────────────────────────────────────────
# Evaluation Cascade
# ──────────────────────────────────────────────────────────────────────

class EvaluationCascade:
    """Multi-stage evaluation with early pruning and multi-metric scoring.

    Supports both 2-qubit (Bell/CNOT) and 3+-qubit (GHZ) targets.
    DEHB-learned parameters can be injected at construction time.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (2 for Bell, 3+ for GHZ).
    target_gates : list of str
        Target gates to evaluate (e.g., ["bell","cnot"] or ["ghz"]).
    noise_amplitude : float
        Noise strength per gate (DEHB-learned).
    qubit_spacing_nm : float
        Physical qubit spacing (DEHB-learned).
    tlf_correlation_length_nm : float
        TLF noise correlation length (DEHB-learned).
    max_sequence_length : int
        Maximum allowed gate sequence length (DEHB-learned).
    stage_samples : list of int
        Number of noise samples per stage [quick, medium, full].
    stage_thresholds : list of float
        Minimum score to advance to the next stage.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        target_gates: List[str] = None,
        noise_amplitude: float = 0.5,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
        max_sequence_length: int = 50,
        stage_samples: List[int] = None,
        stage_thresholds: List[float] = None,
        seed: int = 42,
    ):
        # Default targets based on qubit count
        if target_gates is None:
            if n_qubits <= 2:
                target_gates = ["bell", "cnot"]
            else:
                target_gates = ["ghz"]

        if stage_samples is None:
            stage_samples = [10, 50, 200]
        if stage_thresholds is None:
            stage_thresholds = [0.0, 0.2, 0.4]

        self.n_qubits = n_qubits
        self.target_gates = target_gates
        self.noise_amplitude = noise_amplitude
        self.qubit_spacing_nm = qubit_spacing_nm
        self.tlf_correlation_length_nm = tlf_correlation_length_nm
        self.max_sequence_length = max_sequence_length
        self.stage_thresholds = stage_thresholds
        self.seed = seed

        # Create evaluators for each stage with increasing sample counts
        self._stage_configs = [
            {"n_samples": stage_samples[0], "label": "quick"},
            {"n_samples": stage_samples[1], "label": "medium"},
            {"n_samples": stage_samples[2], "label": "full"},
        ]

        self._evaluators = {}
        for cfg in self._stage_configs:
            self._evaluators[cfg["label"]] = FitnessEvaluator(
                n_qubits=n_qubits,
                target_gates=target_gates,
                n_noise_samples=cfg["n_samples"],
                noise_amplitude=noise_amplitude,
                qubit_spacing_nm=qubit_spacing_nm,
                tlf_correlation_length_nm=tlf_correlation_length_nm,
                max_sequence_length=max_sequence_length,
                timeout_seconds=10.0,
            )

        # Statistics
        self.stats = {
            "total": 0,
            "passed_compile": 0,
            "passed_quick": 0,
            "passed_medium": 0,
            "passed_full": 0,
        }

    def evaluate(
        self,
        code: str,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Callable]]:
        """Evaluate a strategy through the cascade.

        Returns
        -------
        scores : dict or None
            Multi-metric scores if the strategy passes all stages,
            or None if it fails at any stage.
        fn : callable or None
            The compiled function if successful.
        """
        self.stats["total"] += 1
        t0 = time.time()

        # Stage 0: Compile check (includes GHZ smoke test)
        fn = compile_strategy(code, n_qubits=self.n_qubits, target_gates=self.target_gates)
        if fn is None:
            logger.debug("Cascade: failed compile")
            return None, None
        self.stats["passed_compile"] += 1

        # Stage 1: Quick fidelity
        try:
            quick_result = self._evaluators["quick"].evaluate(fn, seed=self.seed)
            if not quick_result["valid"]:
                logger.debug("Cascade: invalid at quick stage")
                return None, None
            quick_score = quick_result["fitness"]
        except Exception:
            return None, None

        if quick_score < self.stage_thresholds[0]:
            logger.debug(f"Cascade: pruned at quick stage ({quick_score:.4f})")
            return None, None
        self.stats["passed_quick"] += 1

        # Stage 2: Medium fidelity
        try:
            med_result = self._evaluators["medium"].evaluate(fn, seed=self.seed)
            med_score = med_result["fitness"] if med_result["valid"] else 0.0
        except Exception:
            return None, None

        if len(self.stage_thresholds) > 1 and med_score < self.stage_thresholds[1]:
            logger.debug(f"Cascade: pruned at medium stage ({med_score:.4f})")
            return None, None
        self.stats["passed_medium"] += 1

        # Stage 3: Full fidelity
        try:
            full_result = self._evaluators["full"].evaluate(fn, seed=self.seed)
            full_score = full_result["fitness"] if full_result["valid"] else 0.0
        except Exception:
            return None, None

        if len(self.stage_thresholds) > 2 and full_score < self.stage_thresholds[2]:
            logger.debug(f"Cascade: pruned at full stage ({full_score:.4f})")
            return None, None
        self.stats["passed_full"] += 1

        # Compute multi-metric scores
        scores = self._compute_scores(code, fn, full_result)

        elapsed = time.time() - t0
        logger.debug(
            f"Cascade: passed all stages in {elapsed:.2f}s, "
            f"combined={scores['combined']:.4f}"
        )

        return scores, fn

    def _compute_scores(
        self,
        code: str,
        fn: Callable,
        full_result: Dict,
    ) -> Dict[str, float]:
        """Compute multi-metric scores for a strategy.

        Metrics:
          - base_fidelity   : raw fidelity from the evaluator
          - gate_efficiency  : 1.0 - penalty for excessive gates
          - correlation_use  : bonus for using correlation parameters
          - combined         : weighted combination
        """
        base_fidelity = full_result.get("fitness", 0.0)

        # Gate efficiency: test with the primary target
        primary_target = self.target_gates[0]
        try:
            gates = fn(primary_target, self.n_qubits, 0.5, 108.0, 81.0)
            gate_count = len(gates) if gates else 0
        except Exception:
            gate_count = 50

        # Optimal gate count depends on target:
        #   bell: 2 (H + CNOT)
        #   cnot: 1
        #   ghz(n): n (H + (n-1) CNOTs)
        if primary_target == "ghz":
            optimal = self.n_qubits  # H + (n-1) CNOTs
        elif primary_target == "bell":
            optimal = 2
        else:
            optimal = 1

        # Penalise above 2x optimal
        if gate_count <= optimal * 2:
            gate_efficiency = 1.0
        elif gate_count <= optimal * 4:
            excess = gate_count - optimal * 2
            gate_efficiency = 1.0 - 0.05 * excess
        else:
            gate_efficiency = max(0.3, 1.0 - 0.1 * (gate_count - optimal * 2))

        # Correlation usage: bonus for adaptive strategies
        corr_keywords = ["nn_correlation", "corr_length_nm", "qubit_spacing_nm"]
        code_body = code.split(":", 1)[-1] if ":" in code else code
        corr_usage = sum(1 for kw in corr_keywords if kw in code_body) / len(corr_keywords)

        # Per-gate scores from the evaluator
        per_gate = full_result.get("per_gate", {})

        # Combined score (primary metric for MAP-Elites)
        combined = (
            0.70 * base_fidelity
            + 0.15 * gate_efficiency
            + 0.15 * corr_usage
        )

        scores = {
            "combined": combined,
            "base_fidelity": base_fidelity,
            "gate_efficiency": gate_efficiency,
            "correlation_use": corr_usage,
            "gate_count": float(gate_count),
        }

        # Add per-gate scores
        for gate_name, gate_score in per_gate.items():
            scores[f"fid_{gate_name}"] = gate_score

        return scores

    def evaluate_quick(self, code: str) -> float:
        """Quick single-score evaluation (for DEHB/BRFD integration)."""
        fn = compile_strategy(code, n_qubits=self.n_qubits, target_gates=self.target_gates)
        if fn is None:
            return 0.0
        try:
            result = self._evaluators["quick"].evaluate(fn, seed=self.seed)
            return result["fitness"] if result["valid"] else 0.0
        except Exception:
            return 0.0

    def get_stats(self) -> Dict:
        """Return cascade statistics."""
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "compile_rate": self.stats["passed_compile"] / total,
            "quick_rate": self.stats["passed_quick"] / total,
            "medium_rate": self.stats["passed_medium"] / total,
            "full_rate": self.stats["passed_full"] / total,
        }
