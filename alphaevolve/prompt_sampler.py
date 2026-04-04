"""
Prompt Sampler for AlphaEvolve.

Constructs prompts optimised for smaller (7B) code models running locally.
Key design choices for 7B compatibility:
  - Short, explicit system prompts with concrete examples
  - Always request full function rewrite (no diff format)
  - Include a working example in every prompt
  - Minimal prose, maximum code context
"""

from __future__ import annotations

import logging
import textwrap
from typing import Dict, List, Optional

import numpy as np

from program_database import Program

logger = logging.getLogger("aedb.prompt_sampler")


# ──────────────────────────────────────────────────────────────────────
# System prompt (single, clear, 7B-friendly)
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a Python code generator. You write ONLY Python code, no explanations.

TASK: Write a function that generates a quantum gate sequence to resist noise.

RULES:
1. Function signature MUST be exactly:
   def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
2. Return a list of Gate objects
3. Gate(gate_type, qubits) or Gate(gate_type, qubits, {{"angle": float}})
4. gate_type is one of: "h", "rx", "ry", "rz", "cnot", "cz", "swap", "identity"
5. qubits is a list of ints, e.g. [0] for single-qubit, [0,1] for two-qubit
6. You may use: import numpy as np
7. Do NOT define the Gate class
8. Do NOT add any text before or after the function
9. Each gate adds {noise_amp:.1f} rad of noise, so fewer gates = better

EXAMPLE of a valid function:
```python
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
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
```
""")


# ──────────────────────────────────────────────────────────────────────
# Mutation hints (short, actionable)
# ──────────────────────────────────────────────────────────────────────

_HINTS = [
    "Add rz rotations before/after gates to cancel correlated phase errors. The angle should depend on nn_correlation.",
    "When nn_correlation > 0.5, add echo pulses (rx with angle=pi) to refocus noise.",
    "Use qubit_spacing_nm / corr_length_nm ratio to decide whether to add correction gates.",
    "Try pre-rotating qubits with ry gates whose angle depends on nn_correlation * pi.",
    "For high correlation, try a symmetric sequence: rotate, apply gate, rotate back.",
    "Add identity gates as wait periods between operations when correlation is low.",
    "Try conditional logic: if nn_correlation > 0.7 use one strategy, else use another.",
    "Reduce gate count by merging consecutive single-qubit rotations.",
    "Add rz(angle=nn_correlation * pi/4) before each cnot to pre-compensate noise.",
    "Try a completely different gate decomposition for the bell state.",
]


class PromptSampler:
    """Constructs short, explicit prompts for 7B code models.

    Parameters
    ----------
    noise_amplitude : float
        Noise strength per gate.
    diff_probability : float
        Ignored (always full rewrite for 7B).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        noise_amplitude: float = 0.5,
        diff_probability: float = 0.0,  # ignored
        seed: int = 42,
    ):
        self.noise_amplitude = noise_amplitude
        self.rng = np.random.RandomState(seed)

    def build_prompt(
        self,
        parent: Program,
        inspirations: List[Program],
        best_program: Optional[Program] = None,
        generation: int = 0,
    ) -> tuple[str, str, bool]:
        """Build a prompt for the LLM.

        Returns (system_prompt, user_prompt, is_diff=False).
        """
        system = SYSTEM_PROMPT.format(noise_amp=self.noise_amplitude)

        # Build user prompt -- keep it SHORT for 7B
        parts = []

        # Parent code with score
        parts.append(f"# Current best function (fidelity={parent.primary_score:.4f}):")
        parts.append(f"```python\n{parent.code.strip()}\n```")

        # One inspiration (if available and different)
        shown = set()
        shown.add(parent.uid if hasattr(parent, 'uid') else id(parent))
        for insp in inspirations[:1]:
            uid = insp.uid if hasattr(insp, 'uid') else id(insp)
            if uid not in shown and insp.primary_score > 0.1:
                parts.append(f"\n# Alternative approach (fidelity={insp.primary_score:.4f}):")
                parts.append(f"```python\n{insp.code.strip()}\n```")
                shown.add(uid)

        # Hint
        hint = self.rng.choice(_HINTS)
        parts.append(f"\n# GOAL: Improve the function above. Hint: {hint}")
        parts.append(f"# Noise per gate: {self.noise_amplitude:.1f} rad. Fewer gates = less noise = higher fidelity.")
        parts.append(f"# nn_correlation ranges from 0 to 1. Use it to adapt the strategy.")
        parts.append(f"\n# Write the improved function below. ONLY the function, nothing else:")

        user_prompt = "\n".join(parts)

        return system, user_prompt, False  # never diff

    def apply_diff(self, parent_code: str, diff_text: str) -> Optional[str]:
        """Not used in 7B mode."""
        return None
