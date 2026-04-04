"""
Prompt Sampler for AlphaEvolve.

Constructs prompts optimised for local code models (7B-14B) running
on the user's RTX 2070 GPU.

Key design choices:
  - Short, explicit system prompts with concrete examples
  - Always request full function rewrite (no diff format)
  - Include a working example in every prompt
  - Minimal prose, maximum code context
  - Supports 2-qubit (Bell) and 3+-qubit (GHZ) targets
"""

from __future__ import annotations

import logging
import textwrap
from typing import Dict, List, Optional

import numpy as np

from program_database import Program

logger = logging.getLogger("aedb.prompt_sampler")


# ──────────────────────────────────────────────────────────────────────
# System prompt (single, clear, model-friendly)
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a Python code generator. You write ONLY Python code, no explanations.

TASK: Write a function that generates a quantum gate sequence to resist noise.

RULES:
1. Function signature MUST be exactly:
   def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
2. Return a list of Gate objects
3. Gate(gate_type, qubits) or Gate(gate_type, qubits, {{"theta": float}})
4. gate_type is one of: "h", "rx", "ry", "rz", "cnot", "cz", "swap", "identity"
5. qubits is a list of ints, e.g. [0] for single-qubit, [0,1] for two-qubit
6. You may use: import numpy as np
7. Do NOT define the Gate class
8. Do NOT add any text before or after the function
9. Each gate adds {noise_amp:.1f} rad of noise, so fewer gates = better
10. target_gate can be "bell" (2-qubit), "cnot" (2-qubit), or "ghz" (n-qubit)
11. For "ghz": create GHZ state |000...0> + |111...1> using H + chain of CNOTs
12. n_qubits can be 2, 3, or more. Use it to build the correct circuit.

EXAMPLE of a valid function:
```python
def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    import numpy as np
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
# Mutation hints (short, actionable, GHZ-aware)
# ──────────────────────────────────────────────────────────────────────

_HINTS = [
    # Correlation-aware strategies
    "Add rz rotations before/after gates to cancel correlated phase errors. The angle should depend on nn_correlation.",
    "When nn_correlation > 0.5, add echo pulses (rx with angle=pi) to refocus noise.",
    "Use qubit_spacing_nm / corr_length_nm ratio to decide whether to add correction gates.",
    "Try pre-rotating qubits with ry gates whose angle depends on nn_correlation * pi.",
    "For high correlation, try a symmetric sequence: rotate, apply gate, rotate back.",
    "Try conditional logic: if nn_correlation > 0.7 use one strategy, else use another.",
    "Add rz(angle=nn_correlation * pi/4) before each cnot to pre-compensate noise.",
    # Gate count optimization
    "Reduce gate count by merging consecutive single-qubit rotations.",
    "For GHZ state, try a tree-shaped CNOT pattern instead of a chain (log depth).",
    "Use fewer correction gates when nn_correlation is low (noise is uncorrelated).",
    # GHZ-specific strategies
    "For GHZ with 3+ qubits, add rz corrections between each CNOT based on qubit distance.",
    "Try a fan-out pattern: H on qubit 0, then CNOT(0,1), CNOT(0,2), ... instead of chain.",
    "For GHZ, add ry pre-rotations that scale with qubit index to compensate accumulated noise.",
    "Try adding identity gates as barriers between CNOT layers to separate noise correlations.",
    "For GHZ, use alternating CNOT directions: (0,1), (2,1), (2,3) to balance noise.",
    # Advanced strategies
    "Try a completely different gate decomposition for the target state.",
    "Add dynamical decoupling: rx(pi), wait, rx(pi) between each CNOT pair.",
    "Use cz gates instead of cnot gates and add hadamard corrections.",
    "Try composite pulse sequences: replace each gate with a sequence that cancels systematic errors.",
    "Add phase corrections that depend on both nn_correlation and qubit_spacing_nm/corr_length_nm.",
]


class PromptSampler:
    """Constructs short, explicit prompts for local code models.

    Parameters
    ----------
    noise_amplitude : float
        Noise strength per gate.
    diff_probability : float
        Ignored (always full rewrite for local models).
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

        # Build user prompt -- keep it SHORT for local models
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

        # Best program if different from parent
        if best_program and best_program.primary_score > parent.primary_score + 0.01:
            best_uid = best_program.uid if hasattr(best_program, 'uid') else id(best_program)
            if best_uid not in shown:
                parts.append(f"\n# Best known (fidelity={best_program.primary_score:.4f}):")
                parts.append(f"```python\n{best_program.code.strip()}\n```")

        # Hint
        hint = self.rng.choice(_HINTS)
        parts.append(f"\n# GOAL: Improve the function above. Hint: {hint}")
        parts.append(f"# Noise per gate: {self.noise_amplitude:.1f} rad. Fewer gates = less noise = higher fidelity.")
        parts.append(f"# nn_correlation ranges from 0 to 1. Use it to adapt the strategy.")
        parts.append(f"# Must handle target_gate='ghz' with n_qubits=3 or more.")
        parts.append(f"\n# Write the improved function below. ONLY the function, nothing else:")

        user_prompt = "\n".join(parts)

        return system, user_prompt, False  # never diff

    def apply_diff(self, parent_code: str, diff_text: str) -> Optional[str]:
        """Not used in local model mode."""
        return None
