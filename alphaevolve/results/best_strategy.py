# Best strategy found by AlphaEvolve
# Generation: 0
# Fitness: 0.997401
# Raw fidelity: 0.998901
# Model: seed:standard
# Per-target: {'bell': 0.9978022805406266, 'cnot': 1.0}
# Seq lengths: {'bell': 2, 'cnot': 1}

import numpy as np
from skeleton import Gate


class EvolvedStrategy:
    """LLM-evolved noise mitigation strategy for silicon spin qubits.
    
    This strategy generates gate sequences that implement target quantum
    operations while mitigating TLF-correlated charge noise. The noise
    has spatial correlations with correlation length l_c (nm) and the
    qubits are spaced at qubit_spacing_nm apart.
    
    Key physics:
    - Correlated charge noise causes correlated dephasing on nearby qubits
    - DFS encoding protects against global (uniform) noise
    - Residual noise from non-uniform (gradient) components causes leakage
    - Echo pulses can refocus some correlated errors
    - The noise correlation between neighbors is exp(-d/l_c) where d is distance
    
    Available gates:
    - Gate("rx", (q,), {"theta": angle})  - X rotation
    - Gate("ry", (q,), {"theta": angle})  - Y rotation  
    - Gate("rz", (q,), {"theta": angle})  - Z rotation
    - Gate("h", (q,))                     - Hadamard
    - Gate("cnot", (q1, q2))              - CNOT
    - Gate("cz", (q1, q2))               - Controlled-Z
    - Gate("swap", (q1, q2))              - SWAP
    - Gate("sqrt_swap", (q1, q2))         - sqrt(SWAP) (native to exchange qubits)
    - Gate("identity", (q,))              - Identity (idle)
    
    The goal is to maximize gate fidelity under correlated noise.
    """
    
    @staticmethod
    def generate_gate_sequence(
        target_gate: str,
        n_qubits: int,
        noise_correlation: float = 0.0,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
    ) -> list:
        """Generate a noise-mitigating gate sequence."""
        if target_gate == "bell":
            return [
                Gate("h", (0,)),
                Gate("cnot", (0, 1)),
            ]
        elif target_gate == "ghz":
            seq = [Gate("h", (0,))]
            for q in range(1, n_qubits):
                seq.append(Gate("cnot", (0, q)))
            return seq
        elif target_gate == "cnot":
            return [Gate("cnot", (0, 1))]
        elif target_gate == "swap":
            return [
                Gate("cnot", (0, 1)),
                Gate("cnot", (1, 0)),
                Gate("cnot", (0, 1)),
            ]
        elif target_gate == "h":
            return [Gate("h", (0,))]
        else:
            return [Gate("identity", (0,))]

