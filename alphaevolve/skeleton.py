"""
Evolvable code skeleton for AlphaEvolve-style noise mitigation discovery.

This module defines the GateSequenceStrategy class, which is the "genome"
that the LLM evolves. The class contains a single method:
    generate_gate_sequence(target_gate, n_qubits, noise_params) -> List[Gate]

The LLM mutates the body of this method to discover novel noise-mitigating
gate decompositions for silicon spin qubits under TLF-correlated noise.

Standard baselines (seed programs) are provided as starting points.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


# ======================================================================
# Gate representation
# ======================================================================

@dataclass
class Gate:
    """A single gate operation in the sequence.
    
    Parameters
    ----------
    gate_type : str
        One of: "rx", "ry", "rz", "cnot", "cz", "swap", "sqrt_swap",
        "exchange", "h", "identity", "barrier"
    qubits : tuple
        Qubit indices this gate acts on. Single-qubit: (q,), Two-qubit: (q1, q2)
    params : dict
        Gate parameters. For rotations: {"theta": float}.
        For exchange: {"J": float, "t": float}.
    """
    gate_type: str
    qubits: tuple
    params: dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


# ======================================================================
# Seed strategies (baselines)
# ======================================================================

class StandardDecomposition:
    """Standard textbook gate decomposition (seed program).
    
    This is the simplest decomposition: just apply the target gate directly.
    No noise mitigation is attempted.
    """
    
    @staticmethod
    def generate_gate_sequence(
        target_gate: str,
        n_qubits: int,
        noise_correlation: float = 0.0,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
    ) -> List[Gate]:
        """Generate a gate sequence to implement the target operation.
        
        Parameters
        ----------
        target_gate : str
            Target gate to implement: "bell", "ghz", "cnot", "swap", "h"
        n_qubits : int
            Number of encoded qubits in the system.
        noise_correlation : float
            Nearest-neighbor charge noise correlation (0 to 1).
        qubit_spacing_nm : float
            Physical distance between neighboring qubits (nm).
        tlf_correlation_length_nm : float
            TLF charge noise correlation length (nm).
            
        Returns
        -------
        List[Gate]
            Ordered list of gates to apply.
        """
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


class EchoDecomposition:
    """Spin echo decomposition baseline.
    
    Wraps the target gate with echo pulses to refocus correlated dephasing.
    This is a well-known technique in NMR and quantum computing.
    """
    
    @staticmethod
    def generate_gate_sequence(
        target_gate: str,
        n_qubits: int,
        noise_correlation: float = 0.0,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
    ) -> List[Gate]:
        if target_gate == "bell":
            return [
                # Echo on qubit 0 before Hadamard
                Gate("rx", (0,), {"theta": np.pi}),
                Gate("rx", (0,), {"theta": np.pi}),
                Gate("h", (0,)),
                # Echo on both qubits before CNOT
                Gate("rx", (0,), {"theta": np.pi}),
                Gate("rx", (1,), {"theta": np.pi}),
                Gate("cnot", (0, 1)),
                Gate("rx", (0,), {"theta": np.pi}),
                Gate("rx", (1,), {"theta": np.pi}),
            ]
        elif target_gate == "cnot":
            return [
                Gate("rx", (0,), {"theta": np.pi}),
                Gate("rx", (1,), {"theta": np.pi}),
                Gate("cnot", (0, 1)),
                Gate("rx", (0,), {"theta": np.pi}),
                Gate("rx", (1,), {"theta": np.pi}),
            ]
        else:
            return StandardDecomposition.generate_gate_sequence(
                target_gate, n_qubits, noise_correlation,
                qubit_spacing_nm, tlf_correlation_length_nm,
            )


class CorrelationAwareDecomposition:
    """Correlation-aware decomposition baseline.
    
    Uses the noise correlation structure to insert targeted refocusing
    pulses. This is a simple heuristic: if correlation is high, add
    more echo pulses; if low, use standard decomposition.
    """
    
    @staticmethod
    def generate_gate_sequence(
        target_gate: str,
        n_qubits: int,
        noise_correlation: float = 0.0,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
    ) -> List[Gate]:
        # Compute correlation-dependent echo depth
        echo_depth = int(np.ceil(noise_correlation * 4))  # 0-4 echo layers
        
        if target_gate == "bell":
            seq = []
            # Pre-gate echo layers
            for _ in range(echo_depth):
                seq.append(Gate("rz", (0,), {"theta": np.pi}))
                seq.append(Gate("rz", (1,), {"theta": np.pi}))
            seq.append(Gate("h", (0,)))
            # Mid-gate echo layers
            for _ in range(echo_depth):
                seq.append(Gate("rz", (0,), {"theta": np.pi}))
                seq.append(Gate("rz", (1,), {"theta": np.pi}))
            seq.append(Gate("cnot", (0, 1)))
            # Post-gate echo layers
            for _ in range(echo_depth):
                seq.append(Gate("rz", (0,), {"theta": np.pi}))
                seq.append(Gate("rz", (1,), {"theta": np.pi}))
            return seq
        else:
            return StandardDecomposition.generate_gate_sequence(
                target_gate, n_qubits, noise_correlation,
                qubit_spacing_nm, tlf_correlation_length_nm,
            )


# ======================================================================
# Strategy template for LLM evolution
# ======================================================================

STRATEGY_TEMPLATE = '''
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
{METHOD_BODY}
'''

# Default method body (seed) - same as StandardDecomposition
DEFAULT_METHOD_BODY = '''        """Generate a noise-mitigating gate sequence."""
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
'''
