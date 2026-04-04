"""
Fitness evaluator for AlphaEvolve-style noise mitigation discovery.

Uses SiliQun's physics engine to evaluate gate sequences under
TLF-correlated charge noise. The fitness score is the average gate
fidelity over multiple noise realizations.

This module does NOT require the full SiliQun gym environment.
Instead, it directly uses the physics primitives (gates, noise,
DFS encoding) for maximum speed.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
import time
import traceback

from skeleton import Gate


# ======================================================================
# Gate matrix library (standalone, no SiliQun dependency for PoC)
# ======================================================================

PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _cnot() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )


def _cz() -> np.ndarray:
    return np.diag([1, 1, 1, -1]).astype(np.complex128)


def _swap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def _sqrt_swap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0],
         [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
         [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def gate_to_matrix(gate: Gate, n_qubits: int) -> np.ndarray:
    """Convert a Gate object to a full 2^n x 2^n unitary matrix.
    
    Parameters
    ----------
    gate : Gate
        The gate to convert.
    n_qubits : int
        Total number of qubits in the system.
        
    Returns
    -------
    np.ndarray
        The 2^n x 2^n unitary matrix.
    """
    dim = 2 ** n_qubits
    
    # Get the base gate matrix
    if gate.gate_type == "rx":
        base = _rx(gate.params.get("theta", 0.0))
    elif gate.gate_type == "ry":
        base = _ry(gate.params.get("theta", 0.0))
    elif gate.gate_type == "rz":
        base = _rz(gate.params.get("theta", 0.0))
    elif gate.gate_type == "h":
        base = _hadamard()
    elif gate.gate_type == "cnot":
        base = _cnot()
    elif gate.gate_type == "cz":
        base = _cz()
    elif gate.gate_type == "swap":
        base = _swap()
    elif gate.gate_type == "sqrt_swap":
        base = _sqrt_swap()
    elif gate.gate_type == "identity":
        return np.eye(dim, dtype=np.complex128)
    elif gate.gate_type == "barrier":
        return np.eye(dim, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown gate type: {gate.gate_type}")
    
    # Embed into full Hilbert space
    if base.shape == (2, 2):
        # Single-qubit gate
        q = gate.qubits[0]
        ops = [PAULI_I] * n_qubits
        ops[q] = base
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    elif base.shape == (4, 4):
        # Two-qubit gate
        q1, q2 = gate.qubits
        if abs(q2 - q1) != 1:
            # Non-adjacent qubits: use SWAP network
            # For simplicity, only support adjacent qubits in PoC
            # For non-adjacent, we decompose via SWAP routing
            return _embed_two_qubit_gate(base, q1, q2, n_qubits)
        else:
            return _embed_two_qubit_gate(base, q1, q2, n_qubits)
    else:
        raise ValueError(f"Unsupported gate matrix shape: {base.shape}")


def _embed_two_qubit_gate(
    base: np.ndarray, q1: int, q2: int, n_qubits: int
) -> np.ndarray:
    """Embed a 4x4 two-qubit gate into the full Hilbert space."""
    dim = 2 ** n_qubits
    
    # Ensure q1 < q2 for consistent ordering
    if q1 > q2:
        # Swap and apply SWAP conjugation
        swap_mat = _swap()
        base = swap_mat @ base @ swap_mat
        q1, q2 = q2, q1
    
    # Build the full matrix using tensor products
    result = np.zeros((dim, dim), dtype=np.complex128)
    
    # Iterate over computational basis states
    for i in range(dim):
        for j in range(dim):
            # Extract the bits at positions q1 and q2
            i_q1 = (i >> (n_qubits - 1 - q1)) & 1
            i_q2 = (i >> (n_qubits - 1 - q2)) & 1
            j_q1 = (j >> (n_qubits - 1 - q1)) & 1
            j_q2 = (j >> (n_qubits - 1 - q2)) & 1
            
            # Check if all other bits match
            i_rest = i & ~((1 << (n_qubits - 1 - q1)) | (1 << (n_qubits - 1 - q2)))
            j_rest = j & ~((1 << (n_qubits - 1 - q1)) | (1 << (n_qubits - 1 - q2)))
            
            if i_rest == j_rest:
                # Look up the 4x4 matrix element
                row = i_q1 * 2 + i_q2
                col = j_q1 * 2 + j_q2
                result[i, j] = base[row, col]
    
    return result


# ======================================================================
# TLF Correlated Noise Model
# ======================================================================

class TLFNoiseModel:
    """TLF-correlated charge noise model for fitness evaluation.
    
    Implements spatially correlated dephasing based on the TLF model
    from Rojas-Arias et al. (arXiv:2603.03051).
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    qubit_spacing_nm : float
        Distance between neighboring qubits (nm).
    tlf_correlation_length_nm : float
        TLF charge noise correlation length (nm).
    noise_amplitude : float
        Dephasing noise amplitude (radians per gate).
    seed : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_qubits: int,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
        noise_amplitude: float = 0.01,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.qubit_spacing_nm = qubit_spacing_nm
        self.tlf_correlation_length_nm = tlf_correlation_length_nm
        self.noise_amplitude = noise_amplitude
        self.rng = np.random.RandomState(seed)
        
        # Build spatial correlation matrix
        self.correlation_matrix = self._build_correlation_matrix()
        
        # Cholesky decomposition for correlated sampling
        # Add small regularization for numerical stability
        C = self.correlation_matrix + 1e-10 * np.eye(n_qubits)
        self.cholesky_L = np.linalg.cholesky(C)
    
    def _build_correlation_matrix(self) -> np.ndarray:
        """Build the spatial correlation matrix from TLF model.
        
        C(i,j) = exp(-|r_i - r_j| / l_c)
        
        For a linear chain: |r_i - r_j| = |i - j| * qubit_spacing_nm
        """
        C = np.zeros((self.n_qubits, self.n_qubits))
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                dist = abs(i - j) * self.qubit_spacing_nm
                C[i, j] = np.exp(-dist / self.tlf_correlation_length_nm)
        return C
    
    def sample_noise(self) -> np.ndarray:
        """Sample correlated dephasing angles for all qubits.
        
        Returns
        -------
        np.ndarray
            Array of dephasing angles (radians) for each qubit.
        """
        # Sample independent standard normals
        z = self.rng.randn(self.n_qubits)
        # Apply Cholesky to get correlated samples
        correlated = self.cholesky_L @ z
        # Scale by noise amplitude
        return self.noise_amplitude * correlated
    
    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """Apply one round of correlated dephasing noise to the state.
        
        Each qubit gets a random Z-rotation with correlated angles.
        
        Parameters
        ----------
        state : np.ndarray
            State vector of dimension 2^n_qubits.
            
        Returns
        -------
        np.ndarray
            Noisy state vector.
        """
        angles = self.sample_noise()
        dim = 2 ** self.n_qubits
        
        # Apply Z-rotation to each qubit
        for q in range(self.n_qubits):
            rz_mat = _rz(angles[q])
            ops = [PAULI_I] * self.n_qubits
            ops[q] = rz_mat
            full_rz = ops[0]
            for op in ops[1:]:
                full_rz = np.kron(full_rz, op)
            state = full_rz @ state
        
        return state


# ======================================================================
# Target state builders
# ======================================================================

def build_target_state(target_gate: str, n_qubits: int) -> np.ndarray:
    """Build the ideal target state for a given gate/state preparation.
    
    Parameters
    ----------
    target_gate : str
        Target: "bell", "ghz", "cnot", "swap", "h"
    n_qubits : int
        Number of qubits.
        
    Returns
    -------
    np.ndarray
        Target state vector.
    """
    dim = 2 ** n_qubits
    zero_state = np.zeros(dim, dtype=np.complex128)
    zero_state[0] = 1.0  # |00...0>
    
    if target_gate == "bell":
        # |Bell> = (|00> + |11>) / sqrt(2)
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1.0 / np.sqrt(2)  # |00>
        state[3] = 1.0 / np.sqrt(2)  # |11>
        return state
    elif target_gate == "ghz":
        # |GHZ> = (|00...0> + |11...1>) / sqrt(2)
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 / np.sqrt(2)
        state[-1] = 1.0 / np.sqrt(2)
        return state
    elif target_gate == "cnot":
        # CNOT|00> = |00> (no change for |00> input)
        # Better: test with |10> -> |11>
        state = np.zeros(4, dtype=np.complex128)
        state[3] = 1.0  # |11> = CNOT|10>
        return state
    elif target_gate == "swap":
        # SWAP|01> = |10>
        state = np.zeros(4, dtype=np.complex128)
        state[2] = 1.0  # |10> = SWAP|01>
        return state
    elif target_gate == "h":
        # H|0> = |+>
        state = np.zeros(2, dtype=np.complex128)
        state[0] = 1.0 / np.sqrt(2)
        state[1] = 1.0 / np.sqrt(2)
        return state
    else:
        return zero_state


def build_initial_state(target_gate: str, n_qubits: int) -> np.ndarray:
    """Build the initial state that the gate sequence acts on.
    
    Parameters
    ----------
    target_gate : str
        Target operation.
    n_qubits : int
        Number of qubits.
        
    Returns
    -------
    np.ndarray
        Initial state vector.
    """
    dim = 2 ** n_qubits
    
    if target_gate in ("bell", "ghz", "h"):
        # Start from |00...0>
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0
        return state
    elif target_gate == "cnot":
        # Start from |10> to test CNOT
        state = np.zeros(4, dtype=np.complex128)
        state[2] = 1.0  # |10>
        return state
    elif target_gate == "swap":
        # Start from |01> to test SWAP
        state = np.zeros(4, dtype=np.complex128)
        state[1] = 1.0  # |01>
        return state
    else:
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0
        return state


# ======================================================================
# Fidelity computation
# ======================================================================

def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity between two pure states.
    
    F = |<psi1|psi2>|^2
    """
    overlap = np.abs(np.vdot(state1, state2)) ** 2
    return float(np.clip(overlap, 0.0, 1.0))


# ======================================================================
# Main fitness evaluator
# ======================================================================

class FitnessEvaluator:
    """Evaluates gate sequence strategies under TLF-correlated noise.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 2 for PoC).
    target_gates : list of str
        Target gates to evaluate on.
    n_noise_samples : int
        Number of noise realizations per evaluation.
    noise_amplitude : float
        Dephasing noise amplitude (radians per gate).
    qubit_spacing_nm : float
        Qubit spacing (nm).
    tlf_correlation_length_nm : float
        TLF correlation length (nm).
    max_sequence_length : int
        Maximum allowed gate sequence length (prevents bloat).
    timeout_seconds : float
        Maximum time for a single evaluation.
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        target_gates: List[str] = None,
        n_noise_samples: int = 100,
        noise_amplitude: float = 0.02,
        qubit_spacing_nm: float = 108.0,
        tlf_correlation_length_nm: float = 81.0,
        max_sequence_length: int = 50,
        timeout_seconds: float = 5.0,
    ):
        self.n_qubits = n_qubits
        self.target_gates = target_gates or ["bell", "cnot"]
        self.n_noise_samples = n_noise_samples
        self.noise_amplitude = noise_amplitude
        self.qubit_spacing_nm = qubit_spacing_nm
        self.tlf_correlation_length_nm = tlf_correlation_length_nm
        self.max_sequence_length = max_sequence_length
        self.timeout_seconds = timeout_seconds
        
        # Precompute noise correlation
        self.nn_correlation = np.exp(
            -qubit_spacing_nm / tlf_correlation_length_nm
        )
    
    def evaluate(
        self,
        strategy_fn: Callable,
        seed: int = 42,
    ) -> Dict:
        """Evaluate a gate sequence strategy.
        
        Parameters
        ----------
        strategy_fn : callable
            Function with signature:
            (target_gate, n_qubits, noise_correlation, 
             qubit_spacing_nm, tlf_correlation_length_nm) -> List[Gate]
        seed : int
            Random seed for noise sampling.
            
        Returns
        -------
        dict
            Evaluation results with keys:
            - "fitness": float (average fidelity across all targets)
            - "per_target": dict of per-target fidelities
            - "sequence_lengths": dict of sequence lengths per target
            - "wall_time": float (seconds)
            - "valid": bool (whether the strategy produced valid sequences)
            - "error": str or None
        """
        start_time = time.time()
        results = {
            "fitness": 0.0,
            "per_target": {},
            "sequence_lengths": {},
            "wall_time": 0.0,
            "valid": True,
            "error": None,
        }
        
        total_fidelity = 0.0
        n_targets = 0
        
        for target in self.target_gates:
            try:
                # Generate the gate sequence
                seq = strategy_fn(
                    target,
                    self.n_qubits,
                    self.nn_correlation,
                    self.qubit_spacing_nm,
                    self.tlf_correlation_length_nm,
                )
                
                # Validate sequence
                if not isinstance(seq, list):
                    results["valid"] = False
                    results["error"] = f"Strategy returned {type(seq)}, expected list"
                    break
                
                if len(seq) == 0:
                    results["valid"] = False
                    results["error"] = "Strategy returned empty sequence"
                    break
                
                if len(seq) > self.max_sequence_length:
                    results["valid"] = False
                    results["error"] = f"Sequence too long: {len(seq)} > {self.max_sequence_length}"
                    break
                
                # Validate all gates
                for g in seq:
                    if not isinstance(g, Gate):
                        results["valid"] = False
                        results["error"] = f"Invalid gate object: {g}"
                        break
                
                if not results["valid"]:
                    break
                
                results["sequence_lengths"][target] = len(seq)
                
                # Determine n_qubits for this target
                target_n = self.n_qubits
                if target in ("bell", "cnot", "swap"):
                    target_n = 2
                elif target == "h":
                    target_n = 1
                
                # Build target and initial states
                target_state = build_target_state(target, target_n)
                initial_state = build_initial_state(target, target_n)
                
                # Precompute gate matrices
                gate_matrices = []
                for g in seq:
                    gate_matrices.append(gate_to_matrix(g, target_n))
                
                # Evaluate over noise realizations
                noise_model = TLFNoiseModel(
                    n_qubits=target_n,
                    qubit_spacing_nm=self.qubit_spacing_nm,
                    tlf_correlation_length_nm=self.tlf_correlation_length_nm,
                    noise_amplitude=self.noise_amplitude,
                    seed=seed,
                )
                
                fidelities = []
                for trial in range(self.n_noise_samples):
                    state = initial_state.copy()
                    
                    # Apply gate sequence with noise after each gate
                    for mat in gate_matrices:
                        state = mat @ state
                        state = noise_model.apply_noise(state)
                    
                    # Compute fidelity
                    f = state_fidelity(state, target_state)
                    fidelities.append(f)
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        break
                
                avg_fidelity = np.mean(fidelities)
                results["per_target"][target] = float(avg_fidelity)
                total_fidelity += avg_fidelity
                n_targets += 1
                
            except Exception as e:
                results["valid"] = False
                results["error"] = f"Error on target '{target}': {str(e)}\n{traceback.format_exc()}"
                break
            
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                break
        
        if n_targets > 0:
            results["fitness"] = total_fidelity / n_targets
        
        results["wall_time"] = time.time() - start_time
        return results


# ======================================================================
# Quick test
# ======================================================================

if __name__ == "__main__":
    from skeleton import StandardDecomposition, EchoDecomposition, CorrelationAwareDecomposition
    
    evaluator = FitnessEvaluator(
        n_qubits=2,
        target_gates=["bell", "cnot"],
        n_noise_samples=200,
        noise_amplitude=0.02,
    )
    
    print("=" * 60)
    print("Fitness Evaluator Test")
    print("=" * 60)
    
    for name, strategy in [
        ("Standard", StandardDecomposition),
        ("Echo", EchoDecomposition),
        ("CorrelationAware", CorrelationAwareDecomposition),
    ]:
        result = evaluator.evaluate(strategy.generate_gate_sequence)
        print(f"\n{name}:")
        print(f"  Fitness:    {result['fitness']:.6f}")
        print(f"  Per-target: {result['per_target']}")
        print(f"  Seq lengths: {result['sequence_lengths']}")
        print(f"  Wall time:  {result['wall_time']:.4f}s")
        print(f"  Valid:      {result['valid']}")
        if result['error']:
            print(f"  Error:      {result['error']}")
