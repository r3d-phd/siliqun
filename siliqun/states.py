"""
Target quantum state builders for silicon spin qubit experiments.

This module provides analytically exact state-vector representations of
the entangled target states used in QUASAR training and evaluation.
All states are returned as normalised complex128 NumPy arrays of length
2**n_qubits in the computational basis.

States
------
bell_state(n)       : Two-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
ghz_state(n)        : n-qubit GHZ state (|00…0⟩ + |11…1⟩)/√2
w_state(n)          : n-qubit W state (|100…0⟩ + |010…0⟩ + … + |00…01⟩)/√n
cluster_state(n)    : 1D cluster / graph state on a line of n qubits
dicke_state(n, k)   : Dicke state |D^n_k⟩ — equal superposition of all
                      n-qubit computational basis states with exactly k ones

References
----------
Dicke, R. H. (1954). Coherence in Spontaneous Radiation Processes.
    Physical Review, 93(1), 99–110.
Briegel, H. J., & Raussendorf, R. (2001). Persistent Entanglement in
    Arrays of Interacting Particles. Physical Review Letters, 86(5), 910–913.
"""

from __future__ import annotations

from itertools import combinations
from typing import Union

import numpy as np

__all__ = [
    "bell_state",
    "ghz_state",
    "w_state",
    "cluster_state",
    "dicke_state",
    "build_target_state",
    "compute_fidelity",
]


# ---------------------------------------------------------------------------
# Individual state builders
# ---------------------------------------------------------------------------

def bell_state(n: int = 2) -> np.ndarray:
    """Return the two-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.

    For n > 2 the state is embedded in the first two qubits with the
    remaining qubits in |0⟩, i.e. (|00⟩ + |11⟩)/√2 ⊗ |0…0⟩.

    Parameters
    ----------
    n : int
        Total number of qubits (must be ≥ 2).

    Returns
    -------
    np.ndarray
        Normalised state vector of length 2**n.

    Examples
    --------
    >>> sv = bell_state(2)
    >>> sv.shape
    (4,)
    >>> abs(sv[0] - 1/np.sqrt(2)) < 1e-12
    True
    """
    if n < 2:
        raise ValueError(f"bell_state requires n >= 2, got n={n}")
    sv = np.zeros(2**n, dtype=np.complex128)
    sv[0] = 1.0 / np.sqrt(2)          # |00…0⟩
    sv[2**(n-1) + 2**(n-2)] = 1.0 / np.sqrt(2)  # |11 0…0⟩ (qubits 0,1 set)
    # Simpler: index where first two bits are 11 in big-endian
    sv = np.zeros(2**n, dtype=np.complex128)
    sv[0b00 * 2**(n-2)] = 1.0 / np.sqrt(2)   # |00⟩|0…0⟩
    sv[0b11 * 2**(n-2)] = 1.0 / np.sqrt(2)   # |11⟩|0…0⟩
    return sv


def ghz_state(n: int) -> np.ndarray:
    """Return the n-qubit GHZ state (|0…0⟩ + |1…1⟩)/√2.

    Parameters
    ----------
    n : int
        Number of qubits (must be ≥ 2).

    Returns
    -------
    np.ndarray
        Normalised state vector of length 2**n.

    Examples
    --------
    >>> sv = ghz_state(3)
    >>> abs(sv[0] - 1/np.sqrt(2)) < 1e-12
    True
    >>> abs(sv[-1] - 1/np.sqrt(2)) < 1e-12
    True
    """
    if n < 2:
        raise ValueError(f"ghz_state requires n >= 2, got n={n}")
    sv = np.zeros(2**n, dtype=np.complex128)
    sv[0] = 1.0 / np.sqrt(2)        # |00…0⟩
    sv[2**n - 1] = 1.0 / np.sqrt(2) # |11…1⟩
    return sv


def w_state(n: int) -> np.ndarray:
    """Return the n-qubit W state: equal superposition of all single-excitation
    basis states.

    |W_n⟩ = (|10…0⟩ + |01…0⟩ + … + |00…1⟩) / √n

    Parameters
    ----------
    n : int
        Number of qubits (must be ≥ 2).

    Returns
    -------
    np.ndarray
        Normalised state vector of length 2**n.

    Examples
    --------
    >>> sv = w_state(3)
    >>> abs(np.linalg.norm(sv) - 1.0) < 1e-12
    True
    """
    if n < 2:
        raise ValueError(f"w_state requires n >= 2, got n={n}")
    sv = np.zeros(2**n, dtype=np.complex128)
    for i in range(n):
        sv[2**i] = 1.0 / np.sqrt(n)
    return sv


def cluster_state(n: int) -> np.ndarray:
    """Return the 1D cluster (graph) state on a line of n qubits.

    Constructed by applying Hadamard to all qubits, then CZ to each
    nearest-neighbour pair (i, i+1).

    Parameters
    ----------
    n : int
        Number of qubits (must be ≥ 2).

    Returns
    -------
    np.ndarray
        Normalised state vector of length 2**n.

    Notes
    -----
    The cluster state is the resource state for measurement-based quantum
    computation. For n=2 it is equivalent (up to local unitaries) to the
    Bell state.

    Examples
    --------
    >>> sv = cluster_state(2)
    >>> abs(np.linalg.norm(sv) - 1.0) < 1e-12
    True
    """
    if n < 2:
        raise ValueError(f"cluster_state requires n >= 2, got n={n}")
    # Start in |+⟩^⊗n = H^⊗n |0…0⟩
    sv = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)

    # Apply CZ to each nearest-neighbour pair
    for i in range(n - 1):
        sv = _apply_cz_sv(sv, i, i + 1, n)
    return sv


def dicke_state(n: int, k: int = 2) -> np.ndarray:
    """Return the Dicke state |D^n_k⟩: equal superposition of all n-qubit
    computational basis states with exactly k excitations.

    |D^n_k⟩ = C(n,k)^{-1/2} Σ_{|x|=k} |x⟩

    Parameters
    ----------
    n : int
        Number of qubits (must be ≥ k).
    k : int
        Number of excitations (default 2 for Dicke-k2).

    Returns
    -------
    np.ndarray
        Normalised state vector of length 2**n.

    Examples
    --------
    >>> sv = dicke_state(4, 2)
    >>> abs(np.linalg.norm(sv) - 1.0) < 1e-12
    True
    >>> # C(4,2) = 6 non-zero amplitudes, each 1/sqrt(6)
    >>> (sv != 0).sum()
    6
    """
    if k > n:
        raise ValueError(f"dicke_state requires k <= n, got k={k}, n={n}")
    sv = np.zeros(2**n, dtype=np.complex128)
    norm = 1.0 / np.sqrt(_comb(n, k))
    for bits in combinations(range(n), k):
        idx = sum(2**b for b in bits)
        sv[idx] = norm
    return sv


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def build_target_state(target: str, n: int) -> np.ndarray:
    """Build a target state by name.

    Parameters
    ----------
    target : str
        One of: "Bell", "GHZ", "W", "Cluster", "Dicke-k2", "Dicke-k3".
    n : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        Normalised state vector of length 2**n.

    Raises
    ------
    ValueError
        If the target name is not recognised.

    Examples
    --------
    >>> sv = build_target_state("GHZ", 3)
    >>> abs(sv[0] - 1/np.sqrt(2)) < 1e-12
    True
    """
    _map = {
        "Bell":     lambda: bell_state(n),
        "GHZ":      lambda: ghz_state(n),
        "W":        lambda: w_state(n),
        "Cluster":  lambda: cluster_state(n),
        "Dicke-k2": lambda: dicke_state(n, k=2),
        "Dicke-k3": lambda: dicke_state(n, k=3),
    }
    if target not in _map:
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Valid targets: {sorted(_map.keys())}"
        )
    return _map[target]()


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def compute_fidelity(
    sv: np.ndarray,
    target: Union[np.ndarray, str],
    n: int = None,
) -> float:
    """Compute the state fidelity F = |⟨target|sv⟩|².

    Parameters
    ----------
    sv : np.ndarray
        Current state vector (will be normalised internally).
    target : np.ndarray or str
        Target state vector, or a target name string (requires n).
    n : int, optional
        Number of qubits — required when target is a string.

    Returns
    -------
    float
        Fidelity in [0, 1].

    Examples
    --------
    >>> sv = ghz_state(3)
    >>> compute_fidelity(sv, sv)
    1.0
    >>> compute_fidelity(sv, "GHZ", n=3)
    1.0
    """
    if isinstance(target, str):
        if n is None:
            raise ValueError("n must be provided when target is a string")
        target = build_target_state(target, n)
    sv_norm = sv / (np.linalg.norm(sv) + 1e-300)
    tgt_norm = target / (np.linalg.norm(target) + 1e-300)
    return float(abs(np.dot(sv_norm.conj(), tgt_norm)) ** 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_cz_sv(sv: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    """Apply CZ gate to qubits q0 and q1 of a state vector in-place."""
    sv = sv.copy()
    for idx in range(2**n):
        bit0 = (idx >> q0) & 1
        bit1 = (idx >> q1) & 1
        if bit0 == 1 and bit1 == 1:
            sv[idx] *= -1
    return sv


def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
