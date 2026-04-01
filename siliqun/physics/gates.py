"""
Quantum gate library for silicon spin qubits.

Provides standard single- and two-qubit gates as dense 2×2 and 4×4
matrices, plus silicon-specific pulse-derived gates (ESR, EDSR,
exchange-based CNOT/SWAP).
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from ..backend import active_backend


# ── Pauli matrices ──────────────────────────────────────────────────

PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Raising / lowering
SIGMA_PLUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=np.complex128)


# ── Pauli gate functions ──────────────────────────────────────────

def pauli_x() -> np.ndarray:
    """Pauli X gate."""
    return PAULI_X.copy()

def pauli_y() -> np.ndarray:
    """Pauli Y gate."""
    return PAULI_Y.copy()

def pauli_z() -> np.ndarray:
    """Pauli Z gate."""
    return PAULI_Z.copy()


# ── Standard single-qubit gates ────────────────────────────────────

def hadamard() -> np.ndarray:
    """Hadamard gate H."""
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def phase_gate(phi: float) -> np.ndarray:
    """Phase gate S(φ) = diag(1, e^{iφ})."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    """Rotation about X axis: Rx(θ) = exp(-iθX/2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    """Rotation about Y axis: Ry(θ) = exp(-iθY/2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    """Rotation about Z axis: Rz(θ) = exp(-iθZ/2)."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def t_gate() -> np.ndarray:
    """T gate (π/8 gate)."""
    return phase_gate(np.pi / 4)


# ── Standard two-qubit gates ───────────────────────────────────────

def cnot() -> np.ndarray:
    """CNOT gate (control on qubit 0, target on qubit 1)."""
    return np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=np.complex128,
    )


def cz() -> np.ndarray:
    """Controlled-Z gate."""
    return np.diag([1, 1, 1, -1]).astype(np.complex128)


def swap() -> np.ndarray:
    """SWAP gate."""
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def sqrt_swap() -> np.ndarray:
    """√SWAP gate — native to exchange-coupled spin qubits."""
    return np.array(
        [[1, 0, 0, 0],
         [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
         [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


# ── Silicon spin qubit specific gates ───────────────────────────────

def esr_rotation(theta: float, phi: float) -> np.ndarray:
    """Electron Spin Resonance (ESR) rotation.

    Single-qubit rotation driven by an oscillating magnetic field
    at the Larmor frequency. Parameterized by rotation angle θ
    and phase φ of the microwave drive.

    U = Rz(φ) · Ry(θ) · Rz(-φ)
    """
    return rz(phi) @ ry(theta) @ rz(-phi)


def edsr_rotation(theta: float, phi: float) -> np.ndarray:
    """Electric Dipole Spin Resonance (EDSR) rotation.

    Single-qubit rotation driven by an oscillating electric field
    via spin-orbit coupling or a micromagnet gradient. Same unitary
    as ESR but physically different mechanism (used in SiMOS and GAA).
    """
    return esr_rotation(theta, phi)


def exchange_gate(J: float, t: float) -> np.ndarray:
    """Exchange interaction gate: U = exp(-i J t (σ₁·σ₂)/4).

    Parameters
    ----------
    J : float
        Exchange coupling strength (Hz).
    t : float
        Interaction time (seconds).

    Returns
    -------
    4×4 unitary matrix in the computational basis.
    """
    be = active_backend()
    # Heisenberg exchange: H = J/4 (XX + YY + ZZ)
    H_exchange = (J / 4) * (
        np.kron(PAULI_X, PAULI_X) +
        np.kron(PAULI_Y, PAULI_Y) +
        np.kron(PAULI_Z, PAULI_Z)
    )
    return be.to_numpy(be.expm(-1j * H_exchange * t))


def exchange_sqrt_swap(J: float) -> np.ndarray:
    """√SWAP via exchange interaction with calibrated timing.

    The timing t = π/(2J) gives a √SWAP gate.
    """
    return exchange_gate(J, np.pi / (2 * J))


# ── Gate-to-MPO conversion ─────────────────────────────────────────

def single_qubit_mpo_tensor(gate: np.ndarray) -> np.ndarray:
    """Convert a 2×2 gate to a rank-4 MPO tensor (1, 2, 2, 1)."""
    return gate.reshape(1, 2, 2, 1)


def two_qubit_gate_to_mpo_tensors(
    gate: np.ndarray,
    max_bond: Optional[int] = None,
) -> tuple:
    """Decompose a 4×4 two-qubit gate into two MPO tensors via SVD.

    Parameters
    ----------
    gate : ndarray
        4×4 unitary matrix.
    max_bond : int, optional
        Maximum bond dimension for truncation.

    Returns
    -------
    (W0, W1) : tuple of ndarray
        W0 has shape (1, 2, 2, χ), W1 has shape (χ, 2, 2, 1).
    """
    be = active_backend()
    # Reshape gate to (d_out0, d_out1, d_in0, d_in1)
    G = be.array(gate).reshape(2, 2, 2, 2)
    # Reorder to (d_out0, d_in0, d_out1, d_in1)
    G = be.transpose(G, (0, 2, 1, 3))
    # Reshape to matrix (d_out0*d_in0, d_out1*d_in1)
    mat = be.reshape(G, (4, 4))

    U, S, Vh = be.svd(mat, full_matrices=False)

    chi = 4
    if max_bond is not None:
        chi = min(chi, max_bond)
    U = U[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]

    # Absorb sqrt(S) into both
    sqrt_S = be.array(np.sqrt(be.to_numpy(S)))
    U = U * sqrt_S[None, :]
    Vh = sqrt_S[:, None] * Vh

    W0 = be.reshape(U, (1, 2, 2, chi))
    W1 = be.reshape(Vh, (chi, 2, 2, 1))

    return be.to_numpy(W0), be.to_numpy(W1)
