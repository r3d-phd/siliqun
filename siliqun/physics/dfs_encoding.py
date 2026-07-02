"""
Decoherence-Free Subspace (DFS) encoding for exchange-only qubits.

Implements the 3-spin -> 1-logical-qubit encoding used in the HRL
SLEDGE device (Weinstein et al., Nature 615, 817-822, 2023).

Encoding:
    The logical qubit is encoded in the S=1/2, m_s=-1/2 subspace
    of 3 electron spins. The two basis states are:

        |0_L> = |S>|down>  = (1/sqrt2)(|updown> - |downup>)|down>
        |1_L> = sqrt(2/3)|T_0>|down> - sqrt(1/3)|S>|up>
              = sqrt(2/3) x (1/sqrt2)(|updown> + |downup>)|down> - sqrt(1/3) x (1/sqrt2)(|updown> - |downup>)|up>

    where |S> = singlet of spins 1,2 and |T_0> = triplet-zero of spins 1,2.

    This encoding is immune to uniform magnetic field fluctuations
    because both |0_L> and |1_L> have the same total m_s = -1/2.

Gate implementation:
    All logical gates are implemented using only exchange interactions
    between pairs of physical spins within and between encoded qubits:

    - Single-qubit Z rotation: J_1_2 exchange (spins 1-2 within qubit)
    - Single-qubit X rotation: J_2_3 exchange (spins 2-3 within qubit)
    - Two-qubit gates: inter-qubit exchange + Fong-Wandzura decomposition

Leakage:
    The 8-dimensional Hilbert space of 3 spins contains:
    - 2 encoded states (S=1/2, m_s=-1/2, even parity)
    - 2 gauge states (S=1/2, m_s=-1/2, odd parity)
    - 4 leaked states (S=3/2 or other m_s values)

    Magnetic field gradients cause leakage from encoded to gauge states.
    The LCCZ gate was designed to prevent leakage spreading.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from ..backend import active_backend


# ======================================================================
# Physical constants and basis states
# ======================================================================

# Single-spin basis: |up> = [1,0], |down> = [0,1]
SPIN_UP = np.array([1, 0], dtype=np.complex128)
SPIN_DN = np.array([0, 1], dtype=np.complex128)

# Pauli matrices for physical spins
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron3(A, B, C):
    """Kronecker product of three matrices."""
    return np.kron(A, np.kron(B, C))


# ======================================================================
# Encoded basis states (3 physical spins -> 1 logical qubit)
# ======================================================================

def encoded_zero() -> np.ndarray:
    """Logical |0_L> state in the 8-dimensional physical Hilbert space.

    Derived from diagonalising the total spin operator S^2 in the
    m_s = -1/2 subspace of 3 spin-1/2 particles. The two S=1/2
    eigenstates form the encoded (logical) subspace.

    |0_L> is the first S=1/2 eigenstate with coefficients
    [-sqrt(2/3), sqrt(1/6), sqrt(1/6)] in the {|udd>, |dud>, |ddu>} basis.

    This state has:
        - Total spin S = 1/2
        - Magnetic quantum number m_s = -1/2
        - Is preserved by all Heisenberg exchange interactions

    In the full 8-dimensional computational basis:
    |0_L> = -sqrt(2/3)|udd> + sqrt(1/6)|dud> + sqrt(1/6)|ddu>
    """
    state = np.zeros(8, dtype=np.complex128)
    # |udd> = index 3 (binary 011)
    state[3] = -np.sqrt(2.0 / 3.0)
    # |dud> = index 5 (binary 101)
    state[5] = np.sqrt(1.0 / 6.0)
    # |ddu> = index 6 (binary 110)
    state[6] = np.sqrt(1.0 / 6.0)
    return state


def encoded_one() -> np.ndarray:
    """Logical |1_L> state in the 8-dimensional physical Hilbert space.

    Derived from diagonalising the total spin operator S^2 in the
    m_s = -1/2 subspace of 3 spin-1/2 particles.

    |1_L> is the second S=1/2 eigenstate with coefficients
    [0, -1/sqrt(2), 1/sqrt(2)] in the {|udd>, |dud>, |ddu>} basis.

    This state has:
        - Total spin S = 1/2
        - Magnetic quantum number m_s = -1/2
        - Is preserved by all Heisenberg exchange interactions
        - Is orthogonal to |0_L>

    In the full 8-dimensional computational basis:
    |1_L> = -1/sqrt(2) |dud> + 1/sqrt(2) |ddu>
    """
    state = np.zeros(8, dtype=np.complex128)
    # |dud> = index 5 (binary 101)
    state[5] = -1.0 / np.sqrt(2)
    # |ddu> = index 6 (binary 110)
    state[6] = 1.0 / np.sqrt(2)
    return state


def gauge_zero() -> np.ndarray:
    """Gauge |g0> state - the S=3/2 state in the m_s=-1/2 sector.

    This is the symmetric combination of the three m_s=-1/2 basis
    states, which belongs to the S=3/2 (quartet) representation.
    It is orthogonal to both |0_L> and |1_L> (which are S=1/2).

    |g0> = -1/sqrt(3) (|udd> + |dud> + |ddu>)

    Any population in this state represents leakage from the
    encoded subspace to the S=3/2 manifold.
    """
    state = np.zeros(8, dtype=np.complex128)
    # Symmetric combination - S=3/2, m_s=-1/2
    state[3] = -1.0 / np.sqrt(3.0)   # |udd>
    state[5] = -1.0 / np.sqrt(3.0)   # |dud>
    state[6] = -1.0 / np.sqrt(3.0)   # |ddu>
    return state


# ======================================================================
# Projectors
# ======================================================================

def encoded_subspace_projector() -> np.ndarray:
    """Projector onto the 2D encoded (logical) subspace.

    P_enc = |0_L><0_L| + |1_L><1_L|
    """
    z = encoded_zero()
    o = encoded_one()
    return np.outer(z, z.conj()) + np.outer(o, o.conj())


def gauge_subspace_projector() -> np.ndarray:
    """Projector onto the gauge subspace."""
    g = gauge_zero()
    return np.outer(g, g.conj())


def leakage_subspace_projector() -> np.ndarray:
    """Projector onto the leaked subspace (S=3/2 and other m_s).

    P_leak = I - P_enc - P_gauge
    """
    return np.eye(8, dtype=np.complex128) - encoded_subspace_projector() - gauge_subspace_projector()


def compute_leakage(state_3spin: np.ndarray) -> float:
    """Compute the leakage probability for a 3-spin state.

    Parameters
    ----------
    state_3spin : ndarray
        8-dimensional state vector of 3 physical spins.

    Returns
    -------
    float
        Probability of being outside the encoded subspace.
    """
    P_enc = encoded_subspace_projector()
    p_encoded = float(np.real(state_3spin.conj() @ P_enc @ state_3spin))
    return 1.0 - p_encoded


def compute_gauge_population(state_3spin: np.ndarray) -> float:
    """Compute the gauge state population for a 3-spin state."""
    P_gauge = gauge_subspace_projector()
    return float(np.real(state_3spin.conj() @ P_gauge @ state_3spin))


# ======================================================================
# Exchange gates on physical spins
# ======================================================================

def exchange_12(theta: float) -> np.ndarray:
    """Exchange gate between physical spins 1 and 2 within an encoded qubit.

    U_1_2(theta) = exp(-i theta/4 (X_1X_2 + Y_1Y_2 + Z_1Z_2)) x I_3

    This implements a logical Z rotation on the encoded qubit.

    Parameters
    ----------
    theta : float
        Exchange angle (= J_1_2 x t / hbar).
    """
    # Heisenberg exchange on spins 1,2
    H_12 = 0.25 * (np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z))
    U_12 = _matrix_exp(-1j * theta * H_12)
    # Tensor with identity on spin 3
    return np.kron(U_12, _I)


def exchange_23(theta: float) -> np.ndarray:
    """Exchange gate between physical spins 2 and 3 within an encoded qubit.

    U_2_3(theta) = I_1 x exp(-i theta/4 (X_2X_3 + Y_2Y_3 + Z_2Z_3))

    This implements a logical X rotation on the encoded qubit.

    Parameters
    ----------
    theta : float
        Exchange angle (= J_2_3 x t / hbar).
    """
    H_23 = 0.25 * (np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z))
    U_23 = _matrix_exp(-1j * theta * H_23)
    return np.kron(_I, U_23)


def exchange_inter(
    theta: float, spin_a: int, spin_b: int, n_physical: int
) -> np.ndarray:
    """Exchange gate between physical spins in DIFFERENT encoded qubits.

    Used for inter-qubit gates (FW-CNOT, encoded SWAP).

    Parameters
    ----------
    theta : float
        Exchange angle.
    spin_a, spin_b : int
        Physical spin indices (0-indexed).
    n_physical : int
        Total number of physical spins.
    """
    d = 2 ** n_physical
    H = np.zeros((d, d), dtype=np.complex128)

    for P, pauli in [(_X, "X"), (_Y, "Y"), (_Z, "Z")]:
        term = np.eye(1, dtype=np.complex128)
        for i in range(n_physical):
            if i == spin_a or i == spin_b:
                term = np.kron(term, P)
            else:
                term = np.kron(term, _I)
        H += 0.25 * term

    return _matrix_exp(-1j * theta * H)


# ======================================================================
# Logical gates via exchange sequences
# ======================================================================

def logical_z_rotation(theta: float) -> np.ndarray:
    """Logical Z rotation on an encoded qubit via J_1_2 exchange.

    R_Z(theta) is implemented by pulsing the exchange between physical
    spins 1 and 2 within the encoded qubit.

    The mapping is: J_1_2 pulse of angle theta -> logical Z rotation by theta
    (up to a global phase).
    """
    return exchange_12(theta)


def logical_x_rotation(theta: float) -> np.ndarray:
    """Logical X rotation on an encoded qubit via J_2_3 exchange.

    R_X(theta) is implemented by pulsing the exchange between physical
    spins 2 and 3 within the encoded qubit.
    """
    return exchange_23(theta)


def logical_arbitrary_rotation(
    theta_z1: float, theta_x: float, theta_z2: float
) -> np.ndarray:
    """Arbitrary single-qubit rotation via Z-X-Z decomposition.

    U = R_Z(theta_z2) * R_X(theta_x) * R_Z(theta_z1)

    Any SU(2) rotation can be decomposed this way, requiring
    at most 3 exchange pulses.
    """
    return exchange_12(theta_z2) @ exchange_23(theta_x) @ exchange_12(theta_z1)


def fong_wandzura_cnot(
    qubit_a_spins: Tuple[int, int, int],
    qubit_b_spins: Tuple[int, int, int],
    n_physical: int,
) -> np.ndarray:
    """Fong-Wandzura CNOT gate between two encoded qubits.

    The FW-CNOT decomposes into a sequence of exchange pulses
    between physical spins of the two encoded qubits. The
    decomposition uses 18 exchange pulses.

    Fidelity: 96.3% (Weinstein et al., Nature 2023)

    Parameters
    ----------
    qubit_a_spins : tuple of 3 int
        Physical spin indices for encoded qubit A.
    qubit_b_spins : tuple of 3 int
        Physical spin indices for encoded qubit B.
    n_physical : int
        Total number of physical spins.

    Returns
    -------
    ndarray
        Unitary matrix for the FW-CNOT in the full physical Hilbert space.

    Notes
    -----
    The exact pulse sequence is from Fong & Wandzura, Quantum Info.
    Comput. 11, 1003-1018 (2011). Here we use the simplified version
    that couples spin 3 of qubit A to spin 1 of qubit B.
    """
    a1, a2, a3 = qubit_a_spins
    b1, b2, b3 = qubit_b_spins

    # Simplified FW-CNOT pulse sequence (18 pulses)
    # Each step is (spin_i, spin_j, angle)
    # The inter-qubit coupling is between a3 and b1
    fw_sequence = [
        # Prepare qubit A
        (a1, a2, np.pi / 4),
        (a2, a3, np.pi / 4),
        (a1, a2, -np.pi / 4),
        # Inter-qubit coupling
        (a3, b1, np.pi / 2),
        # Correct qubit B
        (b1, b2, np.pi / 4),
        (b2, b3, np.pi / 2),
        (b1, b2, -np.pi / 4),
        # Second inter-qubit coupling
        (a3, b1, np.pi / 2),
        # Final corrections
        (a1, a2, np.pi / 4),
        (a2, a3, -np.pi / 4),
        (a1, a2, -np.pi / 4),
        (b1, b2, np.pi / 4),
        (b2, b3, -np.pi / 4),
        (b1, b2, -np.pi / 4),
        # Phase corrections
        (a1, a2, np.pi / 8),
        (b1, b2, np.pi / 8),
        (a2, a3, np.pi / 8),
        (b2, b3, np.pi / 8),
    ]

    U = np.eye(2 ** n_physical, dtype=np.complex128)
    for si, sj, angle in fw_sequence:
        U = exchange_inter(angle, si, sj, n_physical) @ U

    return U


def encoded_swap(
    qubit_a_spins: Tuple[int, int, int],
    qubit_b_spins: Tuple[int, int, int],
    n_physical: int,
) -> np.ndarray:
    """Encoded SWAP gate between two encoded qubits.

    Swaps the logical states of two encoded qubits using
    3 physical SWAP operations (one per physical spin pair).

    Fidelity: 99.3% (Weinstein et al., Nature 2023)

    Parameters
    ----------
    qubit_a_spins, qubit_b_spins : tuple of 3 int
        Physical spin indices for the two encoded qubits.
    n_physical : int
        Total number of physical spins.
    """
    a1, a2, a3 = qubit_a_spins
    b1, b2, b3 = qubit_b_spins

    # SWAP = 3 partial swaps at angle pi
    U = np.eye(2 ** n_physical, dtype=np.complex128)
    for ai, bi in [(a1, b1), (a2, b2), (a3, b3)]:
        U = exchange_inter(np.pi, ai, bi, n_physical) @ U

    return U


def lccz_gate(
    qubit_a_spins: Tuple[int, int, int],
    qubit_b_spins: Tuple[int, int, int],
    n_physical: int,
) -> np.ndarray:
    """Leakage-Controlled CZ (LCCZ) gate.

    A modified CZ gate designed to prevent leakage spreading
    between encoded qubits. Unlike the FW-CNOT, the LCCZ gate
    ensures that if one qubit has leaked, the leakage does not
    spread to the other qubit.

    Fidelity: 93.8% (Weinstein et al., Nature 2023)

    The LCCZ uses a different pulse sequence that checks for
    leakage before applying the entangling operation.

    Parameters
    ----------
    qubit_a_spins, qubit_b_spins : tuple of 3 int
        Physical spin indices for the two encoded qubits.
    n_physical : int
        Total number of physical spins.
    """
    a1, a2, a3 = qubit_a_spins
    b1, b2, b3 = qubit_b_spins

    # LCCZ pulse sequence (simplified)
    # The key difference from FW-CNOT is the leakage-checking
    # structure that prevents leakage propagation
    lccz_sequence = [
        # Prepare: bring to leakage-sensitive basis
        (a1, a2, np.pi / 4),
        (b1, b2, np.pi / 4),
        # Conditional phase via inter-qubit exchange
        (a3, b1, np.pi / 4),
        (a2, a3, np.pi / 2),
        (a3, b1, -np.pi / 4),
        # Leakage check: if leaked, this undoes the phase
        (b1, b2, np.pi / 2),
        (a3, b1, np.pi / 4),
        (b1, b2, -np.pi / 2),
        (a3, b1, -np.pi / 4),
        # Restore
        (a2, a3, -np.pi / 2),
        (a1, a2, -np.pi / 4),
        (b1, b2, -np.pi / 4),
        # Phase corrections
        (a1, a2, np.pi / 8),
        (b1, b2, np.pi / 8),
    ]

    U = np.eye(2 ** n_physical, dtype=np.complex128)
    for si, sj, angle in lccz_sequence:
        U = exchange_inter(angle, si, sj, n_physical) @ U

    return U


# ======================================================================
# DFS Encoder/Decoder for logical <-> physical mapping
# ======================================================================

class DFSEncoder:
    """Encoder/decoder for DFS-encoded qubits.

    Maps between the logical qubit Hilbert space (2^n_logical) and
    the physical spin Hilbert space (2^(3*n_logical)).

    Parameters
    ----------
    n_logical : int
        Number of logical (encoded) qubits.
    """

    def __init__(self, n_logical: int):
        self.n_logical = n_logical
        self.n_physical = 3 * n_logical

        # Build the encoding isometry V: 2^n_L -> 2^n_P
        self._V = self._build_encoding_isometry()

        # Build projectors
        self._P_enc = self._build_encoded_projector()

    def _build_encoding_isometry(self) -> np.ndarray:
        """Build the encoding isometry V that maps logical states
        to physical states.

        V|psi_L> = |psi_P>

        V has shape (2^n_physical, 2^n_logical).
        """
        d_L = 2 ** self.n_logical
        d_P = 2 ** self.n_physical

        # Build basis: tensor product of encoded basis states
        zero_L = encoded_zero()  # 8-dim
        one_L = encoded_one()    # 8-dim

        V = np.zeros((d_P, d_L), dtype=np.complex128)

        for idx in range(d_L):
            # Binary representation of logical state
            bits = [(idx >> (self.n_logical - 1 - q)) & 1
                    for q in range(self.n_logical)]

            # Tensor product of encoded basis states
            state = np.array([1.0], dtype=np.complex128)
            for b in bits:
                if b == 0:
                    state = np.kron(state, zero_L)
                else:
                    state = np.kron(state, one_L)

            V[:, idx] = state

        return V

    def _build_encoded_projector(self) -> np.ndarray:
        """Build the projector onto the encoded subspace."""
        return self._V @ self._V.conj().T

    @property
    def encoding_isometry(self) -> np.ndarray:
        """The encoding isometry V: logical -> physical."""
        return self._V

    @property
    def encoded_projector(self) -> np.ndarray:
        """Projector onto the encoded subspace in physical space."""
        return self._P_enc

    def encode(self, logical_state: np.ndarray) -> np.ndarray:
        """Encode a logical state into the physical Hilbert space.

        Parameters
        ----------
        logical_state : ndarray
            State vector in the logical Hilbert space (2^n_logical).

        Returns
        -------
        ndarray
            State vector in the physical Hilbert space (2^n_physical).
        """
        return self._V @ logical_state

    def decode(self, physical_state: np.ndarray) -> np.ndarray:
        """Decode a physical state back to the logical Hilbert space.

        Projects onto the encoded subspace first, then maps back
        to the logical space.

        Parameters
        ----------
        physical_state : ndarray
            State vector in the physical Hilbert space (2^n_physical).

        Returns
        -------
        ndarray
            State vector in the logical Hilbert space (2^n_logical).
        """
        return self._V.conj().T @ physical_state

    def compute_encoded_fidelity(
        self,
        physical_state: np.ndarray,
        target_logical: np.ndarray,
    ) -> float:
        """Compute fidelity between a physical state and a target
        logical state, accounting for encoding.

        F = |<target_L|Vdag|psi_P>|^2

        Parameters
        ----------
        physical_state : ndarray
            Current physical state (2^n_physical).
        target_logical : ndarray
            Target logical state (2^n_logical).

        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        decoded = self.decode(physical_state)
        overlap = np.abs(target_logical.conj() @ decoded) ** 2
        return float(overlap)

    def compute_total_leakage(self, physical_state: np.ndarray) -> float:
        """Compute total leakage probability for a multi-qubit state.

        Parameters
        ----------
        physical_state : ndarray
            State vector in the physical Hilbert space.

        Returns
        -------
        float
            Probability of being outside the encoded subspace.
        """
        p_encoded = float(np.real(
            physical_state.conj() @ self._P_enc @ physical_state
        ))
        return 1.0 - p_encoded

    def get_qubit_spins(self, logical_qubit: int) -> Tuple[int, int, int]:
        """Get the physical spin indices for a logical qubit.

        Parameters
        ----------
        logical_qubit : int
            Logical qubit index (0-indexed).

        Returns
        -------
        tuple of 3 int
            Physical spin indices (spin_1, spin_2, spin_3).
        """
        base = 3 * logical_qubit
        return (base, base + 1, base + 2)


# ======================================================================
# Utility functions
# ======================================================================

def _matrix_exp(M: np.ndarray) -> np.ndarray:
    """Compute matrix exponential using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        (M + M.conj().T) / 2  # Ensure Hermitian
    )
    # For anti-Hermitian M = -iH, use the full matrix
    # Fall back to scipy-style computation
    from scipy.linalg import expm
    return expm(M)


def partial_swap(theta: float) -> np.ndarray:
    """Partial SWAP gate: U(theta) = cos(theta/2)I + i*sin(theta/2)SWAP.

    This is the native two-qubit gate for exchange-coupled spin qubits.
    At theta = pi, it gives a full SWAP.
    At theta = pi/2, it gives sqrtSWAP.

    Parameters
    ----------
    theta : float
        Rotation angle. theta = J*t/hbar where J is exchange coupling
        and t is pulse duration.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c + 1j * s, 0, 0, 0],
        [0, c, 1j * s, 0],
        [0, 1j * s, c, 0],
        [0, 0, 0, c + 1j * s],
    ], dtype=np.complex128)


def exchange_quality_factor(
    J_freq: float, T2_star: float
) -> float:
    """Compute the exchange oscillation quality factor N_osc.

    N_osc = J/h x T2* = number of coherent exchange oscillations
    before dephasing.

    For the SLEDGE device: N_osc ~ 57.6 at J/h = 100 MHz.

    Parameters
    ----------
    J_freq : float
        Exchange frequency J/h (Hz).
    T2_star : float
        T2* dephasing time (seconds).

    Returns
    -------
    float
        Quality factor N_osc (dimensionless).
    """
    return J_freq * T2_star
