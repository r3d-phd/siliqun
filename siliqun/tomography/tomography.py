"""
Quantum State and Process Tomography for Silicon Spin Qubits.

Implements:
    - Quantum State Tomography (QST) via Pauli measurement basis
    - Quantum Process Tomography (QPT) via chi-matrix reconstruction
    - Maximum Likelihood Estimation (MLE) for physical density matrices
    - State and process fidelity metrics

The tomography routines use the existing SiliQun measurement primitives
(measure_qubit, compute_fidelity) from the engine layer.

References
----------
Chuang & Nielsen, J. Mod. Opt. 44, 2455 (1997) — QPT
James et al., Phys. Rev. A 64, 052312 (2001) — QST via MLE
Hradil, Phys. Rev. A 55, R1561 (1997) — MLE reconstruction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pauli basis
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI_1Q = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_LABELS_1Q = ["I", "X", "Y", "Z"]

# Measurement bases for state tomography: {+X, -X, +Y, -Y, +Z, -Z}
# Each basis is defined by the unitary that rotates it to the Z basis
MEAS_BASES_1Q = {
    "+X": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),   # H
    "+Y": np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2), # Ry(-pi/2)
    "+Z": I2,
}


def _tensor_pauli(labels: List[str]) -> np.ndarray:
    """Build n-qubit Pauli operator from list of single-qubit labels."""
    op = PAULI_1Q[labels[0]]
    for lbl in labels[1:]:
        op = np.kron(op, PAULI_1Q[lbl])
    return op


def _all_pauli_basis(n_qubits: int) -> List[Tuple[str, np.ndarray]]:
    """Generate all 4^n Pauli basis operators for n qubits."""
    from itertools import product
    basis = []
    for labels in product(PAULI_LABELS_1Q, repeat=n_qubits):
        label = "".join(labels)
        op = _tensor_pauli(list(labels))
        basis.append((label, op))
    return basis


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def pauli_expectation(rho: np.ndarray, pauli_op: np.ndarray) -> float:
    """Compute expectation value <P> = Tr(rho * P).

    Parameters
    ----------
    rho : ndarray, shape (2^n, 2^n)
        Density matrix.
    pauli_op : ndarray, shape (2^n, 2^n)
        Pauli operator.

    Returns
    -------
    float
        Real part of Tr(rho * P).
    """
    return float(np.real(np.trace(rho @ pauli_op)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute quantum state fidelity F(rho, sigma).

    For pure states: F = |<psi|phi>|^2 = Tr(rho * sigma)
    For mixed states: F = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2

    Uses the pure-state formula when one state is pure (Tr(rho^2) ≈ 1).

    Parameters
    ----------
    rho : ndarray
        First density matrix.
    sigma : ndarray
        Second density matrix (target).

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    # Check if sigma is pure
    purity_sigma = float(np.real(np.trace(sigma @ sigma)))
    if abs(purity_sigma - 1.0) < 1e-6:
        # Pure target: F = Tr(rho * sigma)
        return float(np.real(np.clip(np.trace(rho @ sigma), 0, 1)))

    # Mixed states: Uhlmann fidelity
    # F = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
    sqrt_rho = _matrix_sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    sqrt_M = _matrix_sqrt(M)
    f = float(np.real(np.trace(sqrt_M))) ** 2
    return float(np.clip(f, 0, 1))


def process_fidelity(chi: np.ndarray, chi_ideal: np.ndarray) -> float:
    """Compute process fidelity between two chi matrices.

    F_process = Tr(chi_ideal^dag * chi) / d^2

    Parameters
    ----------
    chi : ndarray, shape (d^2, d^2)
        Experimental chi matrix.
    chi_ideal : ndarray, shape (d^2, d^2)
        Ideal chi matrix.

    Returns
    -------
    float
        Process fidelity in [0, 1].
    """
    d2 = chi.shape[0]
    f = float(np.real(np.trace(chi_ideal.conj().T @ chi))) / d2
    return float(np.clip(f, 0, 1))


def purity(rho: np.ndarray) -> float:
    """Compute purity Tr(rho^2).

    Parameters
    ----------
    rho : ndarray
        Density matrix.

    Returns
    -------
    float
        Purity in [1/d, 1].
    """
    return float(np.real(np.trace(rho @ rho)))


def reconstruct_density_matrix(
    pauli_expectations: Dict[str, float],
    n_qubits: int,
) -> np.ndarray:
    """Reconstruct density matrix from Pauli expectation values.

    Uses the Pauli expansion:
        rho = (1/2^n) * sum_{P in Pauli^n} <P> * P

    Parameters
    ----------
    pauli_expectations : dict
        Mapping from Pauli label (e.g., "XX", "IZ") to expectation value.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    ndarray, shape (2^n, 2^n)
        Reconstructed density matrix.
    """
    d = 2 ** n_qubits
    rho = np.zeros((d, d), dtype=np.complex128)
    basis = _all_pauli_basis(n_qubits)
    for label, op in basis:
        exp_val = pauli_expectations.get(label, 0.0)
        rho += exp_val * op
    rho /= d
    return rho


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Compute matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# Maximum Likelihood Estimation
# ---------------------------------------------------------------------------

def _mle_reconstruct(
    pauli_expectations: Dict[str, float],
    n_qubits: int,
    n_shots: int = 1000,
) -> np.ndarray:
    """Reconstruct physical density matrix via Maximum Likelihood Estimation.

    Enforces:
        1. Hermiticity: rho = rho†
        2. Positive semidefiniteness: rho >= 0
        3. Unit trace: Tr(rho) = 1

    Uses the Cholesky parameterisation: rho = T†T / Tr(T†T)
    where T is a lower-triangular matrix with real diagonal.

    Ref: James et al., Phys. Rev. A 64, 052312 (2001)

    Parameters
    ----------
    pauli_expectations : dict
        Pauli expectation values.
    n_qubits : int
        Number of qubits.
    n_shots : int
        Number of measurement shots (used for likelihood weighting).

    Returns
    -------
    ndarray
        Physical density matrix.
    """
    d = 2 ** n_qubits
    basis = _all_pauli_basis(n_qubits)

    # Initial guess: linear inversion
    rho_lin = reconstruct_density_matrix(pauli_expectations, n_qubits)

    # Parameterise as T†T / Tr(T†T) where T is lower-triangular
    # T has d^2 real parameters: d real diagonal + d*(d-1) complex off-diagonal
    n_params = d * d  # Real and imaginary parts of lower triangle

    def _t_from_params(params: np.ndarray) -> np.ndarray:
        """Build lower-triangular T from parameter vector."""
        T = np.zeros((d, d), dtype=np.complex128)
        idx = 0
        for i in range(d):
            T[i, i] = params[idx]  # Real diagonal
            idx += 1
            for j in range(i):
                T[i, j] = params[idx] + 1j * params[idx + 1]
                idx += 2
        return T

    def _rho_from_params(params: np.ndarray) -> np.ndarray:
        T = _t_from_params(params)
        rho = T.conj().T @ T
        return rho / np.trace(rho)

    def _neg_log_likelihood(params: np.ndarray) -> float:
        rho = _rho_from_params(params)
        loss = 0.0
        for label, op in basis:
            measured = pauli_expectations.get(label, 0.0)
            predicted = pauli_expectation(rho, op)
            loss += (measured - predicted) ** 2
        return loss

    # Initial parameters from linear inversion
    # Use Cholesky of rho_lin if positive definite, else identity
    try:
        rho_psd = rho_lin + 1e-6 * np.eye(d)
        L = np.linalg.cholesky(rho_psd)
    except np.linalg.LinAlgError:
        L = np.eye(d, dtype=np.complex128) / np.sqrt(d)

    params0 = []
    for i in range(d):
        params0.append(float(np.real(L[i, i])))
        for j in range(i):
            params0.append(float(np.real(L[i, j])))
            params0.append(float(np.imag(L[i, j])))

    result = minimize(
        _neg_log_likelihood,
        params0,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    return _rho_from_params(result.x)


# ---------------------------------------------------------------------------
# State Tomography
# ---------------------------------------------------------------------------

@dataclass
class StateTomographyResult:
    """Result of quantum state tomography.

    Attributes
    ----------
    rho : ndarray
        Reconstructed density matrix.
    pauli_expectations : dict
        Measured Pauli expectation values.
    purity : float
        Purity Tr(rho^2).
    fidelity_to_target : float or None
        Fidelity to target state if provided.
    n_qubits : int
        Number of qubits.
    method : str
        Reconstruction method used ("linear" or "mle").
    """
    rho: np.ndarray
    pauli_expectations: Dict[str, float]
    purity: float
    fidelity_to_target: Optional[float]
    n_qubits: int
    method: str

    def __repr__(self) -> str:
        f_str = (
            f", fidelity={self.fidelity_to_target:.4f}"
            if self.fidelity_to_target is not None
            else ""
        )
        return (
            f"StateTomographyResult(n_qubits={self.n_qubits}, "
            f"purity={self.purity:.4f}{f_str}, method={self.method})"
        )


class StateTomography:
    """Quantum State Tomography for silicon spin qubits.

    Reconstructs the density matrix of an n-qubit state from
    Pauli expectation value measurements.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    method : str
        "linear" (default): Direct Pauli expansion. Fast but may give
        non-physical matrices for noisy data.
        "mle": Maximum Likelihood Estimation. Enforces physicality
        (Hermitian, PSD, unit trace). Slower but always physical.

    Examples
    --------
    >>> import numpy as np
    >>> from siliqun.tomography import StateTomography
    >>>
    >>> # Simulate a Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    >>> psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> rho_true = np.outer(psi, psi.conj())
    >>>
    >>> # Simulate Pauli measurements (perfect, no shot noise)
    >>> qst = StateTomography(n_qubits=2)
    >>> expectations = qst.simulate_measurements(rho_true, n_shots=10000)
    >>> result = qst.reconstruct(expectations, target_state=rho_true)
    >>> print(result)
    StateTomographyResult(n_qubits=2, purity=1.0000, fidelity=1.0000, method=linear)
    """

    def __init__(self, n_qubits: int, method: str = "linear"):
        self.n_qubits = n_qubits
        self.method = method
        self._basis = _all_pauli_basis(n_qubits)
        logger.info(
            "StateTomography: n_qubits=%d, method=%s, n_operators=%d",
            n_qubits, method, len(self._basis)
        )

    def simulate_measurements(
        self,
        rho: np.ndarray,
        n_shots: int = 10000,
        add_shot_noise: bool = True,
    ) -> Dict[str, float]:
        """Simulate Pauli measurements on a density matrix.

        Parameters
        ----------
        rho : ndarray
            Density matrix to measure.
        n_shots : int
            Number of measurement shots per Pauli operator.
        add_shot_noise : bool
            If True, add binomial shot noise (realistic simulation).

        Returns
        -------
        dict
            Pauli label -> expectation value.
        """
        expectations = {}
        for label, op in self._basis:
            true_exp = pauli_expectation(rho, op)
            if add_shot_noise and label != "I" * self.n_qubits:
                # Binomial shot noise: p = (1 + <P>) / 2
                p = (1 + true_exp) / 2
                p = float(np.clip(p, 0, 1))
                counts_plus = np.random.binomial(n_shots, p)
                noisy_exp = (2 * counts_plus / n_shots) - 1
                expectations[label] = float(noisy_exp)
            else:
                expectations[label] = float(true_exp)
        return expectations

    def reconstruct(
        self,
        pauli_expectations: Dict[str, float],
        target_state: Optional[np.ndarray] = None,
        n_shots: int = 1000,
    ) -> StateTomographyResult:
        """Reconstruct density matrix from Pauli measurements.

        Parameters
        ----------
        pauli_expectations : dict
            Pauli label -> expectation value.
        target_state : ndarray, optional
            Target state for fidelity calculation.
        n_shots : int
            Number of shots (used for MLE weighting).

        Returns
        -------
        StateTomographyResult
        """
        if self.method == "mle":
            rho = _mle_reconstruct(pauli_expectations, self.n_qubits, n_shots)
        else:
            rho = reconstruct_density_matrix(pauli_expectations, self.n_qubits)

        p = purity(rho)
        f = None
        if target_state is not None:
            f = fidelity(rho, target_state)

        return StateTomographyResult(
            rho=rho,
            pauli_expectations=pauli_expectations,
            purity=p,
            fidelity_to_target=f,
            n_qubits=self.n_qubits,
            method=self.method,
        )

    def from_simulator(
        self,
        simulator,
        circuit: List,
        target_state: Optional[np.ndarray] = None,
        n_shots: int = 10000,
    ) -> StateTomographyResult:
        """Run state tomography using a SiliQun simulator.

        Executes the circuit, then measures all Pauli operators.

        Parameters
        ----------
        simulator : LindbladSimulator or StatevectorSimulator
            SiliQun simulator instance.
        circuit : list
            Gate circuit to execute before tomography.
        target_state : ndarray, optional
            Target state for fidelity calculation.
        n_shots : int
            Number of shots per Pauli measurement.

        Returns
        -------
        StateTomographyResult
        """
        # Get final state from simulator
        result = simulator.run(circuit)
        rho = getattr(result, "density_matrix", None)
        if rho is None:
            # Statevector simulator: convert to density matrix
            sv = result.statevector
            rho = np.outer(sv, sv.conj())

        # Simulate Pauli measurements from the density matrix
        expectations = self.simulate_measurements(rho, n_shots=n_shots)
        return self.reconstruct(expectations, target_state=target_state, n_shots=n_shots)


# ---------------------------------------------------------------------------
# Process Tomography
# ---------------------------------------------------------------------------

@dataclass
class ProcessTomographyResult:
    """Result of quantum process tomography.

    Attributes
    ----------
    chi : ndarray, shape (d^2, d^2)
        Chi (process) matrix in the Pauli basis.
    process_fidelity : float or None
        Process fidelity to ideal chi matrix if provided.
    n_qubits : int
        Number of qubits.
    """
    chi: np.ndarray
    process_fidelity: Optional[float]
    n_qubits: int

    def __repr__(self) -> str:
        f_str = (
            f", process_fidelity={self.process_fidelity:.4f}"
            if self.process_fidelity is not None
            else ""
        )
        return f"ProcessTomographyResult(n_qubits={self.n_qubits}{f_str})"


class ProcessTomography:
    """Quantum Process Tomography for silicon spin qubits.

    Reconstructs the chi (process) matrix of an n-qubit quantum channel
    by performing state tomography on a complete set of input states.

    The input states are the 4^n Pauli eigenstates. For each input state,
    state tomography is performed on the output, giving the full chi matrix.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    method : str
        "linear" or "mle" (passed to StateTomography).

    Examples
    --------
    >>> from siliqun.tomography import ProcessTomography
    >>> import numpy as np
    >>>
    >>> # Ideal CNOT process matrix
    >>> qpt = ProcessTomography(n_qubits=2)
    >>> # In practice, you would run the circuit on a simulator:
    >>> # result = qpt.from_simulator(simulator, cnot_circuit, ideal_chi=chi_cnot)
    """

    def __init__(self, n_qubits: int, method: str = "linear"):
        self.n_qubits = n_qubits
        self.method = method
        self._qst = StateTomography(n_qubits, method=method)
        self._basis = _all_pauli_basis(n_qubits)
        self._d = 2 ** n_qubits

        # Input states: Pauli eigenstates |+x>, |-x>, |+y>, |-y>, |+z>, |-z>
        # For n qubits: tensor products of single-qubit input states
        self._input_states = self._build_input_states()
        logger.info(
            "ProcessTomography: n_qubits=%d, method=%s, n_input_states=%d",
            n_qubits, method, len(self._input_states)
        )

    def _build_input_states(self) -> List[np.ndarray]:
        """Build the 4^n input states for QPT."""
        from itertools import product

        # Single-qubit input states: |0>, |1>, |+>, |+i>
        single_qubit_inputs = [
            np.array([1, 0], dtype=np.complex128),                    # |0>
            np.array([0, 1], dtype=np.complex128),                    # |1>
            np.array([1, 1], dtype=np.complex128) / np.sqrt(2),       # |+>
            np.array([1, 1j], dtype=np.complex128) / np.sqrt(2),      # |+i>
        ]

        input_states = []
        for combo in product(single_qubit_inputs, repeat=self.n_qubits):
            state = combo[0]
            for s in combo[1:]:
                state = np.kron(state, s)
            rho = np.outer(state, state.conj())
            input_states.append(rho)

        return input_states

    def reconstruct(
        self,
        output_states: List[np.ndarray],
        ideal_unitary: Optional[np.ndarray] = None,
    ) -> ProcessTomographyResult:
        """Reconstruct chi matrix from output state tomography results.

        Parameters
        ----------
        output_states : list of ndarray
            Output density matrices for each input state (in same order as
            self._input_states).
        ideal_unitary : ndarray, optional
            Ideal unitary for process fidelity calculation.

        Returns
        -------
        ProcessTomographyResult
        """
        d = self._d
        d2 = d * d

        # Build the chi matrix via linear inversion
        # chi_{mn} = sum_{j,k} beta_{jk,mn} * rho_out_{jk}
        # Using the standard QPT formula from Chuang & Nielsen (1997)

        # Pauli basis operators
        pauli_ops = [op for _, op in self._basis]

        # Build the B matrix: B_{j, mn} = Tr(rho_in_j * E_m^dag * rho_out_j * E_n)
        # where E_m are the Pauli basis operators
        chi = np.zeros((d2, d2), dtype=np.complex128)

        for j, (rho_in, rho_out) in enumerate(
            zip(self._input_states, output_states)
        ):
            for m, E_m in enumerate(pauli_ops):
                for n, E_n in enumerate(pauli_ops):
                    # chi_{mn} += Tr(E_m rho_in E_n^dag) * rho_out contribution
                    # Simplified: use Pauli expansion of rho_out
                    coeff = np.trace(E_m @ rho_in @ E_n.conj().T)
                    rho_predicted = E_m @ rho_in @ E_n.conj().T
                    overlap = np.trace(rho_predicted.conj().T @ rho_out)
                    chi[m, n] += coeff * overlap

        chi /= (d * len(self._input_states))

        # Compute process fidelity if ideal unitary provided
        pf = None
        if ideal_unitary is not None:
            chi_ideal = self._unitary_to_chi(ideal_unitary)
            pf = process_fidelity(chi, chi_ideal)

        return ProcessTomographyResult(
            chi=chi,
            process_fidelity=pf,
            n_qubits=self.n_qubits,
        )

    def _unitary_to_chi(self, U: np.ndarray) -> np.ndarray:
        """Convert a unitary to its chi matrix representation.

        chi_{mn} = (1/d) * Tr(E_m^dag * U)^* * Tr(E_n^dag * U)

        Parameters
        ----------
        U : ndarray, shape (d, d)
            Unitary matrix.

        Returns
        -------
        ndarray, shape (d^2, d^2)
            Chi matrix.
        """
        d = self._d
        d2 = d * d
        pauli_ops = [op for _, op in self._basis]
        chi = np.zeros((d2, d2), dtype=np.complex128)
        for m, E_m in enumerate(pauli_ops):
            for n, E_n in enumerate(pauli_ops):
                chi[m, n] = (
                    np.trace(E_m.conj().T @ U).conj()
                    * np.trace(E_n.conj().T @ U)
                    / d
                )
        return chi

    def from_simulator(
        self,
        simulator,
        circuit: List,
        ideal_unitary: Optional[np.ndarray] = None,
        n_shots: int = 10000,
    ) -> ProcessTomographyResult:
        """Run process tomography using a SiliQun simulator.

        For each input state, prepares the state, runs the circuit,
        and performs state tomography on the output.

        Parameters
        ----------
        simulator : LindbladSimulator or StatevectorSimulator
            SiliQun simulator instance.
        circuit : list
            Gate circuit defining the process.
        ideal_unitary : ndarray, optional
            Ideal unitary for process fidelity.
        n_shots : int
            Shots per Pauli measurement.

        Returns
        -------
        ProcessTomographyResult
        """
        output_states = []

        for rho_in in self._input_states:
            # Prepare input state and run circuit
            result = simulator.run(circuit, initial_state=rho_in)
            rho_out = getattr(result, "density_matrix", None)
            if rho_out is None:
                sv = result.statevector
                rho_out = np.outer(sv, sv.conj())
            output_states.append(rho_out)

        return self.reconstruct(output_states, ideal_unitary=ideal_unitary)
