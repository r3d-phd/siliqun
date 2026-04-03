"""
Matrix Product State (MPS) - efficient representation of 1D quantum states.

An MPS represents an N-qubit pure state as a chain of rank-3 tensors:

    |psi> = Sum A[0]^{s0} * A[1]^{s1} * ... * A[N-1]^{s_{N-1}} |s0 s1 ... s_{N-1}>

where each A[i] has shape (chi_{i-1}, d, chi_i) with:
    - d: local physical dimension (2 for qubits)
    - chi_i: bond dimension between sites i and i+1
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence
import numpy as np
from ..backend import active_backend


class MPS:
    """Matrix Product State for N-qubit systems.

    Parameters
    ----------
    tensors : list of ndarray
        List of rank-3 tensors [chi_left, d, chi_right] for each site.
        Boundary tensors have chi_left=1 or chi_right=1.
    phys_dim : int
        Local physical dimension (default 2 for qubits).
    """

    def __init__(self, tensors: List[np.ndarray], phys_dim: int = 2):
        self._tensors = list(tensors)
        self._phys_dim = phys_dim
        self._n_sites = len(tensors)
        self._validate()

    def _validate(self):
        """Check tensor dimensions are consistent."""
        for i, t in enumerate(self._tensors):
            if t.ndim != 3:
                raise ValueError(
                    f"Tensor at site {i} has rank {t.ndim}, expected 3 "
                    f"(chi_left, d, chi_right)"
                )
            if t.shape[1] != self._phys_dim:
                raise ValueError(
                    f"Physical dimension at site {i} is {t.shape[1]}, "
                    f"expected {self._phys_dim}"
                )
        # Check bond dimension compatibility
        for i in range(self._n_sites - 1):
            if self._tensors[i].shape[2] != self._tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch between sites {i} and {i+1}: "
                    f"{self._tensors[i].shape[2]} vs {self._tensors[i+1].shape[0]}"
                )

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def phys_dim(self) -> int:
        return self._phys_dim

    @property
    def bond_dims(self) -> List[int]:
        """Return the bond dimensions chi_1, ..., chi_{N-1}."""
        return [self._tensors[i].shape[2] for i in range(self._n_sites - 1)]

    @property
    def max_bond_dim(self) -> int:
        dims = self.bond_dims
        return max(dims) if dims else 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self._tensors[i]

    def __setitem__(self, i: int, tensor: np.ndarray):
        self._tensors[i] = tensor

    def __len__(self) -> int:
        return self._n_sites

    def copy(self) -> MPS:
        """Deep copy of the MPS."""
        return MPS([t.copy() for t in self._tensors], self._phys_dim)

    # -- Creation methods --------------------------------------------

    @classmethod
    def computational_basis(
        cls, n_qubits: int, state: int = 0, phys_dim: int = 2
    ) -> MPS:
        """Create an MPS in a computational basis state.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        state : int
            Integer representation of the basis state (e.g., 0 for |000...0>).
        """
        be = active_backend()
        tensors = []
        for i in range(n_qubits):
            bit = (state >> (n_qubits - 1 - i)) & 1
            t = be.zeros((1, phys_dim, 1))
            t[0, bit, 0] = 1.0
            tensors.append(t)
        return cls(tensors, phys_dim)

    @classmethod
    def ghz_state(cls, n_qubits: int) -> MPS:
        """Create a GHZ state: (|00...0> + |11...1>) / sqrt2."""
        be = active_backend()
        tensors = []
        # First site
        t0 = be.zeros((1, 2, 2))
        t0[0, 0, 0] = 1.0 / np.sqrt(2)
        t0[0, 1, 1] = 1.0 / np.sqrt(2)
        tensors.append(t0)
        # Middle sites
        for _ in range(n_qubits - 2):
            t = be.zeros((2, 2, 2))
            t[0, 0, 0] = 1.0
            t[1, 1, 1] = 1.0
            tensors.append(t)
        # Last site
        tN = be.zeros((2, 2, 1))
        tN[0, 0, 0] = 1.0
        tN[1, 1, 0] = 1.0
        tensors.append(tN)
        return cls(tensors)

    @classmethod
    def bell_state(cls, n_qubits: int = 2) -> MPS:
        """Create a Bell-like state: (|00...0> + |11...1>) / sqrt2.

        For n_qubits=2, this is the standard Bell state |Phi+>.
        For n_qubits>2, this is the GHZ state.
        """
        return cls.ghz_state(n_qubits)

    @classmethod
    def w_state(cls, n_qubits: int) -> MPS:
        """Create a W state: (|100...0> + |010...0> + ... + |000...1>) / sqrtn.

        The W state has exactly one excitation shared equally among all qubits.
        """
        be = active_backend()
        # Build MPS for W state using bond dimension 2
        # Auxiliary space: [0] = no excitation yet, [1] = excitation placed
        tensors = []
        n = n_qubits
        coeff = 1.0 / np.sqrt(n)

        # First site: (1, 2, 2)
        t0 = be.zeros((1, 2, 2))
        t0[0, 0, 0] = 1.0       # pass |0>, no excitation yet
        t0[0, 1, 1] = coeff     # place excitation here
        tensors.append(t0)

        # Middle sites: (2, 2, 2)
        for i in range(1, n - 1):
            t = be.zeros((2, 2, 2))
            t[0, 0, 0] = 1.0    # no excitation yet, pass |0>
            t[0, 1, 1] = coeff  # place excitation here
            t[1, 0, 1] = 1.0    # excitation already placed, pass |0>
            tensors.append(t)

        # Last site: (2, 2, 1)
        tN = be.zeros((2, 2, 1))
        tN[0, 1, 0] = coeff     # place excitation here (last chance)
        tN[1, 0, 0] = 1.0      # excitation already placed, pass |0>
        tensors.append(tN)

        return cls(tensors)

    @classmethod
    def random_state(cls, n_qubits: int, bond_dim: int = 4) -> MPS:
        """Alias for random() for compatibility."""
        return cls.random(n_qubits, bond_dim)

    @classmethod
    def random(
        cls, n_qubits: int, bond_dim: int = 4, phys_dim: int = 2
    ) -> MPS:
        """Create a random MPS with specified bond dimension."""
        be = active_backend()
        tensors = []
        for i in range(n_qubits):
            chi_l = 1 if i == 0 else min(bond_dim, phys_dim**i)
            chi_r = 1 if i == n_qubits - 1 else min(bond_dim, phys_dim**(i + 1))
            chi_l = min(chi_l, bond_dim)
            chi_r = min(chi_r, bond_dim)
            real = be.random_normal((chi_l, phys_dim, chi_r))
            imag = be.random_normal((chi_l, phys_dim, chi_r))
            t = be.array(real + 1j * imag)
            tensors.append(t)
        mps = cls(tensors, phys_dim)
        mps.normalize()
        return mps

    # -- Operations --------------------------------------------------

    def norm(self) -> float:
        """Compute <psi|psi> via left-to-right contraction."""
        be = active_backend()
        # Start with identity transfer matrix
        env = be.ones((1, 1))
        for i in range(self._n_sites):
            A = self._tensors[i]       # (chi_l, d, chi_r)
            A_conj = be.conj(A)        # (chi_l, d, chi_r)*
            # Contract: env(chi_l, chi_l') x A(chi_l, d, chi_r) x A*(chi_l', d, chi_r')
            env = be.einsum("ab,adc,bde->ce", env, A, A_conj)
        return float(np.sqrt(abs(be.to_numpy(env)[0, 0])))

    def normalize(self):
        """Normalize the MPS in-place."""
        n = self.norm()
        if n > 1e-15:
            be = active_backend()
            self._tensors[0] = be.array(self._tensors[0] / n)

    def inner(self, other: MPS) -> complex:
        """Compute <self|other>."""
        be = active_backend()
        if self._n_sites != other._n_sites:
            raise ValueError("MPS must have the same number of sites.")
        env = be.ones((1, 1))
        for i in range(self._n_sites):
            A = other[i]
            B_conj = be.conj(self[i])
            env = be.einsum("ab,adc,bde->ce", env, A, B_conj)
        return complex(be.to_numpy(env)[0, 0])

    def to_dense(self) -> np.ndarray:
        """Convert MPS to full state vector (exponential memory!)."""
        be = active_backend()
        result = self._tensors[0]  # (1, d, chi)
        for i in range(1, self._n_sites):
            # result: (..., chi) x tensor_i: (chi, d, chi')
            result = be.tensordot(result, self._tensors[i], ([result.ndim - 1], [0]))
        # Squeeze boundary dimensions
        shape = tuple(self._phys_dim for _ in range(self._n_sites))
        result = be.reshape(result, shape)
        return be.to_numpy(result)

    def expectation_local(self, op: np.ndarray, site: int) -> complex:
        """Compute <psi|O_site|psi> for a single-site operator O."""
        be = active_backend()
        op = be.array(op)
        env = be.ones((1, 1))
        for i in range(self._n_sites):
            A = self._tensors[i]
            A_conj = be.conj(A)
            if i == site:
                # Insert operator: O_{s,s'} between A and A*
                env = be.einsum("ab,asc,st,bte->ce", env, A, op, A_conj)
            else:
                env = be.einsum("ab,adc,bde->ce", env, A, A_conj)
        return complex(be.to_numpy(env)[0, 0])

    def __repr__(self) -> str:
        bonds = self.bond_dims
        return (
            f"MPS(n_sites={self._n_sites}, phys_dim={self._phys_dim}, "
            f"bond_dims={bonds})"
        )
