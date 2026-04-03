"""
Matrix Product Operator (MPO) - efficient representation of operators
and mixed quantum states.

An MPO represents an N-qubit operator as a chain of rank-4 tensors:

    O = Sum W[0]^{s0,s0'} * W[1]^{s1,s1'} * ... * W[N-1]^{s_{N-1},s_{N-1}'}

where each W[i] has shape (chi_{i-1}, d, d, chi_i) with:
    - d: local physical dimension (2 for qubits)
    - chi_i: bond dimension between sites i and i+1
    - First d index: "ket" (output), second d index: "bra" (input)

For density matrices rho, the MPO represents the mixed state directly,
enabling efficient simulation of noisy quantum systems with bounded
entanglement (as shown by Haah et al. 2024 for noisy circuits).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from ..backend import active_backend
from .mps import MPS


class MPO:
    """Matrix Product Operator for N-qubit operators or density matrices.

    Parameters
    ----------
    tensors : list of ndarray
        List of rank-4 tensors [chi_left, d_out, d_in, chi_right] for each site.
    phys_dim : int
        Local physical dimension (default 2 for qubits).
    """

    def __init__(self, tensors: List[np.ndarray], phys_dim: int = 2):
        self._tensors = list(tensors)
        self._phys_dim = phys_dim
        self._n_sites = len(tensors)
        self._validate()

    def _validate(self):
        for i, t in enumerate(self._tensors):
            if t.ndim != 4:
                raise ValueError(
                    f"Tensor at site {i} has rank {t.ndim}, expected 4 "
                    f"(chi_left, d_out, d_in, chi_right)"
                )
            if t.shape[1] != self._phys_dim or t.shape[2] != self._phys_dim:
                raise ValueError(
                    f"Physical dimensions at site {i} are "
                    f"({t.shape[1]}, {t.shape[2]}), expected "
                    f"({self._phys_dim}, {self._phys_dim})"
                )
        for i in range(self._n_sites - 1):
            if self._tensors[i].shape[3] != self._tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch between sites {i} and {i+1}"
                )

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def phys_dim(self) -> int:
        return self._phys_dim

    @property
    def bond_dims(self) -> List[int]:
        return [self._tensors[i].shape[3] for i in range(self._n_sites - 1)]

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

    def copy(self) -> MPO:
        return MPO([t.copy() for t in self._tensors], self._phys_dim)

    # -- Creation methods --------------------------------------------

    @classmethod
    def identity(cls, n_sites: int, phys_dim: int = 2) -> MPO:
        """Create the identity MPO: I = |0><0| + |1><1| + ..."""
        be = active_backend()
        tensors = []
        for _ in range(n_sites):
            t = be.zeros((1, phys_dim, phys_dim, 1))
            for s in range(phys_dim):
                t[0, s, s, 0] = 1.0
            tensors.append(t)
        return cls(tensors, phys_dim)

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        n_sites: int,
        phys_dim: int = 2,
        max_bond: Optional[int] = None,
        cutoff: float = 1e-12,
    ) -> MPO:
        """Convert a dense 2^N x 2^N matrix to MPO via successive SVDs.

        Parameters
        ----------
        matrix : ndarray
            Dense operator matrix of shape (d^N, d^N).
        n_sites : int
            Number of sites.
        phys_dim : int
            Local physical dimension.
        max_bond : int, optional
            Maximum bond dimension for truncation.
        cutoff : float
            SVD truncation threshold.
        """
        be = active_backend()
        d = phys_dim
        D = d ** n_sites

        if matrix.shape != (D, D):
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match "
                f"({D}, {D}) for {n_sites} sites with d={d}"
            )

        # Reshape to (d, d, d, d, ..., d, d) with 2*n_sites indices
        shape = tuple([d] * (2 * n_sites))
        # Interleave ket and bra indices: s0, s0', s1, s1', ...
        tensor = be.reshape(be.array(matrix), (D, D))

        # Reorder to group (s0, s0', s1, s1', ...)
        row_shape = tuple([d] * n_sites)
        col_shape = tuple([d] * n_sites)
        tensor = be.reshape(tensor, row_shape + col_shape)
        # Current order: s0, s1, ..., s_{N-1}, s0', s1', ..., s_{N-1}'
        # Want: s0, s0', s1, s1', ..., s_{N-1}, s_{N-1}'
        perm = []
        for i in range(n_sites):
            perm.append(i)
            perm.append(n_sites + i)
        tensor = be.transpose(tensor, tuple(perm))

        # Successive SVD from left to right
        tensors = []
        remaining = be.reshape(tensor, (1,) + tuple([d * d] * n_sites))

        for i in range(n_sites - 1):
            chi_left = remaining.shape[0]
            remaining = be.reshape(
                remaining, (chi_left * d * d, -1)
            )
            U, S, Vh = be.svd(remaining, full_matrices=False)

            # Truncation
            s_np = be.to_numpy(S)
            mask = s_np > cutoff
            if max_bond is not None:
                mask[max_bond:] = False
            chi = max(int(np.sum(mask)), 1)

            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]

            # Absorb S into Vh
            site_tensor = be.reshape(U, (chi_left, d, d, chi))
            tensors.append(site_tensor)

            remaining = be.array(np.diag(be.to_numpy(S))) @ Vh

        # Last tensor
        chi_left = remaining.shape[0]
        last_tensor = be.reshape(remaining, (chi_left, d, d, 1))
        tensors.append(last_tensor)

        return cls(tensors, phys_dim)

    @classmethod
    def pure_state_dm(cls, mps: MPS) -> MPO:
        """Create the density matrix MPO rho = |psi><psi| from an MPS."""
        be = active_backend()
        tensors = []
        for i in range(mps.n_sites):
            A = mps[i]  # (chi_l, d, chi_r)
            A_conj = be.conj(A)
            # rho_i = A x A* -> (chi_l*chi_l', d, d, chi_r*chi_r')
            # einsum: 'asc,btc->abst cd' but we want (chi_l*chi_l', d_out, d_in, chi_r*chi_r')
            W = be.einsum("asc,btc->abst", A, A_conj)
            chi_l = A.shape[0]
            chi_r = A.shape[2]
            W = be.reshape(W, (chi_l * chi_l, mps.phys_dim, mps.phys_dim, chi_r * chi_r))
            tensors.append(W)
        # Fix: the einsum above is wrong for the bond contraction.
        # Correct: outer product on bond indices
        tensors_correct = []
        for i in range(mps.n_sites):
            A = mps[i]  # (chi_l, d, chi_r)
            A_conj = be.conj(A)
            # W[a,b,s,t,c,d] = A[a,s,c] * A*[b,t,d]
            W = be.einsum("asc,btd->abstcd", A, A_conj)
            chi_l = A.shape[0]
            d = A.shape[1]
            chi_r = A.shape[2]
            # Reshape to (chi_l*chi_l', d, d, chi_r*chi_r')
            W = be.reshape(W, (chi_l**2, d, d, chi_r**2))
            tensors_correct.append(W)
        return cls(tensors_correct, mps.phys_dim)

    # -- Operations --------------------------------------------------

    def trace(self) -> complex:
        """Compute Tr(O) by contracting bra and ket indices."""
        be = active_backend()
        env = be.ones((1, 1))
        for i in range(self._n_sites):
            W = self._tensors[i]  # (chi_l, d_out, d_in, chi_r)
            # Trace over physical indices: delta_{s,s'}
            traced = be.einsum("assc->ac", W)  # (chi_l, chi_r)
            env = be.einsum("ab,bc->ac", env, traced)
        return complex(be.to_numpy(env)[0, 0])

    def apply_to_mps(self, mps: MPS) -> MPS:
        """Apply this MPO to an MPS: |psi'> = O|psi>.

        Returns a new MPS with potentially larger bond dimension.
        """
        be = active_backend()
        if self._n_sites != mps.n_sites:
            raise ValueError("MPO and MPS must have the same number of sites.")

        new_tensors = []
        for i in range(self._n_sites):
            W = self._tensors[i]   # (chi_W_l, d_out, d_in, chi_W_r)
            A = mps[i]             # (chi_A_l, d, chi_A_r)
            # Contract over d_in = d:
            # B[a,b,s,c,d] = W[a,s,t,c] * A[b,t,d]
            B = be.einsum("astc,btd->abscd", W, A)
            chi_W_l, d_out, _, chi_W_r = W.shape
            chi_A_l, _, chi_A_r = A.shape
            # Fuse bond indices: (chi_W_l*chi_A_l, d_out, chi_W_r*chi_A_r)
            B = be.reshape(B, (chi_W_l * chi_A_l, d_out, chi_W_r * chi_A_r))
            new_tensors.append(B)

        return MPS(new_tensors, self._phys_dim)

    def expectation_mps(self, mps: MPS) -> complex:
        """Compute <psi|O|psi> efficiently."""
        be = active_backend()
        env = be.ones((1, 1, 1))
        for i in range(self._n_sites):
            W = self._tensors[i]     # (chi_W_l, d_out, d_in, chi_W_r)
            A = mps[i]               # (chi_A_l, d, chi_A_r)
            A_conj = be.conj(A)      # (chi_A_l, d, chi_A_r)*
            # env: (chi_A, chi_W, chi_A')
            # Contract: env(a,b,c) * A(a,s,d) * W(b,s,t,e) * A_conj(c,t,f) -> (d,e,f)
            env = be.einsum(
                "abc,asd,bste,ctf->def",
                env, A, W, A_conj
            )
        return complex(be.to_numpy(env)[0, 0, 0])

    def compress(self, max_bond: int, cutoff: float = 1e-12):
        """Compress the MPO bond dimensions via SVD truncation (in-place).

        Performs a left-to-right sweep followed by a right-to-left sweep.
        """
        be = active_backend()
        d = self._phys_dim

        # Left-to-right sweep: QR
        for i in range(self._n_sites - 1):
            W = self._tensors[i]
            chi_l, _, _, chi_r = W.shape
            mat = be.reshape(W, (chi_l * d * d, chi_r))
            Q, R = be.qr(mat)
            new_chi = Q.shape[1]
            self._tensors[i] = be.reshape(Q, (chi_l, d, d, new_chi))
            # Absorb R into next tensor
            W_next = self._tensors[i + 1]
            chi_r_next = W_next.shape[3]
            W_next = be.reshape(W_next, (W_next.shape[0], d * d * chi_r_next))
            W_next = R @ W_next
            self._tensors[i + 1] = be.reshape(W_next, (new_chi, d, d, chi_r_next))

        # Right-to-left sweep: SVD with truncation
        for i in range(self._n_sites - 1, 0, -1):
            W = self._tensors[i]
            chi_l, _, _, chi_r = W.shape
            mat = be.reshape(W, (chi_l, d * d * chi_r))
            U, S, Vh = be.svd(mat, full_matrices=False)

            s_np = be.to_numpy(S)
            mask = s_np > cutoff
            mask[max_bond:] = False
            chi = max(int(np.sum(mask)), 1)

            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]

            self._tensors[i] = be.reshape(Vh, (chi, d, d, chi_r))

            # Absorb U @ diag(S) into previous tensor
            US = U * S[None, :]
            W_prev = self._tensors[i - 1]
            chi_l_prev = W_prev.shape[0]
            W_prev = be.reshape(W_prev, (chi_l_prev * d * d, W_prev.shape[3]))
            W_prev = W_prev @ US
            self._tensors[i - 1] = be.reshape(W_prev, (chi_l_prev, d, d, chi))

    def __repr__(self) -> str:
        bonds = self.bond_dims
        return (
            f"MPO(n_sites={self._n_sites}, phys_dim={self._phys_dim}, "
            f"bond_dims={bonds})"
        )
