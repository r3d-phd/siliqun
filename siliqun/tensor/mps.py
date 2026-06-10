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
    def cluster_linear_state(cls, n_qubits: int) -> "MPS":
        """Create the open-boundary 1D Cluster state as a normalised MPS.

        This is the general version that works for any n_qubits >= 2,
        unlike cluster1d_state which requires powers of 2.

        The 1D cluster state is defined by:
            |C_n> = prod_{i=0}^{n-2} CZ(i, i+1) |+>^{otimes n}

        where |+> = (|0> + |1>) / sqrt(2).

        This has an exact MPS representation with bond dimension 2.

        Reference: Raussendorf & Briegel, PRL 86, 5188 (2001).

        Parameters
        ----------
        n_qubits : int
            Number of qubits (any integer >= 2).
        """
        if n_qubits < 2:
            raise ValueError("cluster_linear_state requires n_qubits >= 2")

        inv_sqrt2 = 1.0 / np.sqrt(2.0)

        # The 1D cluster state has an exact MPS with bond dim 2.
        # Tensors derived from the transfer matrix of the cluster state.
        # For open boundary conditions:
        #   A[0] (left boundary):  shape (1, 2, 2)
        #   A[i] (bulk):           shape (2, 2, 2)
        #   A[n-1] (right boundary): shape (2, 2, 1)
        #
        # The MPS tensors for the cluster state are:
        #   A[0]_{s0, alpha} = <s0| H |alpha>  (H = Hadamard)
        # with the CZ entanglement encoded in the bond structure.
        #
        # Exact construction via sequential application of H and CZ:

        tensors = []

        if n_qubits == 2:
            # |C2> = (1/2)(|00> + |01> + |10> - |11>)
            A0 = np.zeros((1, 2, 2), dtype=complex)
            A0[0, 0, 0] = inv_sqrt2
            A0[0, 1, 1] = inv_sqrt2
            B0 = np.zeros((2, 2, 1), dtype=complex)
            B0[0, 0, 0] = 1.0
            B0[0, 1, 0] = 1.0
            B0[1, 0, 0] = 1.0
            B0[1, 1, 0] = -1.0
            mps = cls([A0, B0])
            mps.normalize()
            return mps

        # General n: build |+>^n then apply CZ gates sequentially
        # Start from |+>^n as product MPS
        tensors = []
        for i in range(n_qubits):
            chi_l = 1 if i == 0 else 2
            chi_r = 1 if i == n_qubits - 1 else 2
            t = np.zeros((chi_l, 2, chi_r), dtype=complex)
            if i == 0:
                t[0, 0, 0] = inv_sqrt2
                t[0, 1, 0] = inv_sqrt2
            elif i == n_qubits - 1:
                t[0, 0, 0] = inv_sqrt2
                t[0, 1, 0] = inv_sqrt2
                t[1, 0, 0] = inv_sqrt2
                t[1, 1, 0] = inv_sqrt2
            else:
                t[0, 0, 0] = inv_sqrt2
                t[0, 1, 0] = inv_sqrt2
                t[1, 0, 1] = inv_sqrt2
                t[1, 1, 1] = inv_sqrt2
            tensors.append(t)

        # Apply CZ(i, i+1) for i = 0..n-2 via two-site SVD
        def _apply_cz(tensors, site):
            A = tensors[site]
            B = tensors[site + 1]
            chi_l = A.shape[0]
            chi_r = B.shape[2]
            theta = np.einsum("asm,mtr->astr", A, B)
            theta[:, 1, 1, :] *= -1  # CZ: flip sign when s0=1, s1=1
            theta_mat = theta.reshape(chi_l * 2, 2 * chi_r)
            U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
            chi_new = min(len(S), 2)
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]
            S_sqrt = np.sqrt(np.maximum(S, 0.0))
            tensors[site] = (U * S_sqrt).reshape(chi_l, 2, chi_new)
            tensors[site + 1] = (Vh.T * S_sqrt).T.reshape(chi_new, 2, chi_r)

        for i in range(n_qubits - 1):
            _apply_cz(tensors, i)

        mps = cls(tensors)
        mps.normalize()
        return mps

    @classmethod
    def cluster1d_state(cls, n_qubits: int) -> "MPS":
        """Create the open-boundary 1D Cluster state as a normalised MPS.

        Uses the strict 2^n **module-fusion** architecture:

          Base (2q):  H(0) H(1) CZ(0,1)  -- exact 2-tensor MPS, bond_dim=2
          4q = fuse(2q_L, 2q_R) + boundary CZ on (qubit 1, qubit 2)
          8q = fuse(4q_L, 4q_R) + boundary CZ on (qubit 3, qubit 4)
          2^k q = fuse(2^(k-1)_L, 2^(k-1)_R) + boundary CZ

        This matches the 2^n propagation architecture of QUASAR ScaleRL v3.

        n_qubits must be a power of 2 (2, 4, 8, 16, ...).
        For arbitrary n, use cluster_linear_state() instead.

        Reference: Verstraete & Cirac, PRA 70, 060302(R) (2004).
        """
        import numpy as np

        if n_qubits < 2:
            raise ValueError("cluster1d_state requires n_qubits >= 2")
        if (n_qubits & (n_qubits - 1)) != 0:
            raise ValueError(
                "cluster1d_state (2^n architecture) requires n_qubits to be a "
                "power of 2, got {}. Use 2, 4, 8, 16, ... "
                "For arbitrary n, use cluster_linear_state() instead.".format(n_qubits)
            )

        # -- Helper: apply CZ gate on sites (i, i+1) of an MPS tensor list --
        def _apply_boundary_cz(tensors, site):
            """Apply CZ(site, site+1) to the MPS tensor list (in-place)."""
            A = tensors[site]
            B = tensors[site + 1]
            chi_l = A.shape[0]
            chi_r = B.shape[2]
            theta = np.einsum("asm,mtr->astr", A, B)
            theta[:, 1, 1, :] *= -1
            theta_mat = theta.reshape(chi_l * 2, 2 * chi_r)
            U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
            chi_new = min(len(S), 2)
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]
            S_sqrt = np.sqrt(np.maximum(S, 0.0))
            tensors[site] = (U * S_sqrt).reshape(chi_l, 2, chi_new)
            tensors[site + 1] = (Vh.T * S_sqrt).T.reshape(chi_new, 2, chi_r)

        # -- Base case: exact 2q Cluster state --
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        A0 = np.zeros((1, 2, 2), dtype=complex)
        A0[0, 0, 0] = inv_sqrt2
        A0[0, 1, 1] = inv_sqrt2
        B0 = np.zeros((2, 2, 1), dtype=complex)
        B0[0, 0, 0] = 1.0
        B0[0, 1, 0] = 1.0
        B0[1, 0, 0] = 1.0
        B0[1, 1, 0] = -1.0
        base_tensors = [A0, B0]

        if n_qubits == 2:
            mps = cls(base_tensors)
            mps.normalize()
            return mps

        # -- Recursive fusion: 2^(k-1) -> 2^k --
        def _fuse(tensors_l, tensors_r):
            n_l = len(tensors_l)
            fused = list(tensors_l) + list(tensors_r)
            _apply_boundary_cz(fused, n_l - 1)
            return fused

        current_tensors = list(base_tensors)
        current_n = 2
        while current_n < n_qubits:
            current_tensors = _fuse(current_tensors, list(current_tensors))
            current_n *= 2

        mps = cls(current_tensors)
        mps.normalize()
        return mps

    @classmethod
    def dicke_state(cls, n_qubits: int, k: int) -> "MPS":
        """Create the Dicke state |D^n_k> as a normalised MPS.

        The Dicke state |D^n_k> is the equal-weight superposition of all
        n-qubit computational basis states with exactly k excitations (ones):

            |D^n_k> = C(n,k)^{-1/2} * sum_{|x|=k} |x>

        where the sum is over all bit strings x of length n with Hamming
        weight k.

        Special cases:
            k=0: |00...0> (computational zero)
            k=1: W state (|D^n_1>)
            k=n: |11...1> (computational one)

        The MPS is constructed using the exact bond-dimension-(k+1) representation
        based on the combinatorial structure of the Dicke state.

        Reference: Bergmann & van Loock, PRA 94, 012311 (2016).

        Parameters
        ----------
        n_qubits : int
            Number of qubits (n >= 1).
        k : int
            Excitation number (0 <= k <= n_qubits).
        """
        if n_qubits < 1:
            raise ValueError("dicke_state requires n_qubits >= 1")
        if not (0 <= k <= n_qubits):
            raise ValueError(
                f"dicke_state requires 0 <= k <= n_qubits, got k={k}, n={n_qubits}"
            )

        from math import comb, sqrt

        # Bond dimension is k+1 (tracks number of excitations placed so far)
        chi = k + 1

        tensors = []
        for site in range(n_qubits):
            n_remaining = n_qubits - site  # qubits remaining including this one
            chi_l = min(site + 1, chi)     # bond dim on left
            chi_r = min(site + 2, chi) if site < n_qubits - 1 else 1

            t = np.zeros((chi_l, 2, chi_r), dtype=complex)

            for j in range(chi_l):
                # j = number of excitations placed in sites 0..site-1
                # Physical index 0 (|0>): j stays the same
                if j < chi_r:
                    # Amplitude: sqrt(C(n_remaining-1, k-j) / C(n_remaining, k-j))
                    # = sqrt((n_remaining - (k-j)) / n_remaining)
                    excitations_needed = k - j
                    if 0 <= excitations_needed <= n_remaining - 1:
                        num = comb(n_remaining - 1, excitations_needed)
                        den = comb(n_remaining, excitations_needed)
                        amp = sqrt(num / den) if den > 0 else 0.0
                        t[j, 0, j] = amp

                # Physical index 1 (|1>): j increases to j+1
                j_new = j + 1
                if j_new < chi_r or (site == n_qubits - 1 and j_new == k):
                    excitations_needed = k - j - 1
                    if 0 <= excitations_needed <= n_remaining - 1:
                        num = comb(n_remaining - 1, excitations_needed)
                        den = comb(n_remaining, excitations_needed + 1)
                        amp = sqrt(num / den) if den > 0 else 0.0
                        if site < n_qubits - 1 and j_new < chi_r:
                            t[j, 1, j_new] = amp
                        elif site == n_qubits - 1 and j_new == k:
                            t[j, 1, 0] = amp

            tensors.append(t)

        # Fall back to dense construction for correctness on edge cases
        # (the combinatorial MPS above handles most cases; dense is the safety net)
        try:
            mps = cls(tensors)
            # Verify norm is close to 1 (sanity check)
            n_val = mps.norm()
            if abs(n_val - 1.0) > 0.1:
                raise ValueError(f"Dicke MPS norm={n_val:.4f}, falling back to dense")
            return mps
        except Exception:
            # Dense fallback: enumerate all basis states with Hamming weight k
            from itertools import combinations
            dim = 2 ** n_qubits
            sv = np.zeros(dim, dtype=complex)
            norm_factor = 1.0 / np.sqrt(comb(n_qubits, k)) if comb(n_qubits, k) > 0 else 1.0
            for positions in combinations(range(n_qubits), k):
                idx = sum(1 << (n_qubits - 1 - p) for p in positions)
                sv[idx] = norm_factor
            return cls.from_dense(sv, n_qubits)

    @classmethod
    def from_dense(cls, sv: np.ndarray, n_qubits: int, max_bond_dim: int = 64) -> "MPS":
        """Convert a dense state vector to MPS via sequential SVD.

        Parameters
        ----------
        sv : ndarray, shape (2^n_qubits,)
            Dense state vector.
        n_qubits : int
            Number of qubits.
        max_bond_dim : int
            Maximum bond dimension (truncation).
        """
        sv = np.array(sv, dtype=complex)
        if sv.shape[0] != 2 ** n_qubits:
            raise ValueError(
                f"State vector length {sv.shape[0]} != 2^{n_qubits}={2**n_qubits}"
            )

        tensors = []
        psi = sv.reshape(1, -1)  # (1, 2^n)

        for i in range(n_qubits - 1):
            chi_l = psi.shape[0]
            psi = psi.reshape(chi_l * 2, -1)  # (chi_l * 2, 2^(n-i-1))
            U, S, Vh = np.linalg.svd(psi, full_matrices=False)
            chi_new = min(len(S), max_bond_dim)
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]
            tensors.append(U.reshape(chi_l, 2, chi_new))
            psi = np.diag(S) @ Vh  # (chi_new, 2^(n-i-1))

        # Last tensor
        chi_l = psi.shape[0]
        tensors.append(psi.reshape(chi_l, 2, 1))

        mps = cls(tensors)
        mps.normalize()
        return mps

    @classmethod
    def random_state(cls, n_qubits: int, bond_dim: int = 4) -> MPS:
        """Create a random normalised MPS (alias for random())."""
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
        env = be.ones((1, 1))
        for i in range(self._n_sites):
            A = self._tensors[i]
            A_conj = be.conj(A)
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
        result = self._tensors[0]
        for i in range(1, self._n_sites):
            result = be.tensordot(result, self._tensors[i], ([result.ndim - 1], [0]))
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
                env = be.einsum("ab,asc,st,bte->ce", env, A, op, A_conj)
            else:
                env = be.einsum("ab,adc,bde->ce", env, A, A_conj)
        return complex(be.to_numpy(env)[0, 0])

    def apply_one_site_gate(self, gate: np.ndarray, site: int) -> None:
        """Apply a single-qubit gate to site in-place (TEBD building block).

        Parameters
        ----------
        gate : ndarray, shape (2, 2)
            Unitary gate matrix.
        site : int
            Target qubit index.
        """
        A = self._tensors[site]  # (chi_l, 2, chi_r)
        # Contract gate into physical index: A'[chi_l, s', chi_r] = sum_s gate[s',s] A[chi_l,s,chi_r]
        self._tensors[site] = np.einsum("sp,lpq->lsq", gate, A)

    def apply_two_site_gate(
        self,
        gate: np.ndarray,
        site: int,
        max_bond_dim: Optional[int] = None,
        svd_cutoff: float = 1e-12,
    ) -> None:
        """Apply a two-qubit gate to sites (site, site+1) via SVD (TEBD step).

        This is the core operation for Time-Evolving Block Decimation (TEBD).
        The gate is applied to the two-site tensor, then decomposed back into
        two MPS tensors via truncated SVD.

        Parameters
        ----------
        gate : ndarray, shape (4, 4) or (2, 2, 2, 2)
            Two-qubit unitary gate. If shape (4,4), it is treated as a matrix
            in the |s0 s1> basis. If shape (2,2,2,2), indices are [s0', s1', s0, s1].
        site : int
            Left site index (gate acts on sites site and site+1).
        max_bond_dim : int or None
            Maximum bond dimension after truncation. None means no truncation.
        svd_cutoff : float
            Singular values below this threshold are discarded.
        """
        if site < 0 or site >= self._n_sites - 1:
            raise ValueError(
                f"site={site} out of range for two-site gate on {self._n_sites}-site MPS"
            )

        gate = np.array(gate, dtype=complex)
        if gate.shape == (4, 4):
            gate = gate.reshape(2, 2, 2, 2)  # [s0', s1', s0, s1]
        elif gate.shape != (2, 2, 2, 2):
            raise ValueError(f"gate must have shape (4,4) or (2,2,2,2), got {gate.shape}")

        A = self._tensors[site]        # (chi_l, 2, chi_m)
        B = self._tensors[site + 1]    # (chi_m, 2, chi_r)
        chi_l = A.shape[0]
        chi_r = B.shape[2]

        # Two-site tensor: theta[chi_l, s0, s1, chi_r]
        theta = np.einsum("asm,mtr->astr", A, B)

        # Apply gate: theta'[chi_l, s0', s1', chi_r] = sum_{s0,s1} gate[s0',s1',s0,s1] * theta[chi_l,s0,s1,chi_r]
        # theta shape: (chi_l, s0, s1, chi_r) — indices: i=chi_l, c=s0, s=s1, j=chi_r
        # gate shape: (s0', s1', s0, s1) — indices: a=s0', b=s1', c=s0, s=s1
        # result[i,a,b,j] = sum_{c,s} gate[a,b,c,s] * theta[i,c,s,j]
        theta_new = np.einsum("abcs,icsj->iabj", gate, theta)
        # theta_new shape: (chi_l, 2, 2, chi_r) -> (chi_l*2, 2*chi_r)
        theta_mat = theta_new.reshape(chi_l * 2, 2 * chi_r)

        # SVD with truncation
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

        # Truncate by cutoff
        mask = S > svd_cutoff
        if max_bond_dim is not None:
            mask[max_bond_dim:] = False
        chi_new = max(1, int(np.sum(mask)))

        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb sqrt(S) into both tensors (symmetric gauge)
        S_sqrt = np.sqrt(S)
        self._tensors[site] = (U * S_sqrt).reshape(chi_l, 2, chi_new)
        self._tensors[site + 1] = (Vh.T * S_sqrt).T.reshape(chi_new, 2, chi_r)

    def tebd_sweep(
        self,
        gates: List[Tuple[int, np.ndarray]],
        max_bond_dim: Optional[int] = None,
        svd_cutoff: float = 1e-12,
    ) -> None:
        """Apply a sequence of two-site gates (one TEBD sweep).

        Parameters
        ----------
        gates : list of (site, gate) tuples
            Each gate is applied to (site, site+1).
        max_bond_dim : int or None
            Maximum bond dimension after each gate application.
        svd_cutoff : float
            SVD truncation cutoff.
        """
        for site, gate in gates:
            self.apply_two_site_gate(gate, site, max_bond_dim, svd_cutoff)

    def __repr__(self) -> str:
        bonds = self.bond_dims
        return (
            f"MPS(n_sites={self._n_sites}, phys_dim={self._phys_dim}, "
            f"bond_dims={bonds})"
        )
