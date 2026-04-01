"""
Abstract compute backend interface for SiliQun.

All tensor operations are dispatched through this interface, enabling
transparent switching between NumPy (CPU), JAX (GPU), and cuQuantum
backends without changing simulation code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import numpy as np


class Backend(ABC):
    """Abstract compute backend providing array operations for tensor networks."""

    name: str = "abstract"

    # ── Array creation ──────────────────────────────────────────────

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        """Create a zero-filled tensor."""

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        """Create a ones-filled tensor."""

    @abstractmethod
    def eye(self, n: int, dtype=None) -> np.ndarray:
        """Create an n×n identity matrix."""

    @abstractmethod
    def array(self, data, dtype=None) -> np.ndarray:
        """Convert data to a backend tensor."""

    @abstractmethod
    def complex_dtype(self):
        """Return the default complex dtype for this backend."""

    @abstractmethod
    def real_dtype(self):
        """Return the default real dtype for this backend."""

    # ── Linear algebra ──────────────────────────────────────────────

    @abstractmethod
    def svd(
        self,
        tensor: np.ndarray,
        full_matrices: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the SVD: A = U @ diag(S) @ Vh."""

    @abstractmethod
    def qr(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the QR decomposition."""

    @abstractmethod
    def eigh(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition of a Hermitian matrix: (eigenvalues, eigenvectors)."""

    @abstractmethod
    def expm(self, matrix: np.ndarray) -> np.ndarray:
        """Matrix exponential."""

    @abstractmethod
    def norm(self, tensor: np.ndarray) -> float:
        """Frobenius norm of a tensor."""

    # ── Tensor operations ───────────────────────────────────────────

    @abstractmethod
    def tensordot(
        self,
        a: np.ndarray,
        b: np.ndarray,
        axes: Union[int, Tuple],
    ) -> np.ndarray:
        """Tensor contraction along specified axes."""

    @abstractmethod
    def einsum(self, subscripts: str, *operands) -> np.ndarray:
        """Einstein summation convention."""

    @abstractmethod
    def reshape(self, tensor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape a tensor."""

    @abstractmethod
    def transpose(self, tensor: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
        """Transpose (permute) tensor axes."""

    @abstractmethod
    def conj(self, tensor: np.ndarray) -> np.ndarray:
        """Complex conjugate."""

    @abstractmethod
    def trace(self, matrix: np.ndarray) -> complex:
        """Matrix trace."""

    @abstractmethod
    def kron(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Kronecker product."""

    # ── Utility ─────────────────────────────────────────────────────

    @abstractmethod
    def to_numpy(self, tensor) -> np.ndarray:
        """Convert backend tensor to NumPy array."""

    @abstractmethod
    def random_normal(
        self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0
    ) -> np.ndarray:
        """Generate random normal tensor."""

    @abstractmethod
    def random_uniform(
        self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0
    ) -> np.ndarray:
        """Generate random uniform tensor."""

    def __repr__(self) -> str:
        return f"Backend({self.name})"
