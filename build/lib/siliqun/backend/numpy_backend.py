"""
NumPy compute backend - the reference implementation.

All operations use NumPy and SciPy, providing a CPU-based baseline
that works on any platform without GPU dependencies.
"""

from typing import Tuple, Union
import numpy as np
from scipy.linalg import expm as scipy_expm

from .base import Backend


class NumPyBackend(Backend):
    """NumPy/SciPy reference backend for CPU computation."""

    name = "numpy"

    def complex_dtype(self):
        return np.complex128

    def real_dtype(self):
        return np.float64

    # -- Array creation ----------------------------------------------

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or self.complex_dtype())

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or self.complex_dtype())

    def eye(self, n, dtype=None):
        return np.eye(n, dtype=dtype or self.complex_dtype())

    def array(self, data, dtype=None):
        return np.asarray(data, dtype=dtype or self.complex_dtype())

    # -- Linear algebra ----------------------------------------------

    def svd(self, tensor, full_matrices=False):
        return np.linalg.svd(tensor, full_matrices=full_matrices)

    def qr(self, tensor):
        return np.linalg.qr(tensor)

    def eigh(self, matrix):
        return np.linalg.eigh(matrix)

    def expm(self, matrix):
        return scipy_expm(matrix)

    def norm(self, tensor):
        return float(np.linalg.norm(tensor))

    # -- Tensor operations -------------------------------------------

    def tensordot(self, a, b, axes):
        return np.tensordot(a, b, axes=axes)

    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, *operands)

    def reshape(self, tensor, shape):
        return np.reshape(tensor, shape)

    def transpose(self, tensor, axes):
        return np.transpose(tensor, axes)

    def conj(self, tensor):
        return np.conj(tensor)

    def trace(self, matrix):
        return np.trace(matrix)

    def kron(self, a, b):
        return np.kron(a, b)

    # -- Utility -----------------------------------------------------

    def to_numpy(self, tensor):
        return np.asarray(tensor)

    def random_normal(self, shape, mean=0.0, std=1.0):
        return np.random.normal(mean, std, size=shape)

    def random_uniform(self, shape, low=0.0, high=1.0):
        return np.random.uniform(low, high, size=shape)
