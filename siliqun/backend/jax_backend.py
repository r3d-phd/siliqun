"""
JAX compute backend — GPU-accelerated with JIT compilation.

Provides automatic differentiation and XLA compilation for
high-performance tensor network operations on GPU/TPU.
"""

from typing import Tuple, Union
import numpy as np

from .base import Backend

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm as jax_expm
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class JAXBackend(Backend):
    """JAX backend for GPU-accelerated computation with JIT."""

    name = "jax"

    def __init__(self):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )

    def complex_dtype(self):
        return jnp.complex128

    def real_dtype(self):
        return jnp.float64

    # ── Array creation ──────────────────────────────────────────────

    def zeros(self, shape, dtype=None):
        return jnp.zeros(shape, dtype=dtype or self.complex_dtype())

    def ones(self, shape, dtype=None):
        return jnp.ones(shape, dtype=dtype or self.complex_dtype())

    def eye(self, n, dtype=None):
        return jnp.eye(n, dtype=dtype or self.complex_dtype())

    def array(self, data, dtype=None):
        return jnp.asarray(data, dtype=dtype or self.complex_dtype())

    # ── Linear algebra ──────────────────────────────────────────────

    def svd(self, tensor, full_matrices=False):
        return jnp.linalg.svd(tensor, full_matrices=full_matrices)

    def qr(self, tensor):
        return jnp.linalg.qr(tensor)

    def eigh(self, matrix):
        return jnp.linalg.eigh(matrix)

    def expm(self, matrix):
        return jax_expm(matrix)

    def norm(self, tensor):
        return float(jnp.linalg.norm(tensor))

    # ── Tensor operations ───────────────────────────────────────────

    def tensordot(self, a, b, axes):
        return jnp.tensordot(a, b, axes=axes)

    def einsum(self, subscripts, *operands):
        return jnp.einsum(subscripts, *operands)

    def reshape(self, tensor, shape):
        return jnp.reshape(tensor, shape)

    def transpose(self, tensor, axes):
        return jnp.transpose(tensor, axes)

    def conj(self, tensor):
        return jnp.conj(tensor)

    def trace(self, matrix):
        return jnp.trace(matrix)

    def kron(self, a, b):
        return jnp.kron(a, b)

    # ── Utility ─────────────────────────────────────────────────────

    def to_numpy(self, tensor):
        return np.asarray(tensor)

    def random_normal(self, shape, mean=0.0, std=1.0):
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        return mean + std * jax.random.normal(key, shape=shape)

    def random_uniform(self, shape, low=0.0, high=1.0):
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        return low + (high - low) * jax.random.uniform(key, shape=shape)
