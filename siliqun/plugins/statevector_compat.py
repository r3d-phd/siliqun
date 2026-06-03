"""
siliqun.plugins.statevector_compat
===================================
Compatibility plugin that adds statevector-simulation primitives to all
SiliQun backends.

Background
----------
SiliQun's ``Backend`` abstract interface was originally designed for
tensor-network (MPS/MPO) operations, which express gate application via
``einsum`` and ``tensordot``.  Statevector simulators — such as the
ANDROMEDA v2 multi-module architecture — use a broader set of primitives:

    1. Reshape state vector (2^n,) → (2, 2, …, 2)  [n axes]
    2. Contract gate (2×2) with axis ``qubit``  via ``tensordot``
    3. Move the contracted result axis back to position ``qubit``
    4. Flatten back to (2^n,)
    5. Compute fidelity via ``dot`` (inner product)
    6. Construct initial states via ``zeros`` / ``ones`` / ``asarray``
    7. Compute absolute values, ranges, and conditional selections

Methods patched (complete statevector API):
    ``moveaxis``, ``asarray``, ``zeros``, ``ones``,
    ``abs``, ``arange``, ``dot``, ``where``

Usage
-----
Import this module *once* before using any SiliQun backend::

    import siliqun.plugins.statevector_compat  # noqa: F401 — side-effect import

Or equivalently, call :func:`apply_patch` explicitly::

    from siliqun.plugins.statevector_compat import apply_patch
    apply_patch()

The patch is idempotent: calling it multiple times is safe.

Upstreaming
-----------
This plugin is a stopgap.  The proper fix is to add these methods as
abstract methods on ``siliqun.backend.base.Backend`` and implement them in
``NumPyBackend``, ``CUDABackend``, and ``JAXBackend``.  A pull request
has been opened to track this: see the project issue tracker.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


# ── NumPy implementations ──────────────────────────────────────────────────────

def _numpy_moveaxis(self, tensor, source, destination):
    import numpy as np
    return np.moveaxis(tensor, source, destination)

def _numpy_asarray(self, data, dtype=None):
    import numpy as np
    return np.asarray(data, dtype=dtype)

def _numpy_zeros(self, shape, dtype=None):
    import numpy as np
    return np.zeros(shape, dtype=dtype if dtype is not None else np.complex128)

def _numpy_ones(self, shape, dtype=None):
    import numpy as np
    return np.ones(shape, dtype=dtype if dtype is not None else np.complex128)

def _numpy_abs(self, tensor):
    import numpy as np
    return np.abs(tensor)

def _numpy_arange(self, *args, **kwargs):
    import numpy as np
    return np.arange(*args, **kwargs)

def _numpy_dot(self, a, b):
    import numpy as np
    return np.dot(a, b)

def _numpy_where(self, condition, x, y):
    import numpy as np
    return np.where(condition, x, y)


# ── CuPy (CUDA) implementations ───────────────────────────────────────────────

def _cuda_moveaxis(self, tensor, source, destination):
    import numpy as np
    cp = self._cp
    with self._device:
        if isinstance(tensor, np.ndarray):
            tensor = cp.asarray(tensor)
        return cp.moveaxis(tensor, source, destination)

def _cuda_asarray(self, data, dtype=None):
    cp = self._cp
    with self._device:
        return cp.asarray(data, dtype=dtype)

def _cuda_zeros(self, shape, dtype=None):
    import cupy as cp
    with self._device:
        return cp.zeros(shape, dtype=dtype if dtype is not None else cp.complex128)

def _cuda_ones(self, shape, dtype=None):
    import cupy as cp
    with self._device:
        return cp.ones(shape, dtype=dtype if dtype is not None else cp.complex128)

def _cuda_abs(self, tensor):
    cp = self._cp
    with self._device:
        return cp.abs(tensor)

def _cuda_arange(self, *args, **kwargs):
    import cupy as cp
    with self._device:
        return cp.arange(*args, **kwargs)

def _cuda_dot(self, a, b):
    cp = self._cp
    with self._device:
        return cp.dot(a, b)

def _cuda_where(self, condition, x, y):
    cp = self._cp
    with self._device:
        return cp.where(condition, x, y)


# ── JAX implementations ────────────────────────────────────────────────────────

def _jax_moveaxis(self, tensor, source, destination):
    import jax.numpy as jnp
    return jnp.moveaxis(tensor, source, destination)

def _jax_asarray(self, data, dtype=None):
    import jax.numpy as jnp
    return jnp.asarray(data, dtype=dtype)

def _jax_zeros(self, shape, dtype=None):
    import jax.numpy as jnp
    return jnp.zeros(shape, dtype=dtype)

def _jax_ones(self, shape, dtype=None):
    import jax.numpy as jnp
    return jnp.ones(shape, dtype=dtype)

def _jax_abs(self, tensor):
    import jax.numpy as jnp
    return jnp.abs(tensor)

def _jax_arange(self, *args, **kwargs):
    import jax.numpy as jnp
    return jnp.arange(*args, **kwargs)

def _jax_dot(self, a, b):
    import jax.numpy as jnp
    return jnp.dot(a, b)

def _jax_where(self, condition, x, y):
    import jax.numpy as jnp
    return jnp.where(condition, x, y)


# ── Patch registry ─────────────────────────────────────────────────────────────

_NUMPY_PATCHES = {
    "moveaxis": _numpy_moveaxis,
    "asarray":  _numpy_asarray,
    "zeros":    _numpy_zeros,
    "ones":     _numpy_ones,
    "abs":      _numpy_abs,
    "arange":   _numpy_arange,
    "dot":      _numpy_dot,
    "where":    _numpy_where,
}

_CUDA_PATCHES = {
    "moveaxis": _cuda_moveaxis,
    "asarray":  _cuda_asarray,
    "zeros":    _cuda_zeros,
    "ones":     _cuda_ones,
    "abs":      _cuda_abs,
    "arange":   _cuda_arange,
    "dot":      _cuda_dot,
    "where":    _cuda_where,
}

_JAX_PATCHES = {
    "moveaxis": _jax_moveaxis,
    "asarray":  _jax_asarray,
    "zeros":    _jax_zeros,
    "ones":     _jax_ones,
    "abs":      _jax_abs,
    "arange":   _jax_arange,
    "dot":      _jax_dot,
    "where":    _jax_where,
}


def _apply_patches(cls, patches: dict, cls_name: str, patched: list[str]) -> None:
    """Inject any missing methods from *patches* into *cls*."""
    added = []
    for name, fn in patches.items():
        if not hasattr(cls, name):
            setattr(cls, name, fn)
            added.append(name)
    if added:
        patched.append(f"{cls_name}({', '.join(added)})")


def apply_patch() -> None:
    """Inject statevector primitives into any SiliQun backend class missing them.

    Methods patched: ``moveaxis``, ``asarray``, ``zeros``, ``ones``,
    ``abs``, ``arange``, ``dot``, ``where``.
    This function is idempotent and safe to call multiple times.
    """
    patched: list[str] = []

    try:
        from siliqun.backend.numpy_backend import NumPyBackend
        _apply_patches(NumPyBackend, _NUMPY_PATCHES, "NumPyBackend", patched)
    except ImportError:
        pass

    try:
        from siliqun.backend.cuda_backend import CUDABackend
        _apply_patches(CUDABackend, _CUDA_PATCHES, "CUDABackend", patched)
    except ImportError:
        pass

    try:
        from siliqun.backend.jax_backend import JAXBackend
        _apply_patches(JAXBackend, _JAX_PATCHES, "JAXBackend", patched)
    except ImportError:
        pass

    if patched:
        logger.info(
            "siliqun.plugins.statevector_compat: patched %s",
            "; ".join(patched),
        )
    else:
        logger.debug(
            "siliqun.plugins.statevector_compat: all backends already have required methods"
        )


# Apply automatically on import (side-effect import pattern)
apply_patch()
