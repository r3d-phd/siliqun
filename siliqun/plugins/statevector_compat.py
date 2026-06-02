"""
siliqun.plugins.statevector_compat
===================================
Compatibility plugin that adds ``moveaxis`` to all SiliQun backends.

Background
----------
SiliQun's ``Backend`` abstract interface was originally designed for
tensor-network (MPS/MPO) operations, which express gate application via
``einsum`` and ``tensordot``.  Statevector simulators — such as the
ANDROMEDA v2 multi-module architecture — use a different primitive:

    1. Reshape state vector (2^n,) → (2, 2, …, 2)  [n axes]
    2. Contract gate (2×2) with axis ``qubit``  via ``tensordot``
    3. **Move** the contracted result axis back to position ``qubit``
    4. Flatten back to (2^n,)

Step 3 requires ``moveaxis``, which is present in both NumPy and CuPy but
was not included in the SiliQun ``Backend`` interface.

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
This plugin is a stopgap.  The proper fix is to add ``moveaxis`` as an
abstract method on ``siliqun.backend.base.Backend`` and implement it in
``NumPyBackend``, ``CUDABackend``, and ``JAXBackend``.  A pull request
has been opened to track this: see the project issue tracker.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def _numpy_moveaxis(self, tensor, source, destination):
    """NumPy implementation of moveaxis for SiliQun backends."""
    import numpy as np
    return np.moveaxis(tensor, source, destination)


def _cuda_moveaxis(self, tensor, source, destination):
    """CuPy implementation of moveaxis for SiliQun CUDA backend."""
    import numpy as np
    cp = self._cp  # CUDABackend stores its cupy reference as self._cp
    with self._device:
        if isinstance(tensor, np.ndarray):
            tensor = cp.asarray(tensor)
        return cp.moveaxis(tensor, source, destination)


def _jax_moveaxis(self, tensor, source, destination):
    """JAX implementation of moveaxis for SiliQun JAX backend."""
    import jax.numpy as jnp
    return jnp.moveaxis(tensor, source, destination)


def apply_patch() -> None:
    """Inject ``moveaxis`` into any SiliQun backend class that is missing it.

    This function is idempotent and safe to call multiple times.
    """
    patched: list[str] = []

    try:
        from siliqun.backend.numpy_backend import NumPyBackend
        if not hasattr(NumPyBackend, "moveaxis"):
            NumPyBackend.moveaxis = _numpy_moveaxis  # type: ignore[attr-defined]
            patched.append("NumPyBackend")
    except ImportError:
        pass

    try:
        from siliqun.backend.cuda_backend import CUDABackend
        if not hasattr(CUDABackend, "moveaxis"):
            CUDABackend.moveaxis = _cuda_moveaxis  # type: ignore[attr-defined]
            patched.append("CUDABackend")
    except ImportError:
        pass

    try:
        from siliqun.backend.jax_backend import JAXBackend
        if not hasattr(JAXBackend, "moveaxis"):
            JAXBackend.moveaxis = _jax_moveaxis  # type: ignore[attr-defined]
            patched.append("JAXBackend")
    except ImportError:
        pass

    if patched:
        logger.info(
            "siliqun.plugins.statevector_compat: patched moveaxis onto %s",
            ", ".join(patched),
        )
    else:
        logger.debug(
            "siliqun.plugins.statevector_compat: all backends already have moveaxis"
        )


# Apply automatically on import (side-effect import pattern)
apply_patch()
