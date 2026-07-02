"""
Compute backend selection and management.

Usage:
    from siliqun.backend import get_backend
    backend = get_backend("numpy")   # CPU reference
    backend = get_backend("jax")     # GPU-accelerated
"""

from .base import Backend
from .numpy_backend import NumPyBackend

_BACKENDS = {
    "numpy": NumPyBackend,
}

# Register optional backends
try:
    from .jax_backend import JAXBackend
    _BACKENDS["jax"] = JAXBackend
except ImportError:
    pass

try:
    from .cuda_backend import CUDABackend
    _BACKENDS["cuda"] = CUDABackend
except ImportError:
    pass

_active_backend = None


def get_backend(name: str = "numpy") -> Backend:
    """Get a compute backend by name."""
    global _active_backend
    name = name.lower()
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown backend '{name}'. Available: {available}"
        )
    _active_backend = _BACKENDS[name]()
    return _active_backend


def set_backend(backend, **kwargs):
    """Set the active backend by name or instance."""
    global _active_backend
    if isinstance(backend, str):
        _active_backend = get_backend(backend, **kwargs) if kwargs else get_backend(backend)
    else:
        _active_backend = backend


def active_backend() -> Backend:
    """Return the currently active backend, defaulting to NumPy."""
    global _active_backend
    if _active_backend is None:
        _active_backend = NumPyBackend()
    return _active_backend
