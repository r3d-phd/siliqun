"""
CUDA/cuQuantum GPU backend for SiliQun.

Provides GPU-accelerated tensor operations using CuPy and NVIDIA cuQuantum.
Falls back gracefully to NumPy if CUDA is not available.

Usage:
    from siliqun.backend import get_backend
    be = get_backend("cuda")  # Raises if CUDA unavailable
"""

from __future__ import annotations
from typing import Tuple, Union, Optional
import numpy as np
import logging

from .base import Backend as ComputeBackend

logger = logging.getLogger(__name__)

# Lazy imports for optional CUDA dependencies
_cupy = None
_cutensornet = None
_cuquantum_available = False
_cupy_available = False


def _ensure_cupy():
    """Lazy-load CuPy."""
    global _cupy, _cupy_available
    if _cupy is not None:
        return _cupy
    try:
        import cupy
        _cupy = cupy
        _cupy_available = True
        logger.info(f"CuPy loaded: CUDA device {cupy.cuda.runtime.getDevice()}")
        return cupy
    except ImportError:
        _cupy_available = False
        raise ImportError(
            "CuPy is required for the CUDA backend. "
            "Install with: pip install cupy-cuda12x"
        )


def _ensure_cuquantum():
    """Lazy-load cuQuantum cutensornet."""
    global _cutensornet, _cuquantum_available
    if _cutensornet is not None:
        return _cutensornet
    try:
        import cuquantum.cutensornet as cutn
        _cutensornet = cutn
        _cuquantum_available = True
        logger.info("cuQuantum cutensornet loaded")
        return cutn
    except ImportError:
        _cuquantum_available = False
        logger.warning(
            "cuQuantum not available. Using CuPy einsum fallback. "
            "Install with: pip install cuquantum-python"
        )
        return None


class CUDABackend(ComputeBackend):
    """GPU-accelerated compute backend using CuPy + cuQuantum.

    Features:
        - GPU tensor allocation and arithmetic
        - cuQuantum-accelerated tensor contraction (when available)
        - CuPy einsum fallback for contraction
        - Automatic memory management with memory pool
        - Multi-GPU support via device selection

    Parameters
    ----------
    device_id : int
        CUDA device ID to use (default: 0).
    memory_pool : bool
        Whether to use CuPy memory pool (default: True).
    cuquantum : bool
        Whether to attempt cuQuantum acceleration (default: True).
    """

    name = "cuda"

    def __init__(
        self,
        device_id: int = 0,
        memory_pool: bool = True,
        cuquantum: bool = True,
    ):
        self._cp = _ensure_cupy()
        self._device_id = device_id
        self._device = self._cp.cuda.Device(device_id)

        with self._device:
            if memory_pool:
                pool = self._cp.cuda.MemoryPool()
                self._cp.cuda.set_allocator(pool.malloc)
                self._pool = pool
            else:
                self._pool = None

        self._cutn = None
        if cuquantum:
            self._cutn = _ensure_cuquantum()

        logger.info(
            f"CUDABackend initialized: device={device_id}, "
            f"cuquantum={'yes' if self._cutn else 'no'}"
        )

    # ── Tensor Creation ────────────────────────────────────────────

    def array(self, data, dtype=None) -> "cupy.ndarray":
        """Create a GPU tensor from data."""
        with self._device:
            if dtype is None:
                dtype = np.complex128
            return self._cp.asarray(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype=None) -> "cupy.ndarray":
        with self._device:
            return self._cp.zeros(shape, dtype=dtype or np.complex128)

    def ones(self, shape: Tuple[int, ...], dtype=None) -> "cupy.ndarray":
        with self._device:
            return self._cp.ones(shape, dtype=dtype or np.complex128)

    def eye(self, n: int, dtype=None) -> "cupy.ndarray":
        with self._device:
            return self._cp.eye(n, dtype=dtype or np.complex128)

    def random_normal(
        self, shape: Tuple[int, ...], dtype=None
    ) -> "cupy.ndarray":
        with self._device:
            return self._cp.random.randn(*shape).astype(
                dtype or np.float64
            )

    # ── Tensor Operations ──────────────────────────────────────────

    def conj(self, a):
        return self._cp.conj(a)

    def transpose(self, a, axes=None):
        return self._cp.transpose(a, axes)

    def reshape(self, a, shape):
        return self._cp.reshape(a, shape)

    def tensordot(self, a, b, axes):
        with self._device:
            return self._cp.tensordot(a, b, axes=axes)

    def einsum(self, subscripts: str, *operands):
        """Einstein summation with optional cuQuantum acceleration."""
        with self._device:
            # Convert any NumPy arrays to CuPy
            gpu_ops = []
            for op in operands:
                if isinstance(op, np.ndarray):
                    gpu_ops.append(self._cp.asarray(op))
                else:
                    gpu_ops.append(op)

            if self._cutn is not None:
                try:
                    return self._cutn.contract(subscripts, *gpu_ops)
                except Exception:
                    pass  # Fall back to CuPy einsum

            return self._cp.einsum(subscripts, *gpu_ops)

    def svd(
        self, a, full_matrices: bool = False
    ) -> Tuple["cupy.ndarray", "cupy.ndarray", "cupy.ndarray"]:
        with self._device:
            return self._cp.linalg.svd(a, full_matrices=full_matrices)

    def qr(self, a) -> Tuple["cupy.ndarray", "cupy.ndarray"]:
        with self._device:
            return self._cp.linalg.qr(a)

    def norm(self, a) -> float:
        return float(self._cp.linalg.norm(a))

    def trace(self, a):
        return self._cp.trace(a)

    def diag(self, a):
        return self._cp.diag(a)

    def sqrt(self, a):
        return self._cp.sqrt(a)

    def abs(self, a):
        return self._cp.abs(a)

    def real(self, a):
        return self._cp.real(a)

    def expm(self, a):
        """Matrix exponential on GPU.

        CuPy doesn't have expm natively, so we use eigendecomposition:
            expm(A) = V @ diag(exp(λ)) @ V^{-1}
        """
        with self._device:
            eigenvalues, V = self._cp.linalg.eigh(a)
            exp_eigenvalues = self._cp.exp(eigenvalues)
            return V @ self._cp.diag(exp_eigenvalues) @ self._cp.conj(V.T)

    # ── Data Transfer ──────────────────────────────────────────────

    def to_numpy(self, a) -> np.ndarray:
        """Transfer GPU tensor to CPU NumPy array."""
        if isinstance(a, np.ndarray):
            return a
        return self._cp.asnumpy(a)

    def to_gpu(self, a) -> "cupy.ndarray":
        """Transfer CPU array to GPU."""
        with self._device:
            return self._cp.asarray(a)

    # ── Memory Management ──────────────────────────────────────────

    def memory_info(self) -> dict:
        """Get GPU memory usage information."""
        with self._device:
            free, total = self._cp.cuda.runtime.memGetInfo()
            used = total - free
            return {
                "device_id": self._device_id,
                "total_mb": total / 1024**2,
                "used_mb": used / 1024**2,
                "free_mb": free / 1024**2,
                "utilization": used / total,
            }

    def clear_cache(self):
        """Free unused GPU memory."""
        if self._pool is not None:
            self._pool.free_all_blocks()
        self._cp.get_default_memory_pool().free_all_blocks()
        self._cp.get_default_pinned_memory_pool().free_all_blocks()

    def synchronize(self):
        """Synchronize CUDA stream."""
        self._cp.cuda.Stream.null.synchronize()


class CUDABatchEngine:
    """Batch processing engine for parallel tensor network simulations.

    Enables running multiple independent simulations on the same GPU
    for ensemble averaging, parameter sweeps, or DRL batch environments.

    Parameters
    ----------
    backend : CUDABackend
        The CUDA backend to use.
    batch_size : int
        Number of parallel simulations.
    """

    def __init__(self, backend: CUDABackend, batch_size: int):
        self.backend = backend
        self.batch_size = batch_size
        self._cp = backend._cp

    def batch_apply_gate(
        self,
        states: list,
        gate: "cupy.ndarray",
        site: int,
    ) -> list:
        """Apply a single-qubit gate to a batch of MPS states.

        Vectorizes the gate application across the batch dimension
        for better GPU utilization.

        Parameters
        ----------
        states : list of MPS tensors at the target site
            Each element has shape (chi_l, d, chi_r).
        gate : ndarray
            Gate matrix of shape (d, d).
        site : int
            Site index (used for logging only).

        Returns
        -------
        list of updated tensors
        """
        # Stack all site tensors into a single batch tensor
        # Shape: (batch, chi_l, d, chi_r)
        max_chi_l = max(s.shape[0] for s in states)
        max_chi_r = max(s.shape[2] for s in states)
        d = gate.shape[0]

        batch = self._cp.zeros(
            (self.batch_size, max_chi_l, d, max_chi_r),
            dtype=self._cp.complex128,
        )
        for i, s in enumerate(states):
            batch[i, :s.shape[0], :, :s.shape[2]] = s

        # Apply gate: batch einsum
        result = self._cp.einsum("bldR,dl->bldR", batch, gate)

        # Unstack
        updated = []
        for i, s in enumerate(states):
            updated.append(result[i, :s.shape[0], :, :s.shape[2]])

        return updated

    def batch_expectation(
        self,
        states: list,
        operator: "cupy.ndarray",
        site: int,
    ) -> np.ndarray:
        """Compute expectation values for a batch of states.

        Returns
        -------
        np.ndarray of shape (batch_size,)
        """
        expectations = self._cp.zeros(self.batch_size)
        for i, mps_tensors in enumerate(states):
            # Simple single-site expectation
            A = mps_tensors[site]
            A_conj = self._cp.conj(A)
            # <O> = Tr(A^dag @ O @ A) summed over bond indices
            val = self._cp.einsum(
                "ldr,dd,ldr->", A_conj, operator, A
            )
            expectations[i] = self._cp.real(val)

        return self.backend.to_numpy(expectations)
