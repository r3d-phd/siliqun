"""
Core Tensor class — the fundamental building block of tensor networks.

A Tensor wraps a multi-dimensional array with named indices (legs),
enabling index-based contraction, decomposition, and reshaping.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import numpy as np
from ..backend import active_backend


class Tensor:
    """A named-index tensor for tensor network operations.

    Parameters
    ----------
    data : array-like
        The tensor data.
    inds : sequence of str
        Names for each axis/index of the tensor.
    tags : set of str, optional
        Tags for grouping and selecting tensors in a network.
    """

    __slots__ = ("_data", "_inds", "_tags")

    def __init__(self, data, inds: Sequence[str], tags: Optional[set] = None):
        be = active_backend()
        self._data = be.array(data)
        self._inds = tuple(inds)
        self._tags = set(tags) if tags else set()

        if len(self._inds) != len(self._data.shape):
            raise ValueError(
                f"Number of index names ({len(self._inds)}) does not match "
                f"tensor rank ({len(self._data.shape)})"
            )

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value):
        be = active_backend()
        self._data = be.array(value)

    @property
    def inds(self) -> Tuple[str, ...]:
        return self._inds

    @property
    def tags(self) -> set:
        return self._tags

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return len(self._inds)

    @property
    def size(self) -> int:
        return int(np.prod(self._data.shape))

    def norm(self) -> float:
        """Frobenius norm of the tensor."""
        be = active_backend()
        return be.norm(self._data)

    def conj(self) -> Tensor:
        """Return the complex conjugate."""
        be = active_backend()
        return Tensor(be.conj(self._data), self._inds, self._tags.copy())

    def reindex(self, mapping: dict) -> Tensor:
        """Rename indices according to a mapping dict."""
        new_inds = tuple(mapping.get(i, i) for i in self._inds)
        return Tensor(self._data.copy(), new_inds, self._tags.copy())

    def fuse_inds(self, inds_to_fuse: Sequence[str], new_ind: str) -> Tensor:
        """Fuse (combine) multiple indices into a single index."""
        be = active_backend()
        axes_to_fuse = [self._inds.index(i) for i in inds_to_fuse]
        other_axes = [a for a in range(self.ndim) if a not in axes_to_fuse]

        perm = other_axes + axes_to_fuse
        data = be.transpose(self._data, tuple(perm))

        other_inds = [self._inds[a] for a in other_axes]
        fused_dim = 1
        for a in axes_to_fuse:
            fused_dim *= self._data.shape[a]

        new_shape = tuple(self._data.shape[a] for a in other_axes) + (fused_dim,)
        data = be.reshape(data, new_shape)

        return Tensor(data, tuple(other_inds) + (new_ind,), self._tags.copy())

    def copy(self) -> Tensor:
        """Return a deep copy."""
        return Tensor(self._data.copy(), self._inds, self._tags.copy())

    def __repr__(self) -> str:
        shape_str = "×".join(str(s) for s in self.shape)
        return f"Tensor(shape={shape_str}, inds={self._inds})"


def contract(t1: Tensor, t2: Tensor) -> Tensor:
    """Contract two tensors over their shared indices.

    Shared indices (same name) are summed over; unshared indices
    are preserved in the result.
    """
    be = active_backend()

    shared = set(t1.inds) & set(t2.inds)
    if not shared:
        raise ValueError("No shared indices to contract over.")

    axes1 = [t1.inds.index(i) for i in shared]
    axes2 = [t2.inds.index(i) for i in shared]

    result_data = be.tensordot(t1.data, t2.data, (axes1, axes2))

    remaining1 = [i for i in t1.inds if i not in shared]
    remaining2 = [i for i in t2.inds if i not in shared]
    result_inds = tuple(remaining1 + remaining2)

    result_tags = t1.tags | t2.tags

    return Tensor(result_data, result_inds, result_tags)


def tensor_svd(
    tensor: Tensor,
    left_inds: Sequence[str],
    right_inds: Optional[Sequence[str]] = None,
    max_bond: Optional[int] = None,
    cutoff: float = 1e-12,
    absorb: str = "both",
    bond_ind: str = "bond",
) -> Tuple[Tensor, Tensor]:
    """SVD decomposition of a tensor into two tensors.

    Parameters
    ----------
    tensor : Tensor
        The tensor to decompose.
    left_inds : sequence of str
        Indices that go to the left (U) tensor.
    right_inds : sequence of str, optional
        Indices that go to the right (Vh) tensor. If None, inferred.
    max_bond : int, optional
        Maximum bond dimension (truncation).
    cutoff : float
        Singular values below this threshold are discarded.
    absorb : str
        Where to absorb singular values: "left", "right", or "both".
    bond_ind : str
        Name for the new bond index.

    Returns
    -------
    left_tensor, right_tensor : Tensor, Tensor
    """
    be = active_backend()

    if right_inds is None:
        right_inds = [i for i in tensor.inds if i not in left_inds]

    left_axes = [tensor.inds.index(i) for i in left_inds]
    right_axes = [tensor.inds.index(i) for i in right_inds]

    perm = left_axes + right_axes
    data = be.transpose(tensor.data, tuple(perm))

    left_shape = tuple(tensor.data.shape[a] for a in left_axes)
    right_shape = tuple(tensor.data.shape[a] for a in right_axes)

    m = int(np.prod(left_shape))
    n = int(np.prod(right_shape))
    mat = be.reshape(data, (m, n))

    U, S, Vh = be.svd(mat, full_matrices=False)

    # Truncation
    mask = S > cutoff
    if max_bond is not None:
        mask[max_bond:] = False
    chi = int(np.sum(be.to_numpy(mask)))
    chi = max(chi, 1)  # keep at least 1

    U = U[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]

    # Absorb singular values
    if absorb == "left":
        U = U * S[None, :]
    elif absorb == "right":
        Vh = S[:, None] * Vh
    elif absorb == "both":
        sqrt_S = be.array(np.sqrt(be.to_numpy(S)))
        U = U * sqrt_S[None, :]
        Vh = sqrt_S[:, None] * Vh

    left_data = be.reshape(U, left_shape + (chi,))
    right_data = be.reshape(Vh, (chi,) + right_shape)

    left_tensor = Tensor(left_data, tuple(left_inds) + (bond_ind,), tensor.tags.copy())
    right_tensor = Tensor(right_data, (bond_ind,) + tuple(right_inds), tensor.tags.copy())

    return left_tensor, right_tensor
