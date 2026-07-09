"""
siliqun.library.policy
======================
PrimitivePolicy — a callable wrapper around a loaded SiliQunLib checkpoint.

A ``PrimitivePolicy`` wraps the actor network weights of a pre-trained SAC
agent and exposes a simple ``__call__`` interface that maps an observation
vector to a deterministic action (the mean of the Gaussian policy).

The class is deliberately framework-agnostic: it uses only NumPy for
inference, so it can be used without PyTorch installed.  When PyTorch is
available, the ``from_state_dict`` class method provides a faster loading
path that avoids serialisation overhead.

Example
-------
>>> from siliqun.library import PrimitiveLibrary
>>> lib = PrimitiveLibrary()
>>> policy = lib.load("GHZ", n_qubits=3, seed=42)
>>> import numpy as np
>>> obs = np.zeros(policy.obs_dim)
>>> action = policy(obs)          # shape (action_dim,), values in [-1, 1]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class PrimitivePolicy:
    """Callable wrapper around a pre-trained SiliQunLib actor network.

    The actor is a two-hidden-layer MLP with tanh activations:

        obs → Linear(obs_dim, h) → tanh → Linear(h, h) → tanh
            → Linear(h, action_dim) → tanh → action ∈ [-1, 1]^action_dim

    Parameters
    ----------
    weights : dict[str, np.ndarray]
        Network weights keyed by layer name, e.g.
        ``{"fc1.weight": ..., "fc1.bias": ..., ...}``.
    metadata : dict
        Registry record for this checkpoint (family, n_qubits, seed, etc.).
    """

    def __init__(
        self,
        weights: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        self._weights = weights
        self._metadata = metadata
        self.obs_dim: int = metadata["obs_dim"]
        self.action_dim: int = metadata["action_dim"]
        self.hidden_dims: List[int] = metadata["hidden_dims"]
        self.family: str = metadata["family"]
        self.n_qubits: int = metadata["n_qubits"]
        self.seed: int = metadata["seed"]
        self.best_fidelity: float = metadata["best_fidelity"]
        self.hardware_profile: str = metadata["hardware_profile"]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Map an observation vector to a deterministic action.

        Parameters
        ----------
        observation : np.ndarray
            Shape ``(obs_dim,)`` or ``(batch, obs_dim)``.

        Returns
        -------
        np.ndarray
            Action array, shape ``(action_dim,)`` or ``(batch, action_dim)``,
            with values in ``[-1, 1]``.
        """
        x = np.asarray(observation, dtype=np.float32)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]  # (1, obs_dim)

        # Layer 1
        x = np.tanh(x @ self._weights["fc1.weight"].T + self._weights["fc1.bias"])
        # Layer 2
        x = np.tanh(x @ self._weights["fc2.weight"].T + self._weights["fc2.bias"])
        # Output layer — mean of Gaussian policy
        x = np.tanh(x @ self._weights["fc3.weight"].T + self._weights["fc3.bias"])

        return x[0] if squeeze else x

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """Return the raw weight dictionary (NumPy arrays)."""
        return dict(self._weights)

    @classmethod
    def from_numpy_npz(cls, npz_path: str | Path, metadata: Dict[str, Any]) -> "PrimitivePolicy":
        """Load a policy from a ``.npz`` weight file.

        Parameters
        ----------
        npz_path : str or Path
            Path to the NumPy ``.npz`` file containing the actor weights.
        metadata : dict
            Registry record for this checkpoint.
        """
        data = np.load(str(npz_path))
        weights = {k: data[k] for k in data.files}
        return cls(weights, metadata)

    @classmethod
    def from_pytorch_pt(
        cls,
        pt_path: str | Path,
        metadata: Dict[str, Any],
    ) -> "PrimitivePolicy":
        """Load a policy from a PyTorch ``.pt`` state-dict file.

        Requires PyTorch to be installed.  Falls back to the ``.npz``
        variant automatically if ``pt_path`` does not exist but a
        sibling ``.npz`` file does.

        Parameters
        ----------
        pt_path : str or Path
            Path to the PyTorch ``.pt`` state-dict file.
        metadata : dict
            Registry record for this checkpoint.
        """
        pt_path = Path(pt_path)
        npz_path = pt_path.with_suffix(".npz")

        if not pt_path.exists() and npz_path.exists():
            return cls.from_numpy_npz(npz_path, metadata)

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to load .pt checkpoints. "
                "Install it with: pip install torch  "
                "or use the .npz variant instead."
            ) from exc

        state_dict = torch.load(str(pt_path), map_location="cpu")
        weights = {k: v.numpy() for k, v in state_dict.items()}
        return cls(weights, metadata)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PrimitivePolicy("
            f"family='{self.family}', "
            f"n_qubits={self.n_qubits}, "
            f"seed={self.seed}, "
            f"best_F={self.best_fidelity:.4f}, "
            f"profile='{self.hardware_profile}')"
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Registry metadata for this checkpoint."""
        return dict(self._metadata)
