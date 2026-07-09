"""
siliqun.library.primitive_library
==================================
PrimitiveLibrary — the main access point for SiliQunLib checkpoints.

Usage
-----
>>> from siliqun.library import PrimitiveLibrary
>>> lib = PrimitiveLibrary()
>>> print(len(lib.list_primitives()))   # 50
>>> policy = lib.load("GHZ", n_qubits=3, seed=42)
>>> print(policy)
PrimitivePolicy(family='GHZ', n_qubits=3, seed=42, best_F=0.9977, profile='simos_nominal')

Checkpoint storage
------------------
Checkpoints are stored as NumPy ``.npz`` files in the ``data/`` subdirectory
of this package.  When a ``.pt`` PyTorch file is present alongside the
``.npz`` file, the PyTorch file is preferred if PyTorch is installed.

If a checkpoint file is not present locally, ``PrimitiveLibrary`` can
generate a synthetic placeholder policy (random weights) for testing
purposes.  Real checkpoints from the v19/v27 training runs on Aziz HPC
and the local GPU are stored in the ``data/`` directory and committed to
the repository.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from siliqun.library.policy import PrimitivePolicy
from siliqun.library.registry import CHECKPOINT_REGISTRY, lookup, list_families

# Default data directory: siliqun/library/data/
_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


class PrimitiveLibrary:
    """Curated repository of pre-trained primitive gate policies.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the checkpoint ``.npz`` / ``.pt`` files.
        Defaults to ``siliqun/library/data/``.
    allow_synthetic : bool, optional
        If ``True`` (default), generate a random-weight placeholder policy
        when the checkpoint file is not found on disk.  Set to ``False``
        to raise ``FileNotFoundError`` instead.
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        allow_synthetic: bool = True,
    ) -> None:
        """Initialise the PrimitiveLibrary.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory containing the checkpoint ``.npz`` / ``.pt`` files.
            Defaults to ``siliqun/library/data/`` inside the package.
        allow_synthetic : bool, optional
            If ``True`` (default), return a random-weight placeholder policy
            when a checkpoint file is not found on disk.  Set to ``False``
            to raise ``FileNotFoundError`` instead, which is safer in
            production workflows where missing checkpoints indicate a
            configuration error.
        """
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self._allow_synthetic = allow_synthetic
        self._cache: Dict[str, PrimitivePolicy] = {}

    # ------------------------------------------------------------------
    # Catalogue API
    # ------------------------------------------------------------------

    def list_primitives(self) -> List[Dict[str, Any]]:
        """Return metadata for all 50 checkpoints in the registry.

        Returns
        -------
        list[dict]
            One dict per checkpoint with keys: ``checkpoint_id``,
            ``family``, ``n_qubits``, ``seed``, ``best_fidelity``,
            ``hardware_profile``, ``action_dim``, ``obs_dim``.
        """
        return list(CHECKPOINT_REGISTRY)

    def list_families(self) -> List[str]:
        """Return the list of distinct target families.

        Returns
        -------
        list[str]
            E.g. ``['Bell', 'GHZ', 'W', 'Cluster', 'Dicke-k2']``.
        """
        return list_families()

    def filter(
        self,
        family: Optional[str] = None,
        n_qubits: Optional[int] = None,
        seed: Optional[int] = None,
        min_fidelity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Filter the registry by one or more criteria.

        Parameters
        ----------
        family : str, optional
            Target state family (case-insensitive).
        n_qubits : int, optional
            Number of qubits.
        seed : int, optional
            Training seed.
        min_fidelity : float, optional
            Minimum ``best_fidelity`` threshold (default 0.0).

        Returns
        -------
        list[dict]
            Matching registry records.
        """
        results = []
        for rec in CHECKPOINT_REGISTRY:
            if family is not None and rec["family"].lower() != family.lower():
                continue
            if n_qubits is not None and rec["n_qubits"] != n_qubits:
                continue
            if seed is not None and rec["seed"] != seed:
                continue
            if rec["best_fidelity"] < min_fidelity:
                continue
            results.append(rec)
        return results

    # ------------------------------------------------------------------
    # Loading API
    # ------------------------------------------------------------------

    def load(
        self,
        family: str,
        n_qubits: int,
        seed: int = 42,
        force_reload: bool = False,
    ) -> PrimitivePolicy:
        """Load a pre-trained primitive policy.

        Parameters
        ----------
        family : str
            Target state family, e.g. ``"GHZ"``, ``"Bell"``, ``"W"``,
            ``"Cluster"``, ``"Dicke-k2"``.
        n_qubits : int
            Number of qubits (2–5).
        seed : int, optional
            Training seed (42 or 456).  Default is 42.
        force_reload : bool, optional
            If ``True``, bypass the in-memory cache and reload from disk.

        Returns
        -------
        PrimitivePolicy
            Callable policy object.

        Raises
        ------
        KeyError
            If no matching checkpoint exists in the registry.
        FileNotFoundError
            If the checkpoint file is missing and ``allow_synthetic=False``.
        """
        meta = lookup(family, n_qubits, seed)
        cid = meta["checkpoint_id"]

        if not force_reload and cid in self._cache:
            return self._cache[cid]

        policy = self._load_from_disk(meta)
        self._cache[cid] = policy
        return policy

    def load_all(
        self,
        family: Optional[str] = None,
        min_fidelity: float = 0.0,
    ) -> List[PrimitivePolicy]:
        """Load all checkpoints matching the given criteria.

        Parameters
        ----------
        family : str, optional
            If given, load only checkpoints of this family.
        min_fidelity : float, optional
            Minimum fidelity threshold.

        Returns
        -------
        list[PrimitivePolicy]
        """
        records = self.filter(family=family, min_fidelity=min_fidelity)
        return [
            self.load(r["family"], r["n_qubits"], r["seed"])
            for r in records
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self, meta: Dict[str, Any]) -> PrimitivePolicy:
        """Attempt to load from .pt then .npz; fall back to synthetic."""
        filename = meta["filename"]
        pt_path  = self._data_dir / filename
        npz_path = self._data_dir / filename.replace(".pt", ".npz")

        if pt_path.exists():
            return PrimitivePolicy.from_pytorch_pt(pt_path, meta)

        if npz_path.exists():
            return PrimitivePolicy.from_numpy_npz(npz_path, meta)

        if self._allow_synthetic:
            warnings.warn(
                f"Checkpoint file not found for '{meta['checkpoint_id']}' "
                f"in {self._data_dir}. Returning a synthetic (random-weight) "
                f"placeholder policy. This is only suitable for testing.",
                UserWarning,
                stacklevel=3,
            )
            return self._make_synthetic(meta)

        raise FileNotFoundError(
            f"Checkpoint file not found: {pt_path} or {npz_path}. "
            f"Run `siliqun-library download` to fetch the checkpoints, "
            f"or set allow_synthetic=True for testing."
        )

    @staticmethod
    def _make_synthetic(meta: Dict[str, Any]) -> PrimitivePolicy:
        """Generate a random-weight placeholder policy for testing.

        Uses Xavier uniform initialisation seeded from the checkpoint's
        training seed so that the same checkpoint ID always produces the
        same synthetic weights, enabling reproducible unit tests.

        Parameters
        ----------
        meta : dict
            Registry record for the requested checkpoint.

        Returns
        -------
        PrimitivePolicy
            Placeholder policy with random weights.  A ``UserWarning`` is
            issued by the caller before this method is invoked.
        """
        rng = np.random.default_rng(seed=meta["seed"])
        obs_dim    = meta["obs_dim"]
        action_dim = meta["action_dim"]
        h1, h2     = meta["hidden_dims"]

        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            """Xavier uniform initialisation for a weight matrix.

            Parameters
            ----------
            fan_in : int
                Number of input units.
            fan_out : int
                Number of output units.

            Returns
            -------
            np.ndarray
                Weight matrix of shape ``(fan_out, fan_in)`` with values
                drawn uniformly from ``[-limit, limit]`` where
                ``limit = sqrt(6 / (fan_in + fan_out))``.
            """
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)

        weights = {
            "fc1.weight": _xavier(obs_dim, h1),
            "fc1.bias":   np.zeros(h1, dtype=np.float32),
            "fc2.weight": _xavier(h1, h2),
            "fc2.bias":   np.zeros(h2, dtype=np.float32),
            "fc3.weight": _xavier(h2, action_dim),
            "fc3.bias":   np.zeros(action_dim, dtype=np.float32),
        }
        return PrimitivePolicy(weights, meta)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation of this library."""
        return (
            f"PrimitiveLibrary("
            f"n_checkpoints={len(CHECKPOINT_REGISTRY)}, "
            f"families={self.list_families()}, "
            f"data_dir='{self._data_dir}')"
        )

    def summary(self) -> str:
        """Return a human-readable summary table of all checkpoints."""
        lines = [
            f"{'Checkpoint ID':<30} {'Family':<12} {'Qubits':>6} "
            f"{'Seed':>6} {'Best F':>8} {'Profile':<16}",
            "-" * 82,
        ]
        for rec in CHECKPOINT_REGISTRY:
            lines.append(
                f"{rec['checkpoint_id']:<30} {rec['family']:<12} "
                f"{rec['n_qubits']:>6} {rec['seed']:>6} "
                f"{rec['best_fidelity']:>8.4f} {rec['hardware_profile']:<16}"
            )
        lines.append("-" * 82)
        lines.append(f"Total: {len(CHECKPOINT_REGISTRY)} checkpoints")
        return "\n".join(lines)
