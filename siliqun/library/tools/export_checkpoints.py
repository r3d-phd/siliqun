"""
siliqun.library.tools.export_checkpoints
=========================================
Convert QUASAR training results to SiliQunLib ``.npz`` checkpoint files.

This script reads the actor network weights saved by the QUASAR v19/v27
training runs (stored as PyTorch ``.pt`` files in the results directory)
and converts them to hardware-agnostic NumPy ``.npz`` files suitable for
distribution as part of SiliQunLib.

Usage
-----
::

    python -m siliqun.library.tools.export_checkpoints \\
        --results-dir /home/raad/quasar_v19/results \\
        --output-dir  siliqun/library/data \\
        --dry-run

Arguments
---------
--results-dir : str
    Directory containing the QUASAR training result JSON files and
    actor checkpoint ``.pt`` files.
--output-dir : str
    Destination directory for the ``.npz`` output files.
    Defaults to ``siliqun/library/data/``.
--dry-run : flag
    Print what would be exported without writing any files.
--min-fidelity : float
    Only export checkpoints with ``best_F >= min_fidelity``.
    Default: 0.90.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the export script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields ``results_dir``, ``output_dir``,
        ``dry_run``, and ``min_fidelity``.
    """
    parser = argparse.ArgumentParser(
        description="Export QUASAR actor checkpoints to SiliQunLib .npz format."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing QUASAR training results (.json + .pt files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory for .npz files (default: siliqun/library/data/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be exported without writing files.",
    )
    parser.add_argument(
        "--min-fidelity",
        type=float,
        default=0.90,
        help="Minimum best_F threshold for export (default: 0.90).",
    )
    return parser.parse_args()


def _load_actor_weights_from_pt(pt_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load actor weights from a PyTorch .pt state-dict file."""
    try:
        import torch  # type: ignore
        state_dict = torch.load(str(pt_path), map_location="cpu")
        # Extract only the actor mean-network layers (fc1, fc2, fc3)
        weights = {}
        for k, v in state_dict.items():
            if any(k.startswith(prefix) for prefix in ("fc1.", "fc2.", "fc3.", "mean_linear.")):
                key = k.replace("mean_linear.", "fc3.")
                weights[key] = v.numpy().astype(np.float32)
        return weights if weights else None
    except Exception as exc:
        print(f"  WARNING: could not load {pt_path}: {exc}", file=sys.stderr)
        return None


def _family_from_target(target: str) -> str:
    """Map a target state name to a SiliQunLib family name."""
    t = target.lower()
    if "bell" in t:
        return "Bell"
    if "ghz" in t:
        return "GHZ"
    if "w" == t or t.startswith("w_") or t.startswith("w-"):
        return "W"
    if "cluster" in t:
        return "Cluster"
    if "dicke" in t:
        return "Dicke-k2"
    return target


def main() -> None:
    """Entry point for the export_checkpoints CLI tool.

    Scans ``args.results_dir`` for QUASAR training result JSON files,
    pairs each with its companion ``.pt`` actor checkpoint, and writes
    a ``.npz`` weight file to ``args.output_dir`` for every checkpoint
    whose ``best_F`` meets the ``--min-fidelity`` threshold.

    Prints a progress line for each file processed and a summary at the end.
    """
    args = _parse_args()
    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all JSON result files
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    exported = 0
    skipped  = 0

    for jf in json_files:
        try:
            with open(jf) as f:
                result = json.load(f)
        except Exception as exc:
            print(f"  SKIP {jf.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        best_f   = result.get("best_F", result.get("best_fidelity", 0.0))
        target   = result.get("target", result.get("target_state", "unknown"))
        n_qubits = result.get("n_qubits", result.get("num_qubits", 0))
        seed     = result.get("seed", 42)
        family   = _family_from_target(str(target))

        if best_f < args.min_fidelity:
            print(f"  SKIP {jf.name}: best_F={best_f:.4f} < {args.min_fidelity}")
            skipped += 1
            continue

        # Look for the companion .pt file
        stem = jf.stem
        pt_candidates = [
            results_dir / f"{stem}.pt",
            results_dir / f"actor_{stem}.pt",
            results_dir / f"actor_{family}_{n_qubits}q_s{seed}.pt",
        ]
        pt_path: Optional[Path] = None
        for c in pt_candidates:
            if c.exists():
                pt_path = c
                break

        out_name = f"{family.replace('-','_')}_{n_qubits}q_s{seed}.npz"
        out_path = output_dir / out_name

        print(f"  {'[DRY]' if args.dry_run else '[EXPORT]'} "
              f"{jf.name} → {out_name}  (F={best_f:.4f})")

        if args.dry_run:
            exported += 1
            continue

        if pt_path is not None:
            weights = _load_actor_weights_from_pt(pt_path)
        else:
            weights = None

        if weights is None:
            print(f"    WARNING: no actor weights found for {jf.name}; "
                  f"skipping (no .pt file).", file=sys.stderr)
            skipped += 1
            continue

        np.savez(str(out_path), **weights)
        exported += 1

    print(f"\nDone. Exported: {exported}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
