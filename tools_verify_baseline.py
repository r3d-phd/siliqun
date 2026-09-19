"""Emit a small machine-readable verification record for the V2 baseline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "BASELINE_VERIFICATION_RECEIPT.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True, check=False)
    required = [
        ROOT / "BASELINE_MANIFEST.json",
        ROOT / "README.md",
        ROOT / "ARCHITECTURE.md",
        ROOT / "docs" / "LEGACY_BOUNDARY.md",
        ROOT / "src" / "siliqun" / "circuit.py",
        ROOT / "src" / "siliqun" / "profiles.py",
        ROOT / "src" / "siliqun" / "pulse.py",
        ROOT / "src" / "siliqun" / "simulator.py",
    ]
    checks = {
        "required_files_present": all(path.is_file() and path.stat().st_size > 0 for path in required),
        "test_suite_passed": result.returncode == 0,
        "no_generated_legacy_tree": not (ROOT / "experiments").exists() and not (ROOT / "results").exists(),
    }
    receipt = {
        "artifact_type": "SILIQUN_V2_STANDALONE_BASELINE_VERIFICATION_RECEIPT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "test_returncode": result.returncode,
        "test_summary": result.stdout[-2000:] + result.stderr[-2000:],
        "file_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_boundary": "Software-boundary verification only; no physics, calibration, hardware, or QEC conclusion.",
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
