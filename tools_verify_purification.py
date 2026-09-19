"""Emit evidence that the standalone baseline is a purified orphan snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY_COMMIT = "14a49bf957b7aeacffc1bd15ba91293d6fd31b83"
OUTPUT = ROOT / "BASELINE_PURIFICATION_RECEIPT.json"


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def forbidden_markers() -> tuple[str, ...]:
    return tuple("".join(parts) for parts in (("gym", "nasium"), ("stable", "_baselines"), ("andro", "meda"), ("qua", "sar"), ("simo", "ra")))


def current_paths() -> set[str]:
    return {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.name != OUTPUT.name
    }


def legacy_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ("git", "show", f"{LEGACY_COMMIT}:{path}"), cwd=ROOT
    )


def main() -> None:
    legacy_paths = set(command("git", "ls-tree", "-r", "--name-only", LEGACY_COMMIT).splitlines())
    current = current_paths()
    shared_paths = sorted(
        path
        for path in legacy_paths & current
        if (ROOT / path).read_bytes() == legacy_bytes(path)
    )
    current_source = "\n".join(
        path.read_text(errors="ignore")
        for path in (ROOT / "src").rglob("*.py")
    ).lower()
    absence_directories = tuple("".join(parts) for parts in (("exper", "iments"), ("res", "ults"), ("alpha", "evolve")))
    baseline_root = command("git", "rev-list", "--max-parents=0", "HEAD")
    parent_fields = command("git", "rev-list", "--parents", "-n", "1", baseline_root).split()
    checks = {
        "legacy_commit_matches_manifest": json.loads((ROOT / "BASELINE_MANIFEST.json").read_text())["legacy_source"]["commit"] == LEGACY_COMMIT,
        "baseline_root_commit_is_orphan": len(parent_fields) == 1,
        "only_license_path_is_shared_with_legacy_tree": shared_paths == ["LICENSE"],
        "excluded_application_markers_absent_from_package": all(marker not in current_source for marker in forbidden_markers()),
        "legacy_artifact_directories_absent": all(not (ROOT / directory).exists() for directory in absence_directories),
    }
    receipt = {
        "artifact_type": "SILIQUN_V2_STANDALONE_BASELINE_PURIFICATION_RECEIPT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "baseline_root_commit": baseline_root,
        "legacy_commit": LEGACY_COMMIT,
        "legacy_tree_file_count": len(legacy_paths),
        "baseline_file_count": len(current),
        "shared_legacy_paths": shared_paths,
        "checks": checks,
        "claim_boundary": "Source-purification verification only; no inference about physics, calibration, hardware, QEC, or performance.",
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
