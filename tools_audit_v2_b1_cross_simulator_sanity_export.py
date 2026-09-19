"""Source-distinct audit for the bounded SiliQun V2 B1 source export."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_CATALOGUE_V1.json"
READINESS = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_READINESS_RECEIPT_V1.json"
EXPORT = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_EXPORT_RECEIPT_V1.json"
EXPORTER = ROOT / "tools_run_v2_b1_cross_simulator_sanity_export.py"
OUTPUT = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_SOURCE_AUDIT_RECEIPT_V1.json"


def _exporter_hash_at_export_commit(commit: str) -> str:
    blob = subprocess.run(
        ["git", "show", f"{commit}:{EXPORTER.name}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def audit() -> dict[str, object]:
    catalogue = json.loads(CATALOGUE.read_text())
    readiness = json.loads(READINESS.read_text())
    exported = json.loads(EXPORT.read_text())
    source = EXPORTER.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    rows = exported["suite"]
    checks = {
        "readiness_passed": readiness["status"] == "PASS" and all(readiness["checks"].values()),
        "source_commit_binds_exporter_hash": _exporter_hash_at_export_commit(exported["source"]["source_commit"])
        == exported["source"]["exporter_sha256"],
        "baseline_root_matches": exported["source"]["baseline_root"] == catalogue["baseline_root"],
        "nominal_profile_status_preserved": exported["profile"] == {
            "identifier": "simos-nominal-literature-v1",
            "calibration_status": "literature_parameterised",
        },
        "fixed_suite_identity_matches": [row["circuit_id"] for row in rows]
        == [entry["id"] for entry in catalogue["fixed_circuits"]],
        "operation_categories_match_fixed_suite": [row["operation_categories"] for row in rows]
        == [
            [{"kind": "drive", "target_arity": 1}],
            [
                {"kind": "drive", "target_arity": 1},
                {"kind": "drive", "target_arity": 1},
                {"kind": "exchange", "target_arity": 2},
                {"kind": "virtual_z", "target_arity": 1},
            ],
        ],
        "probability_digests_are_present": all(
            isinstance(row.get("siliqun_ideal_probability_sha256"), str)
            and len(row["siliqun_ideal_probability_sha256"]) == 64
            for row in rows
        ),
        "forbidden_values_not_retained": all(value is False for value in exported["retention"].values()),
        "runtime_scope_fields_are_false": all(value is False for value in exported["scope"].values()),
        "exporter_has_no_downstream_import": not ({"sivqd", "simora", "gymnasium", "torch", "stable_baselines3"} & imports),
    }
    result = {
        "artifact_type": "SILIQUN_V2_B1_CROSS_SIMULATOR_SANITY_SOURCE_AUDIT_RECEIPT",
        "version": "V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "audit_scope": "Source-export, fixed-suite, and retained-field audit only; no independent SiVQD evaluator, cross-simulator agreement, physical, calibration, SiMORA, algorithm, QEC, or hardware audit.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = audit()
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
