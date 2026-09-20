"""Source-distinct audit for the bounded SiliQun V2 B2 source export."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B2_NATIVE_GATE_SUITE_CATALOGUE_V1.json"
READINESS = DOCS / "V2_B2_NATIVE_GATE_SUITE_READINESS_RECEIPT_V1.json"
EXPORT = DOCS / "V2_B2_NATIVE_GATE_SUITE_EXPORT_RECEIPT_V1.json"
EXPORTER = ROOT / "tools_run_v2_b2_native_gate_suite_export.py"
OUTPUT = DOCS / "V2_B2_NATIVE_GATE_SUITE_SOURCE_AUDIT_RECEIPT_V1.json"


EXPECTED_OPERATION_CATEGORIES = [
    [{"kind": "drive", "target_arity": 1}],
    [{"kind": "drive", "target_arity": 1}, {"kind": "virtual_z", "target_arity": 1}],
    [{"kind": "drive", "target_arity": 1}, {"kind": "virtual_z", "target_arity": 1}],
    [
        {"kind": "drive", "target_arity": 1},
        {"kind": "drive", "target_arity": 1},
        {"kind": "exchange", "target_arity": 2},
        {"kind": "virtual_z", "target_arity": 1},
        {"kind": "drive", "target_arity": 1},
        {"kind": "drive", "target_arity": 1},
    ],
    [{"kind": "virtual_z", "target_arity": 1}],
]

EXPECTED_CIRCUIT_SPECIFICATIONS = [
    {
        "n_qubits": 1,
        "gates": [{"name": "rx", "targets": [0], "parameters": [math.pi / 3]}],
    },
    {
        "n_qubits": 1,
        "gates": [
            {"name": "ry", "targets": [0], "parameters": [-math.pi / 4]},
            {"name": "rz", "targets": [0], "parameters": [math.pi / 5]},
        ],
    },
    {
        "n_qubits": 1,
        "gates": [
            {"name": "rx", "targets": [0], "parameters": [math.pi / 2]},
            {"name": "rz", "targets": [0], "parameters": [math.pi / 2]},
        ],
    },
    {
        "n_qubits": 2,
        "gates": [
            {"name": "ry", "targets": [0], "parameters": [math.pi / 3]},
            {"name": "rx", "targets": [1], "parameters": [math.pi / 5]},
            {"name": "cz", "targets": [0, 1], "parameters": []},
            {"name": "rz", "targets": [0], "parameters": [math.pi / 7]},
            {"name": "ry", "targets": [0], "parameters": [math.pi / 4]},
            {"name": "rx", "targets": [1], "parameters": [math.pi / 6]},
        ],
    },
    {
        "n_qubits": 1,
        "gates": [{"name": "rz", "targets": [0], "parameters": [2 * math.pi]}],
    },
]


def _exporter_hash_at_export_commit(commit: str) -> str:
    blob = subprocess.run(
        ["git", "show", f"{commit}:{EXPORTER.name}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def _imports(tree: ast.AST) -> set[str]:
    names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return names


def audit() -> dict[str, object]:
    catalogue = json.loads(CATALOGUE.read_text())
    readiness = json.loads(READINESS.read_text())
    exported = json.loads(EXPORT.read_text())
    imports = _imports(ast.parse(EXPORTER.read_text()))
    rows = exported["suite"]
    checks = {
        "readiness_passed": readiness["status"] == "PASS" and all(readiness["checks"].values()),
        "source_commit_binds_exporter_hash": _exporter_hash_at_export_commit(exported["source"]["source_commit"])
        == exported["source"]["exporter_sha256"],
        "baseline_root_matches": exported["source"]["baseline_root"] == catalogue["baseline_root"],
        "nominal_profile_status_preserved": exported["profile"] == catalogue["profile"],
        "fixed_suite_identity_matches": [row["circuit_id"] for row in rows]
        == [entry["id"] for entry in catalogue["fixed_circuits"]],
        "fixed_circuit_specifications_match": [row["circuit_specification"] for row in rows]
        == EXPECTED_CIRCUIT_SPECIFICATIONS,
        "operation_categories_match_fixed_suite": [row["operation_categories"] for row in rows]
        == EXPECTED_OPERATION_CATEGORIES,
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
        "artifact_type": "SILIQUN_V2_B2_NATIVE_GATE_SUITE_SOURCE_AUDIT_RECEIPT",
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
