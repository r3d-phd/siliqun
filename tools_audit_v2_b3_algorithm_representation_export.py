"""Source-distinct audit for the bounded SiliQun V2 B3 representation export."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_CATALOGUE_V1.json"
READINESS = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_READINESS_RECEIPT_V1.json"
STATIC = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_STATIC_SPECIFICATION_RECEIPT_V1.json"
EXPORT = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_EXPORT_RECEIPT_V1.json"
EXPORTER = ROOT / "tools_run_v2_b3_algorithm_representation_export.py"
OUTPUT = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_SOURCE_AUDIT_RECEIPT_V1.json"


def _exporter_hash_at_export_commit(commit: str) -> str:
    blob = subprocess.run(["git", "show", f"{commit}:{EXPORTER.name}"], cwd=ROOT, check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def _imports(tree: ast.AST) -> set[str]:
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    return names


def _expected_categories(specification: dict[str, object]) -> list[dict[str, object]]:
    mapping = {"rx": "drive", "ry": "drive", "rz": "virtual_z", "cz": "exchange"}
    return [{"kind": mapping[gate["name"]], "target_arity": len(gate["targets"])} for gate in specification["gates"]]


def audit() -> dict[str, object]:
    catalogue = json.loads(CATALOGUE.read_text())
    readiness = json.loads(READINESS.read_text())
    static = json.loads(STATIC.read_text())
    exported = json.loads(EXPORT.read_text())
    imports = _imports(ast.parse(EXPORTER.read_text()))
    rows = exported["suite"]
    static_by_id = {row["representation_id"]: row for row in static["rows"]}
    expected_ids = [entry["id"] for entry in catalogue["fixed_representations"]]
    checks = {
        "readiness_passed": readiness["status"] == "PASS" and all(readiness["checks"].values()),
        "source_commit_binds_exporter_hash": _exporter_hash_at_export_commit(exported["source"]["source_commit"]) == exported["source"]["exporter_sha256"],
        "baseline_root_matches": exported["source"]["baseline_root"] == catalogue["baseline_root"],
        "nominal_profile_status_preserved": exported["profile"] == catalogue["profile"],
        "fixed_representation_identity_matches": [row["representation_id"] for row in rows] == expected_ids,
        "exact_specification_hashes_match_static_receipt": all(row["representation_id"] in static_by_id and row["circuit_specification_sha256"] == row["static_specification_sha256"] == static_by_id[row["representation_id"]]["circuit_specification_sha256"] for row in rows),
        "native_gate_set_and_categories_match": all(set(gate["name"] for gate in row["circuit_specification"]["gates"]) <= set(catalogue["native_gate_set"]) and row["operation_categories"] == _expected_categories(row["circuit_specification"]) for row in rows),
        "bounded_qubit_counts_match": [row["circuit_specification"]["n_qubits"] for row in rows] == [2, 2, 3],
        "probability_digests_are_present": all(isinstance(row.get("siliqun_ideal_probability_sha256"), str) and len(row["siliqun_ideal_probability_sha256"]) == 64 for row in rows),
        "forbidden_values_not_retained": all(value is False for value in exported["retention"].values()),
        "runtime_scope_fields_are_false": all(value is False for value in exported["scope"].values()),
        "exporter_has_no_downstream_import": not ({"sivqd", "simora", "gymnasium", "torch", "stable_baselines3"} & imports),
    }
    result = {"artifact_type": "SILIQUN_V2_B3_ALGORITHM_REPRESENTATION_SOURCE_AUDIT_RECEIPT", "version": "V1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "audit_scope": "Source-export, static-representation, native-compilation, provenance, and retained-field audit only; no independent SiVQD evaluator, algorithm success, factorization, physical, calibration, SiMORA, QEC, or hardware audit."}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = audit()
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
