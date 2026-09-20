"""Validate the B3 fixed algorithm-representation export design before execution."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_CATALOGUE_V1.json"
PROTOCOL = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_PROTOCOL_V1.md"
AUTHORITY = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_AUTHORITY_V1.md"
EXPORTER = ROOT / "tools_run_v2_b3_algorithm_representation_export.py"
STATIC = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_STATIC_SPECIFICATION_RECEIPT_V1.json"
OUTPUT = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_READINESS_RECEIPT_V1.json"

EXPECTED_IDS = [
    "deutsch_jozsa_two_qubit_balanced_native",
    "grover_two_qubit_fixed_phase_oracle_native",
    "shor_n15_order4_compiled_orbit_component_native",
]


def _imports(tree: ast.AST) -> set[str]:
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    return names


def validate() -> dict[str, object]:
    catalogue = json.loads(CATALOGUE.read_text())
    protocol = PROTOCOL.read_text()
    authority = AUTHORITY.read_text()
    static = json.loads(STATIC.read_text())
    imports = _imports(ast.parse(EXPORTER.read_text()))
    rows = static["rows"]
    checks = {
        "catalogue_is_b3": catalogue["stage"] == "B3_PRESPECIFIED_ALGORITHM_REPRESENTATION_EXPORT",
        "baseline_root_is_pinned": catalogue["baseline_root"] == "143cf5100504ab378f91c61f49c3e14b3901f04d",
        "nominal_profile_is_preserved": catalogue["profile"] == {"identifier": "simos-nominal-literature-v1", "calibration_status": "literature_parameterised"},
        "three_representations_are_fixed": [entry["id"] for entry in catalogue["fixed_representations"]] == EXPECTED_IDS and [row["representation_id"] for row in rows] == EXPECTED_IDS and len(rows) == 3,
        "static_specifications_are_native_and_bounded": all(set(row["native_gate_names"]) <= set(catalogue["native_gate_set"]) and row["n_qubits"] <= 3 and row["gate_count"] > 0 and len(row["circuit_specification_sha256"]) == 64 and row["circuit_specification"]["n_qubits"] == row["n_qubits"] and len(row["circuit_specification"]["gates"]) == row["gate_count"] for row in rows),
        "authority_is_export_only": "FIXED_IDEAL_ALGORITHM_REPRESENTATION_EXPORT_ONLY" in authority and "does **not** permit" in authority,
        "protocol_preserves_algorithm_nonclaims": "neither runs an algorithm end-to-end" in protocol and "cannot establish" in protocol and "factorization" in protocol,
        "release_validation_is_required": "release-validation test" in protocol,
        "exporter_has_no_prohibited_project_import": not ({"sivqd", "simora", "gymnasium", "torch", "stable_baselines3"} & imports),
        "forbidden_fields_are_declared": all(field in protocol.lower() for field in catalogue["forbidden_export_fields"]),
    }
    result = {"artifact_type": "SILIQUN_V2_B3_ALGORITHM_REPRESENTATION_READINESS_RECEIPT", "version": "V1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "claim_ceiling": catalogue["claim_ceiling"]}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    static = build_static = None
    from tools_run_v2_b3_algorithm_representation_export import build_static_specification
    STATIC.write_text(json.dumps(build_static_specification(), indent=2, sort_keys=True) + "\n")
    result = validate()
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
