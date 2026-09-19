"""Validate the B2 fixed native-gate export design before execution."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B2_NATIVE_GATE_SUITE_CATALOGUE_V1.json"
PROTOCOL = DOCS / "V2_B2_NATIVE_GATE_SUITE_PROTOCOL_V1.md"
AUTHORITY = DOCS / "V2_B2_NATIVE_GATE_SUITE_AUTHORITY_V1.md"
EXPORTER = ROOT / "tools_run_v2_b2_native_gate_suite_export.py"
OUTPUT = DOCS / "V2_B2_NATIVE_GATE_SUITE_READINESS_RECEIPT_V1.json"


EXPECTED_IDS = [
    "native_single_rx_pi_over_3",
    "native_single_ry_minus_pi_over_4_rz_pi_over_5",
    "native_single_rx_pi_over_2_rz_pi_over_2",
    "native_two_qubit_ordered_cz_interference",
    "native_single_rz_two_pi_probability_invariant",
]
EXPECTED_SEQUENCES = [["rx"], ["ry", "rz"], ["rx", "rz"], ["ry", "rx", "cz", "rz", "ry", "rx"], ["rz"]]


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


def validate() -> dict[str, object]:
    catalogue = json.loads(CATALOGUE.read_text())
    protocol = PROTOCOL.read_text()
    authority = AUTHORITY.read_text()
    imports = _imports(ast.parse(EXPORTER.read_text()))
    circuits = catalogue["fixed_circuits"]
    checks = {
        "catalogue_is_b2": catalogue["stage"] == "B2_PRESPECIFIED_NATIVE_GATE_SUITE_EXPORT",
        "baseline_root_is_pinned": catalogue["baseline_root"] == "143cf5100504ab378f91c61f49c3e14b3901f04d",
        "nominal_profile_is_preserved": catalogue["profile"] == {
            "identifier": "simos-nominal-literature-v1",
            "calibration_status": "literature_parameterised",
        },
        "five_circuits_are_fixed_and_native": [entry["id"] for entry in circuits] == EXPECTED_IDS
        and [entry["gate_sequence"] for entry in circuits] == EXPECTED_SEQUENCES
        and all(gate in {"rx", "ry", "rz", "cz"} for entry in circuits for gate in entry["gate_sequence"]),
        "coverage_rationale_is_declared": all(
            phrase in protocol
            for phrase in ("negative angle", "global-phase invariant", "least-significant-bit")
        ),
        "authority_is_export_only": "FIXED_IDEAL_EXPORT_ONLY" in authority and "does **not** permit" in authority,
        "protocol_preserves_nonclaims": "does not compare pulse physics, noise, fidelity, hardware behavior, QEC, learned control, or algorithms" in protocol,
        "release_validation_is_required": "separately invoked release-validation test" in protocol,
        "exporter_has_no_prohibited_project_import": not ({"sivqd", "simora", "gymnasium", "torch", "stable_baselines3"} & imports),
        "forbidden_fields_are_declared": all(field in protocol.lower() for field in catalogue["forbidden_export_fields"]),
    }
    result = {
        "artifact_type": "SILIQUN_V2_B2_NATIVE_GATE_SUITE_READINESS_RECEIPT",
        "version": "V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "claim_ceiling": catalogue["claim_ceiling"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = validate()
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
