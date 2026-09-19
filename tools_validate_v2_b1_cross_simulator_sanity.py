"""Validate the B1 fixed-circuit export design before execution."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_CATALOGUE_V1.json"
PROTOCOL = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_PROTOCOL_V1.md"
AUTHORITY = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_AUTHORITY_V1.md"
EXPORTER = ROOT / "tools_run_v2_b1_cross_simulator_sanity_export.py"
OUTPUT = DOCS / "V2_B1_CROSS_SIMULATOR_SANITY_READINESS_RECEIPT_V1.json"


def validate() -> dict[str, object]:
    catalogue = json.loads(CATALOGUE.read_text())
    protocol = PROTOCOL.read_text()
    authority = AUTHORITY.read_text()
    tree = ast.parse(EXPORTER.read_text())
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
    checks = {
        "catalogue_is_b1": catalogue["stage"] == "B1_FIXED_CIRCUIT_SANITY_EXPORT",
        "baseline_root_is_pinned": catalogue["baseline_root"] == "143cf5100504ab378f91c61f49c3e14b3901f04d",
        "nominal_profile_is_preserved": catalogue["profile"] == {
            "identifier": "simos-nominal-literature-v1",
            "calibration_status": "literature_parameterised",
        },
        "suite_is_fixed_and_native": [entry["id"] for entry in catalogue["fixed_circuits"]]
        == ["native_single_rx_pi_over_2", "native_two_qubit_cz_phase"],
        "authority_is_export_only": "FIXED_IDEAL_EXPORT_ONLY" in authority and "does **not** permit" in authority,
        "protocol_preserves_nonclaims": "not compare pulse physics, noise, fidelity, hardware behavior, QEC, learned control, or algorithms" in protocol,
        "exporter_has_no_prohibited_project_import": not ({"sivqd", "simora", "gymnasium", "torch", "stable_baselines3"} & imports),
        "forbidden_fields_are_declared": all(field in protocol.lower() for field in catalogue["forbidden_export_fields"]),
    }
    result = {
        "artifact_type": "SILIQUN_V2_B1_CROSS_SIMULATOR_SANITY_READINESS_RECEIPT",
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
