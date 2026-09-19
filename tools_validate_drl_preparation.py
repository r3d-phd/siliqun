"""Emit a machine-readable readiness receipt for V2 DRL preparation only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from siliqun import Circuit, simos_nominal_profile
from siliqun.contracts import (
    DESIGN_ONLY_PROHIBITED_OPERATIONS,
    SafePPOCalibrationPluginSpec,
    TranslationRequest,
    prohibit_runtime_operation,
    translate,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "V2_DRL_PREPARATION_READINESS_RECEIPT_V1.json"
CATALOGUE = ROOT / "docs" / "V2_DRL_PREPARATION_CATALOGUE_V1.json"
CONTRACT = ROOT / "docs" / "V2_PROFILE_CIRCUIT_TRANSLATION_CONTRACT_V1.md"
PROTOCOL = ROOT / "docs" / "V2_SAFE_PPO_CALIBRATION_PLUGIN_PROTOCOL_V1.md"
AUTHORITY = ROOT / "docs" / "V2_DRL_PREPARATION_AUTHORITY_V1.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    catalogue = json.loads(CATALOGUE.read_text())
    request = TranslationRequest(Circuit(2).rz(0.25, 0).cz(0, 1), simos_nominal_profile())
    receipt = translate(request)
    plugin = SafePPOCalibrationPluginSpec()
    plugin.validate()
    runtime_refusals = []
    for operation in sorted(DESIGN_ONLY_PROHIBITED_OPERATIONS):
        try:
            prohibit_runtime_operation(operation)
        except RuntimeError:
            runtime_refusals.append(operation)
    text = "\n".join(path.read_text() for path in (CONTRACT, PROTOCOL, AUTHORITY))
    checks = {
        "authority_is_design_only": "DESIGN_AND_STATIC_VALIDATION_ONLY" in AUTHORITY.read_text(),
        "catalogue_scope_is_design_only": catalogue["status"] == "design_and_static_validation_only",
        "translation_receipt_is_non_executable": "not a device command" in receipt.claim_ceiling,
        "nominal_profile_remains_literature_parameterised": receipt.request.profile.calibration_status == "literature_parameterised",
        "plugin_is_ppo_only_and_command_disabled": plugin.algorithm == "PPO" and plugin.command_mode == "disabled",
        "plugin_is_limited_to_five_qubits": plugin.maximum_qubits == 5,
        "every_prohibited_runtime_operation_refuses": runtime_refusals == sorted(DESIGN_ONLY_PROHIBITED_OPERATIONS),
        "required_comparators_are_declared": len(plugin.comparators) == 3,
        "te_pws_is_reserved_only": plugin.te_pws_status == "reserved_not_implemented",
        "no_training_or_device_access_in_authority": all(
            phrase in AUTHORITY.read_text()
            for phrase in ("does **not** permit PPO training", "hardware job submission", "provider API access")
        ),
        "scope_exclusions_are_documented": all(
            phrase in text
            for phrase in ("SiMORA", "No policy is trained", "no hardware command")
        ),
    }
    result = {
        "artifact_type": "SILIQUN_V2_DRL_PREPARATION_READINESS_RECEIPT",
        "version": "V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "translation_receipt": receipt.to_manifest(),
        "plugin_spec": plugin.to_manifest(),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (CATALOGUE, CONTRACT, PROTOCOL, AUTHORITY)
        },
        "runtime_environment": {
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "execution_mode": "local_static_validation",
        },
        "claim_ceiling": (
            "Static design validation only; no PPO training, named-device calibration, "
            "device access, hardware command, SiMORA policy change, QEC result, or performance result."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
