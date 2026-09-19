"""Check that the D0 disposition preserves the static-readiness evidence ceiling."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / "docs" / "V2_DRL_PREPARATION_READINESS_RECEIPT_V1.json"
REVIEW = ROOT / "docs" / "V2_DRL_PREPARATION_VERITAS_REVIEW_RECORD_V1.json"
DISPOSITION = ROOT / "docs" / "V2_DRL_PREPARATION_DISPOSITION_V1.md"
OUTPUT = ROOT / "docs" / "V2_DRL_PREPARATION_DISPOSITION_VALIDATION_RECEIPT_V1.json"


def main() -> None:
    receipt = json.loads(RECEIPT.read_text())
    review = json.loads(REVIEW.read_text())
    report = DISPOSITION.read_text()
    checks = {
        "readiness_passes_all_declared_checks": receipt["status"] == "PASS" and len(receipt["checks"]) == 11 and all(receipt["checks"].values()),
        "review_records_final_factual_verification": review["factual_scope_review"]["final_result"]["verdict"] == "VERIFIED",
        "review_preserves_reasoning_limit": review["review_disposition"].startswith("The factual D0 claim is VERIFIED"),
        "report_states_d0_only": "D0 static preparation completed" in report,
        "report_preserves_nonclaims": all(
            phrase in report
            for phrase in ("does not train PPO", "not a named-device calibration package", "not PPO training")
        ),
        "report_preserves_next_gate": "read-only D1 eligibility review" in report,
        "report_defines_supervisory_boundary": "SiMORA deterministic supervisor" in report,
    }
    result = {
        "artifact_type": "SILIQUN_V2_DRL_PREPARATION_DISPOSITION_VALIDATION_RECEIPT",
        "version": "V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "claim_ceiling": "Receipt-bound D0 reporting only; no PPO, calibration, hardware, or runtime-performance conclusion.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
