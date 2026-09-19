"""Validate that the D1 report matches the frozen public-evidence eligibility decision."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CATALOGUE = ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_CATALOGUE_V1.json"
PROTOCOL = ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_PROTOCOL_V1.md"
DISPOSITION = ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_DISPOSITION_V1.md"
SOURCE_NOTES = ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_SOURCE_NOTES_V1.md"
REVIEW = ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_VERITAS_REVIEW_RECORD_V1.json"
OUTPUT = ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_VALIDATION_RECEIPT_V1.json"


def main() -> None:
    catalogue = json.loads(CATALOGUE.read_text())
    protocol = PROTOCOL.read_text()
    disposition = DISPOSITION.read_text()
    source_notes = SOURCE_NOTES.read_text()
    review = json.loads(REVIEW.read_text())
    candidate_statuses = [candidate["strict_d1_status"] for candidate in catalogue["candidates"]]
    checks = {
        "decision_refuses_d1": catalogue["decision"]["status"] == "D1_REFUSED_METADATA_AUTHORIZATION_ABSENT",
        "every_candidate_refused": candidate_statuses == ["REFUSED", "REFUSED"],
        "preferred_target_is_conditional": catalogue["decision"]["preferred_conditional_next_target"] == "intel_tunnel_falls_lqc_qcf",
        "protocol_preserves_no_hardware_action": all(
            phrase in protocol
            for phrase in ("did not create an account", "log into a provider", "submit a job", "hardware command")
        ),
        "disposition_preserves_claim_ceiling": all(
            phrase in disposition
            for phrase in ("not a D1 pass", "does not permit PPO training", "does not permit PPO training")
        ) and "realistic SiMOS noise" in disposition,
        "source_notes_include_both_candidates": "Quantum Motion" in source_notes and "Intel Tunnel Falls" in source_notes,
        "factual_review_is_verified": review["factual_claim_review"]["verdict"] == "VERIFIED" and review["factual_claim_review"]["confidence_percent"] == 90,
        "reasoning_limit_is_preserved": review["reasoning_chain_review"]["result"] == "NO_VERDICT" and "NO REASONING VERDICT" in disposition,
        "sources_are_public_url_cited": all(
            url in source_notes
            for url in (
                "https://www.intel.com/",
                "https://quantummotion.com/",
                "https://www.nqcc.ac.uk/",
                "https://www.qubitcollaboratory.org/",
            )
        ),
    }
    result = {
        "artifact_type": "SILIQUN_V2_D1_SILICON_ELIGIBILITY_VALIDATION_RECEIPT",
        "version": "V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "claim_ceiling": "Public-evidence D1 eligibility assessment only; no named-device authorization, calibration record, PPO training, SiMORA runtime, or hardware claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
