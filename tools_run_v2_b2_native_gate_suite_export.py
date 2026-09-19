"""Export five fixed SiliQun V2 ideal native-gate digests for B2.

The output is intentionally non-executable and retains no state vector,
probability vector, pulse parameter, fidelity, device, or controller input.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from siliqun import Circuit, StateVectorSimulator
from siliqun.contracts import TranslationRequest, translate
from siliqun.profiles import simos_nominal_profile


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CATALOGUE = DOCS / "V2_B2_NATIVE_GATE_SUITE_CATALOGUE_V1.json"
AUTHORITY = DOCS / "V2_B2_NATIVE_GATE_SUITE_AUTHORITY_V1.md"
OUTPUT = DOCS / "V2_B2_NATIVE_GATE_SUITE_EXPORT_RECEIPT_V1.json"
BASELINE_ROOT = "143cf5100504ab378f91c61f49c3e14b3901f04d"


def _canonical_probability_digest(probabilities: Any) -> str:
    values = [round(float(value), 15) for value in probabilities]
    encoded = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def fixed_circuits() -> tuple[tuple[str, Circuit], ...]:
    return (
        ("native_single_rx_pi_over_3", Circuit(1).rx(math.pi / 3, 0)),
        (
            "native_single_ry_minus_pi_over_4_rz_pi_over_5",
            Circuit(1).ry(-math.pi / 4, 0).rz(math.pi / 5, 0),
        ),
        (
            "native_single_rx_pi_over_2_rz_pi_over_2",
            Circuit(1).rx(math.pi / 2, 0).rz(math.pi / 2, 0),
        ),
        (
            "native_two_qubit_ordered_cz_interference",
            Circuit(2)
            .ry(math.pi / 3, 0)
            .rx(math.pi / 5, 1)
            .cz(0, 1)
            .rz(math.pi / 7, 0)
            .ry(math.pi / 4, 0)
            .rx(math.pi / 6, 1),
        ),
        ("native_single_rz_two_pi_probability_invariant", Circuit(1).rz(2 * math.pi, 0)),
    )


def _circuit_spec(circuit: Circuit) -> dict[str, Any]:
    return {
        "n_qubits": circuit.n_qubits,
        "gates": [
            {"name": gate.name, "targets": list(gate.targets), "parameters": list(gate.parameters)}
            for gate in circuit.gates
        ],
    }


def _operation_categories(receipt: Any) -> list[dict[str, int | str]]:
    return [{"kind": pulse.kind, "target_arity": len(pulse.targets)} for pulse in receipt.pulse_schedule.pulses]


def build_export() -> dict[str, Any]:
    catalogue = json.loads(CATALOGUE.read_text())
    profile = simos_nominal_profile()
    rows: list[dict[str, Any]] = []
    for circuit_id, circuit in fixed_circuits():
        translation = translate(TranslationRequest(circuit=circuit, profile=profile))
        result = StateVectorSimulator(circuit.n_qubits).run(circuit)
        manifest = translation.to_manifest()
        rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_specification": _circuit_spec(circuit),
                "circuit_sha256": translation.circuit_sha256,
                "translation_manifest_sha256": _canonical_sha256(manifest),
                "pulse_schedule_sha256": translation.pulse_schedule_sha256,
                "operation_categories": _operation_categories(translation),
                "siliqun_ideal_probability_sha256": _canonical_probability_digest(result.probabilities),
            }
        )
        del result, translation, manifest
    return {
        "artifact_type": "SILIQUN_V2_B2_NATIVE_GATE_SUITE_EXPORT_RECEIPT",
        "version": "V1",
        "status": "PASS",
        "source": {
            "repository": "r3d-phd/siliqun",
            "source_commit": _source_head(),
            "baseline_root": BASELINE_ROOT,
            "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "profile": {
            "identifier": profile.identifier,
            "calibration_status": profile.calibration_status,
        },
        "suite": rows,
        "suite_count": len(rows),
        "retention": {
            "state_vector_retained": False,
            "probability_vector_retained": False,
            "pulse_amplitude_retained": False,
            "pulse_phase_retained": False,
            "pulse_duration_retained": False,
            "fidelity_retained": False,
            "noise_draw_retained": False,
            "reward_retained": False,
            "target_state_retained": False,
            "measurement_result_retained": False,
            "logical_observable_retained": False,
            "algorithm_answer_retained": False,
        },
        "scope": {
            "simora_imported_or_invoked": False,
            "detector_or_decoder_bound": False,
            "frame_action_issued": False,
            "algorithm_ingress_established": False,
            "provider_or_hardware_accessed": False,
            "ppo_used": False,
        },
        "claim_ceiling": catalogue["claim_ceiling"],
    }


def main() -> None:
    if "FIXED_IDEAL_EXPORT_ONLY" not in AUTHORITY.read_text():
        raise SystemExit("b2_export_refused_authority")
    OUTPUT.write_text(json.dumps(build_export(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
