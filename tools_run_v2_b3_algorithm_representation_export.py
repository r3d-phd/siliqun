"""Export three fixed ideal algorithm-associated circuit representations for B3.

The output retains digest-only circuit-convention evidence. It is not algorithm success,
factorization, hardware, calibration, physical, QEC, or SiMORA evidence.
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
CATALOGUE = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_CATALOGUE_V1.json"
AUTHORITY = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_AUTHORITY_V2.md"
OUTPUT = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_EXPORT_RECEIPT_V1.json"
STATIC_SPECIFICATION = DOCS / "V2_B3_ALGORITHM_REPRESENTATION_STATIC_SPECIFICATION_RECEIPT_V1.json"
BASELINE_ROOT = "143cf5100504ab378f91c61f49c3e14b3901f04d"


def _canonical_probability_digest(probabilities: Any) -> str:
    values = [round(float(value), 15) for value in probabilities]
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def _native_h(circuit: Circuit, target: int) -> Circuit:
    return circuit.rz(math.pi, target).ry(math.pi / 2, target)


def _native_x(circuit: Circuit, target: int) -> Circuit:
    return circuit.rx(math.pi, target)


def _native_t(circuit: Circuit, target: int) -> Circuit:
    return circuit.rz(math.pi / 4, target)


def _native_tdg(circuit: Circuit, target: int) -> Circuit:
    return circuit.rz(-math.pi / 4, target)


def _native_cx(circuit: Circuit, control: int, target: int) -> Circuit:
    return _native_h(circuit, target).cz(control, target).rz(math.pi, target).ry(math.pi / 2, target)


def _native_ccx(circuit: Circuit, control_a: int, control_b: int, target: int) -> Circuit:
    _native_h(circuit, target)
    _native_cx(circuit, control_b, target)
    _native_tdg(circuit, target)
    _native_cx(circuit, control_a, target)
    _native_t(circuit, target)
    _native_cx(circuit, control_b, target)
    _native_tdg(circuit, target)
    _native_cx(circuit, control_a, target)
    _native_t(circuit, control_b)
    _native_t(circuit, target)
    _native_h(circuit, target)
    _native_cx(circuit, control_a, control_b)
    _native_t(circuit, control_a)
    _native_tdg(circuit, control_b)
    return _native_cx(circuit, control_a, control_b)


def _deutsch_jozsa_representation() -> Circuit:
    circuit = Circuit(2)
    _native_x(circuit, 1)
    _native_h(circuit, 0)
    _native_h(circuit, 1)
    _native_cx(circuit, 0, 1)
    return _native_h(circuit, 0)


def _grover_representation() -> Circuit:
    circuit = Circuit(2)
    _native_h(circuit, 0)
    _native_h(circuit, 1)
    circuit.cz(0, 1)
    _native_h(circuit, 0)
    _native_h(circuit, 1)
    _native_x(circuit, 0)
    _native_x(circuit, 1)
    circuit.cz(0, 1)
    _native_x(circuit, 0)
    _native_x(circuit, 1)
    _native_h(circuit, 0)
    return _native_h(circuit, 1)


def _shor_order4_work_orbit_transition() -> Circuit:
    """Compile one uncontrolled order-four work-orbit transition, not Shor.

    q0 and q1 encode the restricted work orbit 1,2,4,8 as 00,01,10,11.
    The native CNOT-equivalent q0 -> q1 followed by X-equivalent on q0
    advances this encoding modulo four, corresponding to multiplication by
    two on that restricted orbit. There is no control register, QFT,
    measurement, post-processing, order, or factoring answer.
    """
    work_lsb, work_msb = 0, 1
    circuit = Circuit(2)
    _native_cx(circuit, work_lsb, work_msb)
    return _native_x(circuit, work_lsb)


def fixed_representations() -> tuple[tuple[str, Circuit], ...]:
    return (
        ("deutsch_jozsa_two_qubit_balanced_native", _deutsch_jozsa_representation()),
        ("grover_two_qubit_fixed_phase_oracle_native", _grover_representation()),
        ("shor_n15_order4_work_orbit_transition_native", _shor_order4_work_orbit_transition()),
    )


def _circuit_spec(circuit: Circuit) -> dict[str, Any]:
    return {"n_qubits": circuit.n_qubits, "gates": [{"name": gate.name, "targets": list(gate.targets), "parameters": list(gate.parameters)} for gate in circuit.gates]}


def _operation_categories(receipt: Any) -> list[dict[str, int | str]]:
    return [{"kind": pulse.kind, "target_arity": len(pulse.targets)} for pulse in receipt.pulse_schedule.pulses]


def build_static_specification() -> dict[str, Any]:
    rows = []
    for representation_id, circuit in fixed_representations():
        specification = _circuit_spec(circuit)
        rows.append({
            "representation_id": representation_id,
            "n_qubits": circuit.n_qubits,
            "gate_count": len(circuit.gates),
            "native_gate_names": sorted({gate["name"] for gate in specification["gates"]}),
            "circuit_specification": specification,
            "circuit_specification_sha256": _canonical_sha256(specification),
        })
    return {
        "artifact_type": "SILIQUN_V2_B3_ALGORITHM_REPRESENTATION_STATIC_SPECIFICATION_RECEIPT",
        "version": "V1",
        "status": "PASS",
        "rows": rows,
        "row_count": len(rows),
        "static_scope": "Pre-execution circuit structure and SHA-256 specification hashes only; no simulator state, probability, translation, pulse, hardware, algorithm answer, or physical claim.",
    }


def build_export() -> dict[str, Any]:
    catalogue = json.loads(CATALOGUE.read_text())
    profile = simos_nominal_profile()
    static_specification = json.loads(STATIC_SPECIFICATION.read_text())
    static_by_id = {row["representation_id"]: row for row in static_specification["rows"]}
    rows: list[dict[str, Any]] = []
    for representation_id, circuit in fixed_representations():
        translation = translate(TranslationRequest(circuit=circuit, profile=profile))
        result = StateVectorSimulator(circuit.n_qubits).run(circuit)
        specification = _circuit_spec(circuit)
        manifest = translation.to_manifest()
        rows.append({
            "representation_id": representation_id,
            "circuit_specification": specification,
            "circuit_specification_sha256": _canonical_sha256(specification),
            "circuit_sha256": translation.circuit_sha256,
            "translation_manifest_sha256": _canonical_sha256(manifest),
            "pulse_schedule_sha256": translation.pulse_schedule_sha256,
            "operation_categories": _operation_categories(translation),
            "siliqun_ideal_probability_sha256": _canonical_probability_digest(result.probabilities),
            "static_specification_sha256": static_by_id[representation_id]["circuit_specification_sha256"],
        })
        del result, translation, manifest, specification
    return {
        "artifact_type": "SILIQUN_V2_B3_ALGORITHM_REPRESENTATION_EXPORT_RECEIPT",
        "version": "V1",
        "status": "PASS",
        "source": {"repository": "r3d-phd/siliqun", "source_commit": _source_head(), "baseline_root": BASELINE_ROOT, "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "profile": {"identifier": profile.identifier, "calibration_status": profile.calibration_status},
        "suite": rows,
        "suite_count": len(rows),
        "retention": {"state_vector_retained": False, "probability_vector_retained": False, "pulse_amplitude_retained": False, "pulse_phase_retained": False, "pulse_duration_retained": False, "fidelity_retained": False, "noise_draw_retained": False, "reward_retained": False, "target_state_retained": False, "measurement_result_retained": False, "logical_observable_retained": False, "algorithm_answer_retained": False, "algorithm_success_probability_retained": False, "factoring_output_retained": False, "oracle_answer_retained": False},
        "scope": {"simora_imported_or_invoked": False, "detector_or_decoder_bound": False, "frame_action_issued": False, "algorithm_ingress_established": False, "provider_or_hardware_accessed": False, "ppo_used": False},
        "claim_ceiling": catalogue["claim_ceiling"],
    }


def main() -> None:
    if "REBASELINED_TWO_QUBIT_FIXED_IDEAL_EXPORT_ONLY" not in AUTHORITY.read_text():
        raise SystemExit("b3_export_refused_authority")
    OUTPUT.write_text(json.dumps(build_export(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
