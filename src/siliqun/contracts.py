"""Static V2 translation and safe-PPO preparation contracts.

The module validates software representations only. It contains neither a
learned-control environment nor a policy-training or device-control path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

from .circuit import Circuit
from .profiles import TechnologyProfile
from .pulse import GateToPulseCompiler, PulseSchedule


BASELINE_ROOT_COMMIT: Final = "143cf5100504ab378f91c61f49c3e14b3901f04d"
TRANSLATION_CONTRACT_VERSION: Final = "siliqun-profile-circuit-translation-v1"
CIRCUIT_GRAMMAR_VERSION: Final = "siliqun-circuit-v1"
PULSE_SCHEMA_VERSION: Final = "siliqun-pulse-schedule-v1"
PROFILE_SCHEMA_VERSION: Final = "siliqun-technology-profile-v1"
SAFE_PPO_PROTOCOL_VERSION: Final = "siliqun-safe-calibration-ppo-v1"

_ALLOWED_DESIGN_MODE: Final = "design_only"
_ALLOWED_ACTION_TEMPLATES: Final = frozenset(
    {
        "select_diagnostic_template",
        "select_approved_pulse_template",
        "request_recalibration_review",
        "abstain",
    }
)
_REQUIRED_COMPARATORS: Final = frozenset(
    {"fixed_nominal", "independent_bayesian", "physics_informed_mpc_or_estimator"}
)
DESIGN_ONLY_PROHIBITED_OPERATIONS: Final = frozenset(
    {
        "train_policy",
        "create_checkpoint",
        "access_device_metadata",
        "call_provider_api",
        "submit_job",
        "export_pulse",
        "modify_supervisory_policy",
    }
)


def prohibit_runtime_operation(operation: str) -> None:
    """Refuse any operation outside the V2 design-only authority.

    This explicit guard belongs in the static package so accidental callers
    cannot reinterpret its data objects as permission for training or control.
    """

    if operation not in DESIGN_ONLY_PROHIBITED_OPERATIONS:
        raise ValueError("operation is not recognized by the design-only boundary")
    raise RuntimeError("V2 DRL preparation is design-only; runtime operation prohibited")


@dataclass(frozen=True)
class TranslationRequest:
    """A version-pinned request to map a circuit into a pulse representation."""

    circuit: Circuit
    profile: TechnologyProfile
    baseline_root_commit: str = BASELINE_ROOT_COMMIT
    contract_version: str = TRANSLATION_CONTRACT_VERSION
    circuit_grammar_version: str = CIRCUIT_GRAMMAR_VERSION
    pulse_schema_version: str = PULSE_SCHEMA_VERSION
    profile_schema_version: str = PROFILE_SCHEMA_VERSION

    def validate(self) -> None:
        expected = {
            "baseline_root_commit": (self.baseline_root_commit, BASELINE_ROOT_COMMIT),
            "contract_version": (self.contract_version, TRANSLATION_CONTRACT_VERSION),
            "circuit_grammar_version": (self.circuit_grammar_version, CIRCUIT_GRAMMAR_VERSION),
            "pulse_schema_version": (self.pulse_schema_version, PULSE_SCHEMA_VERSION),
            "profile_schema_version": (self.profile_schema_version, PROFILE_SCHEMA_VERSION),
        }
        mismatched = [name for name, (actual, required) in expected.items() if actual != required]
        if mismatched:
            raise ValueError("translation identity mismatch: " + ", ".join(mismatched))
        if self.circuit.n_qubits > self.profile.qubit_count:
            raise ValueError("circuit requires more qubits than the profile declares")
        compiler = GateToPulseCompiler(self.profile)
        for gate in self.circuit.gates:
            compiler.compile_gate(gate)

    def circuit_sha256(self) -> str:
        encoded = json.dumps(
            {
                "n_qubits": self.circuit.n_qubits,
                "gates": [
                    {"name": gate.name, "targets": gate.targets, "parameters": gate.parameters}
                    for gate in self.circuit.gates
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TranslationReceipt:
    """Non-executable result of a validated representation translation."""

    request: TranslationRequest
    pulse_schedule: PulseSchedule
    circuit_sha256: str
    pulse_schedule_sha256: str
    claim_ceiling: str = (
        "Software representation compatibility only; not a device command, "
        "calibration result, physical prediction, or performance claim."
    )

    def to_manifest(self) -> dict[str, object]:
        return {
            "contract_version": self.request.contract_version,
            "baseline_root_commit": self.request.baseline_root_commit,
            "circuit_grammar_version": self.request.circuit_grammar_version,
            "pulse_schema_version": self.request.pulse_schema_version,
            "profile_schema_version": self.request.profile_schema_version,
            "profile_id": self.request.profile.identifier,
            "calibration_status": self.request.profile.calibration_status,
            "circuit_sha256": self.circuit_sha256,
            "pulse_schedule_sha256": self.pulse_schedule_sha256,
            "pulse_schedule": self.pulse_schedule.to_manifest(),
            "claim_ceiling": self.claim_ceiling,
        }


def translate(request: TranslationRequest) -> TranslationReceipt:
    """Validate and translate to a profile-bound pulse schedule representation."""

    request.validate()
    aggregate = PulseSchedule(profile_id=request.profile.identifier)
    compiler = GateToPulseCompiler(request.profile)
    for gate in request.circuit.gates:
        aggregate.pulses.extend(compiler.compile_gate(gate).pulses)
    schedule_sha = hashlib.sha256(
        json.dumps(aggregate.to_manifest(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return TranslationReceipt(
        request=request,
        pulse_schedule=aggregate,
        circuit_sha256=request.circuit_sha256(),
        pulse_schedule_sha256=schedule_sha,
    )


@dataclass(frozen=True)
class SafePPOCalibrationPluginSpec:
    """Static contract for a future safety-constrained PPO calibration plugin."""

    protocol_version: str = SAFE_PPO_PROTOCOL_VERSION
    algorithm: str = "PPO"
    maximum_qubits: int = 5
    mode: str = _ALLOWED_DESIGN_MODE
    action_templates: frozenset[str] = _ALLOWED_ACTION_TEMPLATES
    comparators: frozenset[str] = _REQUIRED_COMPARATORS
    command_mode: str = "disabled"
    te_pws_status: str = "reserved_not_implemented"

    def validate(self) -> None:
        if self.protocol_version != SAFE_PPO_PROTOCOL_VERSION:
            raise ValueError("unexpected PPO preparation protocol version")
        if self.algorithm != "PPO":
            raise ValueError("PPO is the only authorized algorithm")
        if self.maximum_qubits < 1 or self.maximum_qubits > 5:
            raise ValueError("the preparation scope is limited to one through five qubits")
        if self.mode != _ALLOWED_DESIGN_MODE:
            raise ValueError("only design_only mode is authorized in this package")
        if self.command_mode != "disabled":
            raise ValueError("command mode must remain disabled")
        if self.te_pws_status != "reserved_not_implemented":
            raise ValueError("TE-PWS must remain reserved and unimplemented")
        if not self.action_templates or not self.action_templates <= _ALLOWED_ACTION_TEMPLATES:
            raise ValueError("action templates exceed the approved registry")
        if self.comparators != _REQUIRED_COMPARATORS:
            raise ValueError("all required non-learning comparators are mandatory")

    def to_manifest(self) -> dict[str, object]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "algorithm": self.algorithm,
            "maximum_qubits": self.maximum_qubits,
            "mode": self.mode,
            "command_mode": self.command_mode,
            "action_templates": sorted(self.action_templates),
            "comparators": sorted(self.comparators),
            "te_pws_status": self.te_pws_status,
            "claim_ceiling": (
                "Static protocol validation only; no training, calibration, "
                "device access, control action, or performance result."
            ),
        }
