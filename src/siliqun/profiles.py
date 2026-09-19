"""Technology-profile provenance and calibration-status invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


_ALLOWED_CALIBRATION_STATUS = {"literature_parameterised", "virtual_reference", "named_device"}
_REQUIRED_NAMED_DEVICE_FIELDS = {
    "device_id",
    "uncertainty_model",
    "heldout_validation",
    "measurement_model",
    "noise_cross_spectra",
}


@dataclass(frozen=True)
class TechnologyProfile:
    """Technology-specific inputs with an explicit calibration claim ceiling."""

    identifier: str
    technology_family: str
    qubit_count: int
    connectivity: tuple[tuple[int, int], ...]
    native_gates: tuple[str, ...]
    parameters: Mapping[str, float]
    citations: tuple[str, ...]
    calibration_status: str
    calibration_evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier or not self.technology_family:
            raise ValueError("profile identity fields are required")
        if not isinstance(self.qubit_count, int) or self.qubit_count <= 0:
            raise ValueError("qubit_count must be a positive integer")
        if self.calibration_status not in _ALLOWED_CALIBRATION_STATUS:
            raise ValueError("unknown calibration_status")
        if not self.native_gates:
            raise ValueError("at least one native gate is required")
        if not self.citations:
            raise ValueError("at least one source citation is required")
        for edge in self.connectivity:
            if len(edge) != 2 or edge[0] == edge[1]:
                raise ValueError("connectivity edges must contain two distinct endpoints")
            if any(index < 0 or index >= self.qubit_count for index in edge):
                raise ValueError("connectivity endpoint is outside the profile")
        if self.calibration_status == "named_device":
            missing = _REQUIRED_NAMED_DEVICE_FIELDS - set(self.calibration_evidence)
            if missing:
                raise ValueError(
                    "named_device profiles require: " + ", ".join(sorted(missing))
                )

    @property
    def is_named_device(self) -> bool:
        return self.calibration_status == "named_device"

    def supports(self, gate: str) -> bool:
        return gate.lower() in self.native_gates

    def to_manifest(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "technology_family": self.technology_family,
            "qubit_count": self.qubit_count,
            "connectivity": [list(edge) for edge in self.connectivity],
            "native_gates": list(self.native_gates),
            "parameters": dict(self.parameters),
            "citations": list(self.citations),
            "calibration_status": self.calibration_status,
            "calibration_evidence": dict(self.calibration_evidence),
        }


def simos_nominal_profile() -> TechnologyProfile:
    """Return a two-qubit literature-parameterised SiMOS nominal profile.

    The values are fixed nominal metadata for software tests and examples. They
    do not identify, calibrate, or predict any particular physical device.
    """

    return TechnologyProfile(
        identifier="simos-nominal-literature-v1",
        technology_family="silicon-mos-quantum-dot",
        qubit_count=2,
        connectivity=((0, 1),),
        native_gates=("rx", "ry", "rz", "cz", "measure"),
        parameters={
            "t1_s": 1.0e-3,
            "t2_s": 5.0e-4,
            "one_qubit_duration_s": 10.0e-9,
            "two_qubit_duration_s": 100.0e-9,
            "two_qubit_error_proxy": 0.01,
        },
        citations=(
            "doi:10.1038/s41586-022-04592-2",
            "doi:10.1038/s41586-022-04541-z",
            "doi:10.1038/s41586-022-04553-9",
        ),
        calibration_status="literature_parameterised",
    )
