"""Pulse-schedule schema and a small native-operation compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .circuit import Gate
from .profiles import TechnologyProfile


@dataclass(frozen=True)
class Pulse:
    """A validated pulse-level event representation, not a hardware command."""

    kind: str
    channel: str
    targets: tuple[int, ...]
    duration_s: float
    amplitude: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"drive", "exchange", "virtual_z", "measure"}:
            raise ValueError("unknown pulse kind")
        if not self.channel:
            raise ValueError("channel is required")
        if not self.targets or any(target < 0 for target in self.targets):
            raise ValueError("at least one non-negative target is required")
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if not math.isfinite(self.amplitude) or not math.isfinite(self.phase_rad):
            raise ValueError("pulse values must be finite")


@dataclass
class PulseSchedule:
    """Ordered schedule with a declared profile identifier."""

    profile_id: str
    pulses: list[Pulse] = field(default_factory=list)

    def append(self, pulse: Pulse) -> "PulseSchedule":
        self.pulses.append(pulse)
        return self

    @property
    def duration_s(self) -> float:
        return sum(pulse.duration_s for pulse in self.pulses)

    def to_manifest(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "duration_s": self.duration_s,
            "pulses": [
                {
                    "kind": pulse.kind,
                    "channel": pulse.channel,
                    "targets": list(pulse.targets),
                    "duration_s": pulse.duration_s,
                    "amplitude": pulse.amplitude,
                    "phase_rad": pulse.phase_rad,
                }
                for pulse in self.pulses
            ],
        }


class GateToPulseCompiler:
    """Map a declared native gate to a profile-bound schedule representation."""

    def __init__(self, profile: TechnologyProfile) -> None:
        self.profile = profile

    def compile_gate(self, gate: Gate) -> PulseSchedule:
        schedule = PulseSchedule(profile_id=self.profile.identifier)
        one_qubit_duration = self.profile.parameters["one_qubit_duration_s"]
        two_qubit_duration = self.profile.parameters["two_qubit_duration_s"]
        if gate.name == "rx":
            self._require_native("rx")
            return schedule.append(
                Pulse("drive", f"drive-{gate.targets[0]}", gate.targets, one_qubit_duration, gate.parameters[0], 0.0)
            )
        if gate.name == "ry":
            self._require_native("ry")
            return schedule.append(
                Pulse("drive", f"drive-{gate.targets[0]}", gate.targets, one_qubit_duration, gate.parameters[0], math.pi / 2)
            )
        if gate.name == "rz":
            self._require_native("rz")
            return schedule.append(
                Pulse("virtual_z", f"frame-{gate.targets[0]}", gate.targets, 0.0, gate.parameters[0], 0.0)
            )
        if gate.name == "cz":
            self._require_native("cz")
            edge = tuple(sorted(gate.targets))
            return schedule.append(
                Pulse("exchange", f"exchange-{edge[0]}-{edge[1]}", gate.targets, two_qubit_duration, 1.0, 0.0)
            )
        raise ValueError(f"no baseline native pulse mapping for {gate.name}")

    def _require_native(self, gate: str) -> None:
        if not self.profile.supports(gate):
            raise ValueError(f"profile does not declare native support for {gate}")
