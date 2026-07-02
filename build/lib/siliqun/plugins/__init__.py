"""
SiliQun SDK Plugins
====================
Standard ecosystem plugin interfaces for the generalised SiliQun platform.

Provides:
    - SiliQunBackendV2: Qiskit BackendV2-compatible backend
    - SiliQunDevice: PennyLane Device API v2-compatible device
    - OpenQASM3Compiler: Standards-based circuit compiler
    - OpenPulseSchedule: OpenPulse-compatible pulse schedule

Standards:
    - Qiskit BackendV2: https://docs.quantum.ibm.com/api/qiskit/providers
    - PennyLane Device API: https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.Device.html
    - OpenQASM 3.0: https://openqasm.com/versions/3.0/
    - OpenPulse: https://openqasm.com/versions/3.0/language/openpulse.html
"""

from .pennylane_device import SiliQunDevice, register_pennylane_device, PENNYLANE_AVAILABLE
from ..compiler.qasm3_compiler import (
    SiliQunBackendV2,
    OpenQASM3Compiler,
    CompiledPulseSchedule,
    SiliQunJob,
    SiliQunResult,
    QISKIT_AVAILABLE,
)
from ..pulse.openpulse_schedule import (
    OpenPulseSchedule,
    Channel,
    ChannelType,
    Waveform,
    WaveformShape,
    PlayInstruction,
    ShiftPhaseInstruction,
    SetFrequencyInstruction,
    AcquireInstruction,
    from_siliqun_pulse_sequence,
)

__all__ = [
    "SiliQunBackendV2", "SiliQunJob", "SiliQunResult", "QISKIT_AVAILABLE",
    "SiliQunDevice", "register_pennylane_device", "PENNYLANE_AVAILABLE",
    "OpenQASM3Compiler", "CompiledPulseSchedule",
    "OpenPulseSchedule", "Channel", "ChannelType", "Waveform", "WaveformShape",
    "PlayInstruction", "ShiftPhaseInstruction", "SetFrequencyInstruction",
    "AcquireInstruction", "from_siliqun_pulse_sequence",
]
