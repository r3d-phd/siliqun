"""
SiliQun OpenPulse-Compatible Pulse Schedule
=============================================
Provides an OpenPulse-semantics-aligned pulse schedule representation for
silicon spin qubit devices. This module bridges SiliQun's physics-accurate
exchange and drive pulses with the OpenPulse standard's channel/frame model.

OpenPulse is the pulse-level extension of OpenQASM 3.0, defining a standard
grammar for specifying waveforms, frames, and ports for quantum hardware control.

Standards:
    - OpenPulse specification (part of OpenQASM 3.0):
      https://openqasm.com/versions/3.0/language/openpulse.html
    - Qiskit Pulse (reference implementation):
      https://docs.quantum.ibm.com/api/qiskit/pulse

References:
    [1] Alexander et al., "Qiskit pulse: programming quantum computers through
        the cloud with pulses", Quantum Science and Technology, 2020.
    [2] Cross et al., "OpenQASM 3: A broader and deeper quantum assembly language",
        ACM Transactions on Quantum Computing, 2022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ── OpenPulse Channel Types ────────────────────────────────────────────────────

class ChannelType(Enum):
    """OpenPulse channel types for silicon spin qubit control.

    Maps to the physical control lines of a silicon spin qubit device:
    - DRIVE: Microwave drive line for ESR/EDSR single-qubit rotations
    - EXCHANGE: Voltage pulse line for exchange coupling (J gate)
    - READOUT: RF reflectometry readout line
    - ACQUIRE: Digitiser acquisition channel
    """
    DRIVE = "drive"          # d{n}: ESR/EDSR microwave drive
    EXCHANGE = "exchange"    # u{n}: Exchange coupling voltage pulse
    READOUT = "readout"      # m{n}: RF reflectometry readout
    ACQUIRE = "acquire"      # a{n}: Digitiser acquisition


@dataclass
class Channel:
    """An OpenPulse-compatible channel for silicon spin qubit control.

    Parameters
    ----------
    channel_type : ChannelType
        Type of the channel (drive, exchange, readout, acquire).
    index : int
        Channel index (maps to qubit index for drive/acquire,
        or bond index for exchange).
    sample_rate_ghz : float
        DAC/ADC sample rate in GHz. Default 1.0 GHz (1 ns resolution).
    """
    channel_type: ChannelType
    index: int
    sample_rate_ghz: float = 1.0  # GHz

    @property
    def name(self) -> str:
        prefix = {
            ChannelType.DRIVE: "d",
            ChannelType.EXCHANGE: "u",
            ChannelType.READOUT: "m",
            ChannelType.ACQUIRE: "a",
        }[self.channel_type]
        return f"{prefix}{self.index}"

    def __repr__(self):
        return f"Channel({self.name}, {self.sample_rate_ghz} GHz)"


# ── Waveform Envelopes ─────────────────────────────────────────────────────────

class WaveformShape(Enum):
    """Standard waveform shapes for pulse envelopes."""
    SQUARE = "square"           # Rectangular pulse (exchange gates)
    GAUSSIAN = "gaussian"       # Gaussian envelope (ESR drive pulses)
    DRAG = "drag"               # DRAG pulse (reduces leakage in ESR)
    COSINE_RAMP = "cosine_ramp" # Adiabatic ramp for exchange pulses
    CUSTOM = "custom"           # User-defined waveform array


@dataclass
class Waveform:
    """An OpenPulse-compatible waveform definition.

    Parameters
    ----------
    shape : WaveformShape
        Envelope shape.
    amplitude : float
        Peak amplitude (normalised, 0–1 for drive; Hz for exchange J).
    duration_ns : float
        Pulse duration in nanoseconds.
    sigma_ns : float, optional
        Gaussian sigma in nanoseconds (for GAUSSIAN/DRAG shapes).
    beta : float, optional
        DRAG correction coefficient (dimensionless).
    samples : np.ndarray, optional
        Custom waveform samples (for CUSTOM shape).
    """
    shape: WaveformShape
    amplitude: float
    duration_ns: float
    sigma_ns: Optional[float] = None
    beta: Optional[float] = None
    samples: Optional[np.ndarray] = None

    def to_samples(self, sample_rate_ghz: float = 1.0) -> np.ndarray:
        """Generate waveform samples at the given sample rate.

        Parameters
        ----------
        sample_rate_ghz : float
            DAC sample rate in GHz.

        Returns
        -------
        np.ndarray
            Complex waveform samples (I + jQ).
        """
        n_samples = max(1, int(self.duration_ns * sample_rate_ghz))
        t = np.linspace(0, self.duration_ns, n_samples)

        if self.shape == WaveformShape.SQUARE:
            envelope = np.ones(n_samples) * self.amplitude

        elif self.shape == WaveformShape.GAUSSIAN:
            sigma = self.sigma_ns or (self.duration_ns / 4)
            t0 = self.duration_ns / 2
            envelope = self.amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

        elif self.shape == WaveformShape.DRAG:
            sigma = self.sigma_ns or (self.duration_ns / 4)
            t0 = self.duration_ns / 2
            gauss = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
            d_gauss = -(t - t0) / sigma ** 2 * gauss
            beta = self.beta or 0.0
            envelope = self.amplitude * (gauss + 1j * beta * d_gauss)

        elif self.shape == WaveformShape.COSINE_RAMP:
            # Adiabatic cosine ramp: J(t) = A/2 * (1 - cos(πt/T))
            envelope = self.amplitude / 2 * (1 - np.cos(np.pi * t / self.duration_ns))

        elif self.shape == WaveformShape.CUSTOM:
            if self.samples is None:
                raise ValueError("Custom waveform requires 'samples' array.")
            return np.asarray(self.samples, dtype=complex)

        else:
            envelope = np.ones(n_samples) * self.amplitude

        return np.asarray(envelope, dtype=complex)


# ── OpenPulse-Compatible Instructions ─────────────────────────────────────────

@dataclass
class PlayInstruction:
    """OpenPulse 'play' instruction: play a waveform on a channel.

    Equivalent to: play(waveform, channel) in OpenPulse grammar.
    """
    waveform: Waveform
    channel: Channel
    t_start_ns: float = 0.0
    frame_frequency_hz: float = 0.0  # Rotating frame frequency

    @property
    def t_end_ns(self) -> float:
        return self.t_start_ns + self.waveform.duration_ns

    def __repr__(self):
        return (f"play({self.waveform.shape.value}, {self.channel.name}, "
                f"t={self.t_start_ns:.1f}ns, A={self.waveform.amplitude:.3g})")


@dataclass
class ShiftPhaseInstruction:
    """OpenPulse 'shift_phase' instruction: shift the frame phase.

    Equivalent to: shift_phase(theta, channel) in OpenPulse grammar.
    """
    phase_rad: float
    channel: Channel
    t_start_ns: float = 0.0

    def __repr__(self):
        return f"shift_phase({self.phase_rad:.3f} rad, {self.channel.name})"


@dataclass
class SetFrequencyInstruction:
    """OpenPulse 'set_frequency' instruction: set the frame frequency.

    Equivalent to: set_frequency(freq, channel) in OpenPulse grammar.
    """
    frequency_hz: float
    channel: Channel
    t_start_ns: float = 0.0

    def __repr__(self):
        return f"set_frequency({self.frequency_hz/1e6:.3f} MHz, {self.channel.name})"


@dataclass
class AcquireInstruction:
    """OpenPulse 'acquire' instruction: trigger qubit readout.

    Equivalent to: acquire(duration, channel, kernel) in OpenPulse grammar.
    """
    duration_ns: float
    channel: Channel
    acquire_channel: Channel
    t_start_ns: float = 0.0
    kernel: str = "boxcar"  # Integration kernel type

    def __repr__(self):
        return f"acquire({self.duration_ns:.1f}ns, {self.channel.name})"


# Union type for all OpenPulse instructions
PulseInstruction = Union[
    PlayInstruction,
    ShiftPhaseInstruction,
    SetFrequencyInstruction,
    AcquireInstruction,
]


# ── OpenPulse Schedule ─────────────────────────────────────────────────────────

@dataclass
class OpenPulseSchedule:
    """An OpenPulse-compatible pulse schedule for silicon spin qubit control.

    This is the standards-aligned replacement for SiliQun's custom
    PulseSequence class. It organises instructions by channel and supports
    export to Qiskit Pulse Schedule format.

    Parameters
    ----------
    name : str
        Schedule name (used in OpenPulse 'cal' blocks).
    instructions : list of PulseInstruction
        Ordered list of pulse instructions.
    device_name : str
        Name of the target device.
    sample_rate_ghz : float
        Global DAC/ADC sample rate in GHz.

    Examples
    --------
    >>> sched = OpenPulseSchedule(name="bell_state")
    >>> sched.play(WaveformShape.GAUSSIAN, amplitude=0.5, duration_ns=50,
    ...            channel_type=ChannelType.DRIVE, qubit=0)
    >>> sched.exchange(J_hz=50e6, duration_ns=10, qubit_i=0, qubit_j=1)
    >>> print(sched.summary())
    """
    name: str = "siliqun_schedule"
    instructions: List[PulseInstruction] = field(default_factory=list)
    device_name: str = "simos"
    sample_rate_ghz: float = 1.0

    @property
    def total_duration_ns(self) -> float:
        """Total schedule duration in nanoseconds."""
        if not self.instructions:
            return 0.0
        ends = []
        for inst in self.instructions:
            if hasattr(inst, 't_end_ns'):
                ends.append(inst.t_end_ns)
            elif hasattr(inst, 'duration_ns'):
                ends.append(inst.t_start_ns + inst.duration_ns)
            else:
                ends.append(inst.t_start_ns)
        return max(ends) if ends else 0.0

    @property
    def channels(self) -> List[Channel]:
        """List of unique channels used in this schedule."""
        seen = {}
        for inst in self.instructions:
            if hasattr(inst, 'channel'):
                ch = inst.channel
                seen[ch.name] = ch
        return list(seen.values())

    def play(
        self,
        shape: WaveformShape,
        amplitude: float,
        duration_ns: float,
        channel_type: ChannelType,
        qubit: int,
        t_start_ns: float = 0.0,
        sigma_ns: Optional[float] = None,
        beta: Optional[float] = None,
        frame_frequency_hz: float = 0.0,
    ) -> "OpenPulseSchedule":
        """Add a play instruction (drive pulse on a qubit)."""
        waveform = Waveform(
            shape=shape, amplitude=amplitude, duration_ns=duration_ns,
            sigma_ns=sigma_ns, beta=beta,
        )
        channel = Channel(channel_type=channel_type, index=qubit,
                          sample_rate_ghz=self.sample_rate_ghz)
        self.instructions.append(PlayInstruction(
            waveform=waveform, channel=channel,
            t_start_ns=t_start_ns, frame_frequency_hz=frame_frequency_hz,
        ))
        return self

    def exchange(
        self,
        J_hz: float,
        duration_ns: float,
        qubit_i: int,
        qubit_j: int,
        t_start_ns: float = 0.0,
        shape: WaveformShape = WaveformShape.SQUARE,
    ) -> "OpenPulseSchedule":
        """Add an exchange pulse instruction (two-qubit J gate)."""
        waveform = Waveform(
            shape=shape, amplitude=J_hz, duration_ns=duration_ns,
        )
        # Exchange channel index encodes the bond: i * 100 + j (for small n)
        bond_idx = qubit_i * 100 + qubit_j
        channel = Channel(channel_type=ChannelType.EXCHANGE, index=bond_idx,
                          sample_rate_ghz=self.sample_rate_ghz)
        self.instructions.append(PlayInstruction(
            waveform=waveform, channel=channel, t_start_ns=t_start_ns,
        ))
        return self

    def shift_phase(
        self,
        phase_rad: float,
        qubit: int,
        t_start_ns: float = 0.0,
    ) -> "OpenPulseSchedule":
        """Add a phase shift instruction."""
        channel = Channel(channel_type=ChannelType.DRIVE, index=qubit,
                          sample_rate_ghz=self.sample_rate_ghz)
        self.instructions.append(ShiftPhaseInstruction(
            phase_rad=phase_rad, channel=channel, t_start_ns=t_start_ns,
        ))
        return self

    def acquire(
        self,
        qubit: int,
        duration_ns: float = 1000.0,
        t_start_ns: float = 0.0,
    ) -> "OpenPulseSchedule":
        """Add a readout + acquire instruction pair."""
        readout_ch = Channel(ChannelType.READOUT, qubit, self.sample_rate_ghz)
        acquire_ch = Channel(ChannelType.ACQUIRE, qubit, self.sample_rate_ghz)
        self.instructions.append(AcquireInstruction(
            duration_ns=duration_ns, channel=readout_ch,
            acquire_channel=acquire_ch, t_start_ns=t_start_ns,
        ))
        return self

    def summary(self) -> str:
        """Return a human-readable schedule summary."""
        lines = [
            f"OpenPulseSchedule '{self.name}' on {self.device_name}:",
            f"  Total duration: {self.total_duration_ns:.1f} ns",
            f"  Instructions: {len(self.instructions)}",
            f"  Channels: {[ch.name for ch in self.channels]}",
        ]
        for inst in self.instructions:
            lines.append(f"    {inst}")
        return "\n".join(lines)

    def to_qiskit_schedule(self):
        """Export to a Qiskit Pulse Schedule object.

        Returns
        -------
        qiskit.pulse.Schedule
            Qiskit-compatible pulse schedule.

        Raises
        ------
        ImportError
            If Qiskit is not installed.
        """
        try:
            import qiskit.pulse as qpulse
        except ImportError:
            raise ImportError("Qiskit is required for to_qiskit_schedule(). "
                              "Install with: pip install qiskit")

        sched = qpulse.Schedule(name=self.name)
        for inst in self.instructions:
            if isinstance(inst, PlayInstruction):
                samples = inst.waveform.to_samples(self.sample_rate_ghz)
                wf = qpulse.library.Waveform(samples=samples, name=inst.waveform.shape.value)
                if inst.channel.channel_type == ChannelType.DRIVE:
                    ch = qpulse.DriveChannel(inst.channel.index)
                elif inst.channel.channel_type == ChannelType.EXCHANGE:
                    ch = qpulse.ControlChannel(inst.channel.index)
                elif inst.channel.channel_type == ChannelType.READOUT:
                    ch = qpulse.MeasureChannel(inst.channel.index)
                else:
                    continue
                t_start = int(inst.t_start_ns * self.sample_rate_ghz)
                sched.insert(t_start, qpulse.Play(wf, ch), inplace=True)
            elif isinstance(inst, ShiftPhaseInstruction):
                ch = qpulse.DriveChannel(inst.channel.index)
                t_start = int(inst.t_start_ns * self.sample_rate_ghz)
                sched.insert(t_start, qpulse.ShiftPhase(inst.phase_rad, ch), inplace=True)
        return sched

    def to_openpulse_string(self) -> str:
        """Export to an OpenPulse grammar string (OpenQASM 3.0 cal block).

        Returns
        -------
        str
            OpenPulse-formatted string suitable for embedding in OpenQASM 3.0.
        """
        lines = [
            f"// SiliQun OpenPulse schedule: {self.name}",
            f"// Device: {self.device_name}",
            f"// Duration: {self.total_duration_ns:.1f} ns",
            "cal {",
        ]
        for inst in self.instructions:
            if isinstance(inst, PlayInstruction):
                ch = inst.channel.name
                shape = inst.waveform.shape.value
                amp = inst.waveform.amplitude
                dur = inst.waveform.duration_ns
                t = inst.t_start_ns
                lines.append(
                    f"    // t={t:.1f}ns: play({shape}, amp={amp:.4g}, dur={dur:.1f}ns, ch={ch})"
                )
                lines.append(f"    play({shape}({amp}, {dur}ns), {ch});")
            elif isinstance(inst, ShiftPhaseInstruction):
                lines.append(
                    f"    shift_phase({inst.phase_rad:.4f}, {inst.channel.name});"
                )
            elif isinstance(inst, SetFrequencyInstruction):
                lines.append(
                    f"    set_frequency({inst.frequency_hz:.6e}, {inst.channel.name});"
                )
        lines.append("}")
        return "\n".join(lines)


# ── Converter: SiliQun PulseSequence → OpenPulseSchedule ──────────────────────

def from_siliqun_pulse_sequence(
    pulse_seq,
    device_name: str = "simos",
    sample_rate_ghz: float = 1.0,
    schedule_name: str = "converted",
) -> OpenPulseSchedule:
    """Convert a legacy SiliQun PulseSequence to an OpenPulseSchedule.

    Parameters
    ----------
    pulse_seq : PulseSequence
        Legacy SiliQun pulse sequence object.
    device_name : str
        Name of the target device.
    sample_rate_ghz : float
        DAC sample rate in GHz.
    schedule_name : str
        Name for the resulting schedule.

    Returns
    -------
    OpenPulseSchedule
        Standards-aligned pulse schedule.
    """
    from .lindblad import DrivePulse, ExchangePulse

    sched = OpenPulseSchedule(
        name=schedule_name,
        device_name=device_name,
        sample_rate_ghz=sample_rate_ghz,
    )

    for pulse in getattr(pulse_seq, 'pulses', []):
        if isinstance(pulse, DrivePulse):
            waveform = Waveform(
                shape=WaveformShape.GAUSSIAN,
                amplitude=pulse.amplitude,
                duration_ns=pulse.duration * 1e9,
                sigma_ns=pulse.duration * 1e9 / 4,
            )
            channel = Channel(
                channel_type=ChannelType.DRIVE,
                index=pulse.qubit,
                sample_rate_ghz=sample_rate_ghz,
            )
            sched.instructions.append(PlayInstruction(
                waveform=waveform,
                channel=channel,
                t_start_ns=pulse.t_start * 1e9,
                frame_frequency_hz=pulse.frequency,
            ))
        elif isinstance(pulse, ExchangePulse):
            waveform = Waveform(
                shape=WaveformShape.SQUARE,
                amplitude=pulse.J,
                duration_ns=pulse.duration * 1e9,
            )
            bond_idx = pulse.qubit_i * 100 + pulse.qubit_j
            channel = Channel(
                channel_type=ChannelType.EXCHANGE,
                index=bond_idx,
                sample_rate_ghz=sample_rate_ghz,
            )
            sched.instructions.append(PlayInstruction(
                waveform=waveform,
                channel=channel,
                t_start_ns=pulse.t_start * 1e9,
            ))

    return sched
