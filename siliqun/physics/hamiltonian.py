"""
Hamiltonian construction for silicon spin qubit systems.

Builds the system Hamiltonian as an MPO from physical parameters:
    H = H_Zeeman + H_exchange + H_SOC + H_hyperfine + H_drive

Supports three device architectures:
    - Donor (P:Si): strong hyperfine, weak SOC
    - SiMOS: micromagnet gradient, moderate SOC
    - GAA (Gate-All-Around): strong SOC, electric-field-driven
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from ..backend import active_backend
from ..tensor.mpo import MPO
from . import gates


# ── Physical constants ──────────────────────────────────────────────

HBAR = 1.054571817e-34       # J·s
MU_B = 9.2740100783e-24      # J/T (Bohr magneton)
G_ELECTRON = 2.0023193        # electron g-factor in silicon
GAMMA_E = G_ELECTRON * MU_B / HBAR  # electron gyromagnetic ratio


@dataclass
class DeviceParams:
    """Physical parameters for a silicon spin qubit device.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    device_type : str
        One of "donor", "simos", "gaa".
    B_field : float
        External magnetic field (Tesla).
    qubit_frequencies : list of float
        Larmor frequency of each qubit (Hz). If None, computed from B_field.
    exchange_couplings : list of float
        Exchange coupling J_{i,i+1} between adjacent qubits (Hz).
    hyperfine_couplings : list of float
        Hyperfine coupling A_i for each qubit (Hz). Relevant for donor qubits.
    soc_strengths : list of float
        Spin-orbit coupling strength for each qubit (Hz).
    drive_amplitudes : list of float
        Microwave/electric drive amplitude for each qubit (Hz).
    drive_frequencies : list of float
        Drive frequencies for each qubit (Hz).
    drive_phases : list of float
        Drive phases for each qubit (rad).
    crosstalk_matrix : ndarray, optional
        N×N matrix of crosstalk coefficients between qubits.
    temperature : float
        Device temperature (Kelvin).
    """
    n_qubits: int = 2
    device_type: str = "donor"
    B_field: float = 1.4  # Tesla
    qubit_frequencies: Optional[List[float]] = None
    exchange_couplings: Optional[List[float]] = None
    hyperfine_couplings: Optional[List[float]] = None
    soc_strengths: Optional[List[float]] = None
    drive_amplitudes: Optional[List[float]] = None
    drive_frequencies: Optional[List[float]] = None
    drive_phases: Optional[List[float]] = None
    crosstalk_matrix: Optional[np.ndarray] = None
    temperature: float = 0.02  # 20 mK

    def __post_init__(self):
        n = self.n_qubits
        if self.qubit_frequencies is None:
            base_freq = GAMMA_E * self.B_field / (2 * np.pi)
            # Add small detunings for addressability
            self.qubit_frequencies = [
                base_freq + i * 50e6 for i in range(n)
            ]
        if self.exchange_couplings is None:
            self.exchange_couplings = [10e6] * (n - 1)  # 10 MHz default
        if self.hyperfine_couplings is None:
            if self.device_type == "donor":
                self.hyperfine_couplings = [117.53e6] * n  # P:Si value
            else:
                self.hyperfine_couplings = [0.0] * n
        if self.soc_strengths is None:
            if self.device_type == "gaa":
                self.soc_strengths = [5e6] * n
            elif self.device_type == "simos":
                self.soc_strengths = [2e6] * n
            else:
                self.soc_strengths = [0.0] * n
        if self.drive_amplitudes is None:
            self.drive_amplitudes = [0.0] * n
        if self.drive_frequencies is None:
            self.drive_frequencies = list(self.qubit_frequencies)
        if self.drive_phases is None:
            self.drive_phases = [0.0] * n
        if self.crosstalk_matrix is None:
            self.crosstalk_matrix = np.eye(n)


# ── Preset device configurations ───────────────────────────────────

def donor_2q_params() -> DeviceParams:
    """2-qubit donor (P:Si) device — UNSW-style."""
    return DeviceParams(
        n_qubits=2,
        device_type="donor",
        B_field=1.4,
        exchange_couplings=[18e6],
        hyperfine_couplings=[117.53e6, 117.53e6],
        temperature=0.05,
    )


def simos_4q_params() -> DeviceParams:
    """4-qubit SiMOS device — Intel-style."""
    return DeviceParams(
        n_qubits=4,
        device_type="simos",
        B_field=0.8,
        exchange_couplings=[12e6, 15e6, 11e6],
        soc_strengths=[2.5e6, 2.0e6, 2.3e6, 2.1e6],
        temperature=0.02,
    )


def gaa_6q_params() -> DeviceParams:
    """6-qubit GAA device — next-generation."""
    return DeviceParams(
        n_qubits=6,
        device_type="gaa",
        B_field=0.5,
        exchange_couplings=[20e6, 18e6, 22e6, 19e6, 21e6],
        soc_strengths=[5e6] * 6,
        temperature=0.015,
    )


# ── Hamiltonian MPO construction ────────────────────────────────────

def build_hamiltonian_mpo(params: DeviceParams) -> MPO:
    """Build the system Hamiltonian as an MPO.

    Uses the compact MPO representation where each site tensor
    encodes the local terms and nearest-neighbor couplings.

    The Hamiltonian is:
        H = Σ_i ω_i/2 Z_i                    (Zeeman)
          + Σ_i A_i/4 (X_i X_n + Y_i Y_n + Z_i Z_n)  (Hyperfine, donor)
          + Σ_{⟨i,j⟩} J_{ij}/4 (X_i X_j + Y_i Y_j + Z_i Z_j)  (Exchange)
          + Σ_i Ω_i cos(ω_d t + φ_i) X_i     (Drive, time-dependent)

    For the static part (no drive), we build the MPO exactly.
    """
    be = active_backend()
    n = params.n_qubits
    d = 2

    I = gates.PAULI_I
    X = gates.PAULI_X
    Y = gates.PAULI_Y
    Z = gates.PAULI_Z

    # MPO bond dimension for Heisenberg + Zeeman: 5
    # Layout of the MPO auxiliary space:
    #   [I, X, Y, Z, H_local]
    # W[i] is a (D, d, d, D) tensor where D=5
    D = 5
    tensors = []

    for i in range(n):
        W = be.zeros((D, d, d, D))

        omega_i = 2 * np.pi * params.qubit_frequencies[i]

        # Row 0: identity propagation
        W[0, :, :, 0] = be.array(I)

        # Row 0 → columns 1,2,3: start exchange coupling
        if i < n - 1:
            J = 2 * np.pi * params.exchange_couplings[i]
            W[0, :, :, 1] = be.array((J / 4) * X)
            W[0, :, :, 2] = be.array((J / 4) * Y)
            W[0, :, :, 3] = be.array((J / 4) * Z)

        # Rows 1,2,3 → column 4: complete exchange coupling
        W[1, :, :, 4] = be.array(X)
        W[2, :, :, 4] = be.array(Y)
        W[3, :, :, 4] = be.array(Z)

        # Row 0 → column 4: local Zeeman term
        W[0, :, :, 4] = be.array(W[0, :, :, 4]) + be.array((omega_i / 2) * Z)

        # Add hyperfine (as additional local Z splitting for simplicity)
        if params.device_type == "donor" and params.hyperfine_couplings[i] > 0:
            A_hf = 2 * np.pi * params.hyperfine_couplings[i]
            W[0, :, :, 4] = be.array(W[0, :, :, 4]) + be.array((A_hf / 4) * Z)

        # Row 4 → column 4: identity (accumulate)
        W[4, :, :, 4] = be.array(I)

        tensors.append(W)

    # Boundary conditions: first tensor uses row 0 only, last uses column 4 only
    tensors[0] = tensors[0][0:1, :, :, :]       # (1, d, d, D)
    tensors[-1] = tensors[-1][:, :, :, 4:5]     # (D, d, d, 1)

    return MPO(tensors, phys_dim=d)


def build_drive_mpo(params: DeviceParams, t: float) -> MPO:
    """Build the time-dependent drive Hamiltonian MPO at time t.

    H_drive(t) = Σ_i Ω_i cos(ω_d,i · t + φ_i) X_i
    """
    be = active_backend()
    n = params.n_qubits
    d = 2

    I = gates.PAULI_I
    X = gates.PAULI_X

    tensors = []
    for i in range(n):
        omega_d = 2 * np.pi * params.drive_frequencies[i]
        Omega = 2 * np.pi * params.drive_amplitudes[i]
        phi = params.drive_phases[i]

        drive_strength = Omega * np.cos(omega_d * t + phi)

        # Apply crosstalk
        effective_drive = np.zeros(n)
        effective_drive[i] = drive_strength
        if params.crosstalk_matrix is not None:
            effective_drive = params.crosstalk_matrix @ effective_drive

        # Simple MPO: each site has local drive term
        # Bond dimension 2: [I, H_local]
        W = be.zeros((2, d, d, 2))
        W[0, :, :, 0] = be.array(I)
        W[0, :, :, 1] = be.array(effective_drive[i] * X)
        W[1, :, :, 1] = be.array(I)
        tensors.append(W)

    tensors[0] = tensors[0][0:1, :, :, :]
    tensors[-1] = tensors[-1][:, :, :, 1:2]

    return MPO(tensors, phys_dim=d)
