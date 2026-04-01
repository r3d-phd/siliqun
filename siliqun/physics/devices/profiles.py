"""
Device profiles for silicon spin qubit architectures.

Each profile encapsulates the complete physical specification of a
device: Hamiltonian parameters, noise characteristics, connectivity
topology, and native gate set.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..hamiltonian import DeviceParams, donor_2q_params, simos_4q_params, gaa_6q_params
from ..noise.channels import NoiseParams, default_noise_params


@dataclass
class DeviceProfile:
    """Complete physical specification of a silicon spin qubit device.

    Parameters
    ----------
    name : str
        Human-readable device name.
    device_type : str
        Architecture type: "donor", "simos", or "gaa".
    n_qubits : int
        Number of qubits.
    hamiltonian_params : DeviceParams
        Hamiltonian parameters.
    noise_params : NoiseParams
        Noise model parameters.
    connectivity : list of tuple
        List of (i, j) pairs indicating which qubits are coupled.
    native_gates : list of str
        Names of natively supported gates.
    gate_times : dict
        Gate execution times in seconds.
    qubit_layout : list of tuple
        Physical (x, y) coordinates of each qubit (nm).
    """
    name: str
    device_type: str
    n_qubits: int
    hamiltonian_params: DeviceParams
    noise_params: NoiseParams
    connectivity: List[Tuple[int, int]]
    native_gates: List[str]
    gate_times: Dict[str, float]
    qubit_layout: Optional[List[Tuple[float, float]]] = None

    @property
    def is_linear(self) -> bool:
        """Check if the connectivity is a linear chain."""
        for i, j in self.connectivity:
            if abs(i - j) != 1:
                return False
        return True


# ── Preset device profiles ──────────────────────────────────────────

def donor_device(n_qubits: int = 2) -> DeviceProfile:
    """Phosphorus donor in silicon (P:Si) device.

    Based on UNSW experimental parameters:
    - ESR-driven single-qubit gates
    - Exchange-coupled two-qubit gates
    - Very long T1 (30s), moderate T2* (0.5ms)
    - Strong hyperfine coupling (117.53 MHz)
    """
    params = DeviceParams(
        n_qubits=n_qubits,
        device_type="donor",
        B_field=1.4,
        exchange_couplings=[18e6] * (n_qubits - 1),
        hyperfine_couplings=[117.53e6] * n_qubits,
        temperature=0.05,
    )
    noise = default_noise_params(n_qubits, "donor")
    connectivity = [(i, i + 1) for i in range(n_qubits - 1)]
    spacing = 15.0  # nm between donors

    return DeviceProfile(
        name=f"Donor-{n_qubits}Q",
        device_type="donor",
        n_qubits=n_qubits,
        hamiltonian_params=params,
        noise_params=noise,
        connectivity=connectivity,
        native_gates=["Rz", "ESR_Rx", "ESR_Ry", "Exchange_SWAP"],
        gate_times={
            "single": 1e-6,    # 1 μs ESR rotation
            "two": 100e-9,     # 100 ns exchange gate
            "readout": 10e-6,  # 10 μs readout
        },
        qubit_layout=[(i * spacing, 0.0) for i in range(n_qubits)],
    )


def simos_device(n_qubits: int = 4) -> DeviceProfile:
    """SiMOS quantum dot device.

    Based on Intel / UNSW SiMOS parameters:
    - EDSR-driven via micromagnet gradient
    - Exchange-coupled two-qubit gates
    - Moderate T1 (10s), shorter T2* (20μs)
    - Higher charge noise sensitivity
    """
    params = DeviceParams(
        n_qubits=n_qubits,
        device_type="simos",
        B_field=0.8,
        exchange_couplings=[12e6 + np.random.uniform(-2e6, 2e6)
                           for _ in range(n_qubits - 1)],
        soc_strengths=[2e6 + np.random.uniform(-0.5e6, 0.5e6)
                      for _ in range(n_qubits)],
        temperature=0.02,
    )
    noise = default_noise_params(n_qubits, "simos")
    connectivity = [(i, i + 1) for i in range(n_qubits - 1)]
    spacing = 80.0  # nm between quantum dots

    return DeviceProfile(
        name=f"SiMOS-{n_qubits}Q",
        device_type="simos",
        n_qubits=n_qubits,
        hamiltonian_params=params,
        noise_params=noise,
        connectivity=connectivity,
        native_gates=["Rz", "EDSR_Rx", "EDSR_Ry", "Exchange_SWAP", "CZ"],
        gate_times={
            "single": 200e-9,  # 200 ns EDSR rotation
            "two": 50e-9,      # 50 ns exchange gate
            "readout": 5e-6,   # 5 μs readout
        },
        qubit_layout=[(i * spacing, 0.0) for i in range(n_qubits)],
    )


def gaa_device(n_qubits: int = 6) -> DeviceProfile:
    """Gate-All-Around (GAA) nanowire device.

    Next-generation architecture:
    - Strong spin-orbit coupling for all-electric control
    - Faster gates but shorter coherence times
    - Higher charge noise
    """
    params = DeviceParams(
        n_qubits=n_qubits,
        device_type="gaa",
        B_field=0.5,
        exchange_couplings=[20e6 + np.random.uniform(-3e6, 3e6)
                           for _ in range(n_qubits - 1)],
        soc_strengths=[5e6 + np.random.uniform(-1e6, 1e6)
                      for _ in range(n_qubits)],
        temperature=0.015,
    )
    noise = default_noise_params(n_qubits, "gaa")
    connectivity = [(i, i + 1) for i in range(n_qubits - 1)]
    spacing = 60.0  # nm between nanowire dots

    return DeviceProfile(
        name=f"GAA-{n_qubits}Q",
        device_type="gaa",
        n_qubits=n_qubits,
        hamiltonian_params=params,
        noise_params=noise,
        connectivity=connectivity,
        native_gates=["Rz", "EDSR_Rx", "EDSR_Ry", "sqrt_SWAP", "CZ"],
        gate_times={
            "single": 50e-9,   # 50 ns all-electric rotation
            "two": 30e-9,      # 30 ns exchange gate
            "readout": 3e-6,   # 3 μs readout
        },
        qubit_layout=[(i * spacing, 0.0) for i in range(n_qubits)],
    )


# ── Profile registry ───────────────────────────────────────────────

DEVICE_REGISTRY = {
    "donor": donor_device,
    "simos": simos_device,
    "gaa": gaa_device,
}


def get_device_profile(
    device_type: str, n_qubits: Optional[int] = None
) -> DeviceProfile:
    """Get a device profile by type and qubit count."""
    if device_type not in DEVICE_REGISTRY:
        raise ValueError(
            f"Unknown device type '{device_type}'. "
            f"Available: {list(DEVICE_REGISTRY.keys())}"
        )
    factory = DEVICE_REGISTRY[device_type]
    if n_qubits is not None:
        return factory(n_qubits)
    return factory()
