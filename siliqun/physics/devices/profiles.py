"""
Device profiles for silicon spin qubit architectures.

Each profile encapsulates the complete physical specification of a
device: Hamiltonian parameters, noise characteristics, connectivity
topology, and native gate set.

Supported architectures:
    - Donor (P:Si): phosphorus donors in silicon (UNSW)
    - SiMOS: MOS quantum dots with micromagnet (Intel/UNSW)
    - GAA: gate-all-around nanowire dots (next-generation)
    - SLEDGE: exchange-only encoded qubits in Si/SiGe (HRL)
              Calibrated to Weinstein et al., Nature 615, 817-822 (2023)
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
        Architecture type: "donor", "simos", "gaa", or "sledge".
    n_qubits : int
        Number of qubits (logical qubits for DFS-encoded devices).
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
    grid_shape : tuple, optional
        (rows, cols) for 2D grid topologies. None for linear chains.
    dfs_encoded : bool
        Whether qubits use DFS encoding (3 physical spins per logical).
    n_physical_qubits : int, optional
        Total physical qubits. For DFS: 3 x n_qubits. Otherwise: n_qubits.
    sequential_pulsing : bool
        Whether exchange pulses must be applied sequentially (no
        simultaneous pulsing on adjacent qubits). True for SLEDGE.
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
    grid_shape: Optional[Tuple[int, int]] = None
    dfs_encoded: bool = False
    n_physical_qubits: Optional[int] = None
    sequential_pulsing: bool = False

    def __post_init__(self):
        if self.n_physical_qubits is None:
            if self.dfs_encoded:
                self.n_physical_qubits = 3 * self.n_qubits
            else:
                self.n_physical_qubits = self.n_qubits

    @property
    def is_linear(self) -> bool:
        """Check if the connectivity is a linear chain."""
        for i, j in self.connectivity:
            if abs(i - j) != 1:
                return False
        return True

    @property
    def is_2d(self) -> bool:
        """Check if the device has a 2D grid topology."""
        return self.grid_shape is not None

    @property
    def n_edges(self) -> int:
        """Number of coupling edges in the connectivity graph."""
        return len(self.connectivity)


# ======================================================================
# 2D grid topology helpers
# ======================================================================

def _grid_connectivity(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Generate nearest-neighbor connectivity for a 2D grid.

    Qubit indexing: row-major order.
        q(r, c) = r * cols + c

    Returns list of (i, j) edges with i < j.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # Right neighbour
            if c + 1 < cols:
                edges.append((idx, idx + 1))
            # Down neighbour
            if r + 1 < rows:
                edges.append((idx, idx + cols))
    return edges


def _grid_layout(
    rows: int, cols: int, spacing: float = 80.0
) -> List[Tuple[float, float]]:
    """Generate physical (x, y) coordinates for a 2D grid.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    spacing : float
        Physical spacing between adjacent dots (nm).
    """
    layout = []
    for r in range(rows):
        for c in range(cols):
            layout.append((c * spacing, r * spacing))
    return layout


# ======================================================================
# Preset device profiles
# ======================================================================

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
            "single": 1e-6,    # 1 us ESR rotation
            "two": 100e-9,     # 100 ns exchange gate
            "readout": 10e-6,  # 10 us readout
        },
        qubit_layout=[(i * spacing, 0.0) for i in range(n_qubits)],
    )


def simos_device(n_qubits: int = 4) -> DeviceProfile:
    """SiMOS quantum dot device.

    Based on Intel / UNSW SiMOS parameters:
    - EDSR-driven via micromagnet gradient
    - Exchange-coupled two-qubit gates
    - Moderate T1 (10s), shorter T2* (20us)
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
            "readout": 5e-6,   # 5 us readout
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
            "readout": 3e-6,   # 3 us readout
        },
        qubit_layout=[(i * spacing, 0.0) for i in range(n_qubits)],
    )


def sledge_device(
    n_qubits: int = 3,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> DeviceProfile:
    """HRL SLEDGE device - exchange-only encoded qubits in Si/SiGe.

    Calibrated to experimental data from:
        Weinstein et al., "Universal logic with encoded spin qubits
        in silicon", Nature 615, 817-822 (2023).

    Architecture:
    - Exchange-only control (no microwaves needed)
    - Decoherence-free subspace (DFS) encoding:
      3 physical spins -> 1 logical qubit
    - Quadratic error suppression: eps ~ (t/T2*)^2
    - Sequential pulsing (no simultaneous exchange on adjacent pairs)
    - Si/SiGe heterostructure with overlapping Al gates

    Experimental parameters (from the paper):
    ---------------------------------------------
    Exchange frequency J/h:      100 MHz (tunable via gate voltage)
    Exchange oscillation quality: N_osc = 57.6 at J/h = 100 MHz
    T2* (Gaussian decay):        3.5 us
    Single-qubit Clifford error:  (1.1 +/- 0.1) x 10-^3
    Two-qubit CNOT error:         3.7% (FW-CNOT, 96.3% fidelity)
    Two-qubit Clifford error:     2.9% (97.1% fidelity, RB)
    Encoded SWAP error:           0.7% (99.3% fidelity)
    LCCZ error:                   6.2% (93.8% fidelity)
    Leakage per Clifford:         (3 +/- 1) x 10-^4
    SPAM fidelity:                ~96%
    Dot spacing:                  ~80 nm
    Temperature:                  ~20 mK

    Parameters
    ----------
    n_qubits : int
        Number of logical (encoded) qubits. Default 3.
    grid_shape : tuple, optional
        (rows, cols) for 2D grid topology. If None, uses linear chain.
        For 9 logical qubits, use (3, 3).
    """
    # Physical qubits: 3 per logical qubit
    n_physical = 3 * n_qubits

    # Determine topology
    if grid_shape is not None:
        rows, cols = grid_shape
        assert rows * cols == n_qubits, (
            f"grid_shape {grid_shape} does not match n_qubits={n_qubits}"
        )
        connectivity = _grid_connectivity(rows, cols)
        spacing = 80.0  # nm between logical qubit centres
        qubit_layout = _grid_layout(rows, cols, spacing)
    else:
        connectivity = [(i, i + 1) for i in range(n_qubits - 1)]
        spacing = 80.0
        qubit_layout = [(i * spacing, 0.0) for i in range(n_qubits)]

    # Exchange couplings: J/h ~ 100 MHz with +/-5 MHz variation
    # (tunable via gate voltage, variation from fabrication disorder)
    n_edges = len(connectivity)
    exchange_couplings_list = [
        100e6 + np.random.uniform(-5e6, 5e6) for _ in range(n_edges)
    ]

    # Build Hamiltonian params
    # For DFS encoding, the Hamiltonian operates on logical qubits
    # but the exchange couplings are between physical spins within
    # and between encoded qubits
    params = DeviceParams(
        n_qubits=n_qubits,
        device_type="sledge",
        B_field=0.0,  # No external B field needed (exchange-only)
        exchange_couplings=exchange_couplings_list,
        hyperfine_couplings=[0.0] * n_qubits,  # Si/SiGe: no nuclear spins
        soc_strengths=[0.0] * n_qubits,  # No SOC needed
        temperature=0.02,  # 20 mK
    )

    # Noise parameters calibrated to Nature 2023 data
    noise = default_noise_params(n_qubits, "sledge")

    return DeviceProfile(
        name=f"SLEDGE-{n_qubits}Q"
              + (f"-{grid_shape[0]}x{grid_shape[1]}" if grid_shape else ""),
        device_type="sledge",
        n_qubits=n_qubits,
        hamiltonian_params=params,
        noise_params=noise,
        connectivity=connectivity,
        native_gates=[
            "Exchange_partial_SWAP",  # U(theta) = cos(theta/2)I + i*sin(theta/2)SWAP
            "Encoded_SWAP",           # Full SWAP between encoded qubits
            "FW_CNOT",                # Fong-Wandzura CNOT decomposition
            "LCCZ",                   # Leakage-controlled CZ
        ],
        gate_times={
            "single": 10e-9,     # 10 ns (single exchange pulse)
            "two": 100e-9,       # ~100 ns (multi-pulse FW-CNOT)
            "swap": 30e-9,       # 30 ns encoded SWAP
            "lccz": 200e-9,      # ~200 ns LCCZ gate
            "readout": 10e-6,    # 10 us Pauli spin blockade readout
            "idle": 10e-9,       # 10 ns typical idle between pulses
        },
        qubit_layout=qubit_layout,
        grid_shape=grid_shape,
        dfs_encoded=True,
        n_physical_qubits=n_physical,
        sequential_pulsing=True,
    )


def riken_5q_device(
    n_qubits: int = 5,
) -> DeviceProfile:
    """RIKEN 5-qubit Si/SiGe spin qubit device.

    Based on the noise characterization by Rojas-Arias et al.,
    arXiv:2603.03051 (2026). This is a linear array of 5 single-spin
    qubits (NOT DFS-encoded) on a Si/SiGe heterostructure.

    Key experimental parameters:
    ----------------------------
    Qubit spacing:               108 nm
    TLF correlation length:      81 nm (l_c)
    TLF areal density:           3e10 cm^-2
    NN charge noise correlation: 0.26 (measured: 0.33-0.57 tunable)
    Global magnetic drift:       ~8 Hz/s
    N_c (correlation in spacings): 0.75
    """
    connectivity = [(i, i + 1) for i in range(n_qubits - 1)]
    spacing = 108.0  # nm (RIKEN device)
    qubit_layout = [(i * spacing, 0.0) for i in range(n_qubits)]

    n_edges = len(connectivity)
    exchange_couplings_list = [
        50e6 + np.random.uniform(-3e6, 3e6) for _ in range(n_edges)
    ]

    params = DeviceParams(
        n_qubits=n_qubits,
        device_type="riken_5q",
        B_field=0.8,  # External B field (T)
        exchange_couplings=exchange_couplings_list,
        hyperfine_couplings=[0.0] * n_qubits,  # Isotopically purified Si-28
        soc_strengths=[0.0] * n_qubits,
        temperature=0.02,  # 20 mK
    )

    noise = NoiseParams(
        t1_times=[50.0] * n_qubits,
        t2_star_times=[20e-6] * n_qubits,
        t2_echo_times=[100e-6] * n_qubits,
        charge_noise_amplitude=1.5e-6,
        charge_noise_correlation_length=1,
        measurement_fidelity=0.990,
        dephasing_model="gaussian",
        exchange_frequency=50e6,
        pulse_duration=100e-9,
        idle_duration=50e-9,
        n_exchange_oscillations=20.0,
        # TLF correlation model (Rojas-Arias et al., 2026)
        tlf_density=3e10,
        tlf_correlation_length=81.0,
        qubit_spacing=108.0,
        magnetic_drift_rate=8.0,
        correlation_model="tlf",
    )

    return DeviceProfile(
        name=f"RIKEN-{n_qubits}Q",
        device_type="riken_5q",
        n_qubits=n_qubits,
        hamiltonian_params=params,
        noise_params=noise,
        connectivity=connectivity,
        native_gates=["CZ", "Rx", "Ry", "Rz"],
        gate_times={
            "single": 100e-9,
            "two": 200e-9,
            "readout": 5e-6,
            "idle": 50e-9,
        },
        qubit_layout=qubit_layout,
        grid_shape=None,
        dfs_encoded=False,
        n_physical_qubits=n_qubits,
        sequential_pulsing=True,
    )


# ======================================================================
# Profile registry
# ======================================================================

DEVICE_REGISTRY = {
    "donor": donor_device,
    "simos": simos_device,
    "gaa": gaa_device,
    "sledge": sledge_device,
    "riken_5q": riken_5q_device,
}


def get_device_profile(
    device_type: str,
    n_qubits: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> DeviceProfile:
    """Get a device profile by type and qubit count.

    Parameters
    ----------
    device_type : str
        One of "donor", "simos", "gaa", "sledge".
    n_qubits : int, optional
        Number of qubits. Uses default if not specified.
    grid_shape : tuple, optional
        (rows, cols) for 2D grid topology (only for SLEDGE).
    """
    if device_type not in DEVICE_REGISTRY:
        raise ValueError(
            f"Unknown device type '{device_type}'. "
            f"Available: {list(DEVICE_REGISTRY.keys())}"
        )
    factory = DEVICE_REGISTRY[device_type]

    kwargs = {}
    if n_qubits is not None:
        kwargs["n_qubits"] = n_qubits
    if grid_shape is not None and device_type == "sledge":
        kwargs["grid_shape"] = grid_shape

    return factory(**kwargs)


# ======================================================================
# Convenience functions for common configurations
# ======================================================================

def sledge_linear(n_qubits: int) -> DeviceProfile:
    """SLEDGE device with linear chain topology."""
    return sledge_device(n_qubits=n_qubits, grid_shape=None)


def sledge_2x2() -> DeviceProfile:
    """SLEDGE device with 2x2 grid (4 logical qubits)."""
    return sledge_device(n_qubits=4, grid_shape=(2, 2))


def sledge_3x2() -> DeviceProfile:
    """SLEDGE device with 3x2 grid (6 logical qubits)."""
    return sledge_device(n_qubits=6, grid_shape=(3, 2))


def sledge_4x2() -> DeviceProfile:
    """SLEDGE device with 4x2 grid (8 logical qubits)."""
    return sledge_device(n_qubits=8, grid_shape=(4, 2))


def sledge_3x3() -> DeviceProfile:
    """SLEDGE device with 3x3 grid (9 logical qubits)."""
    return sledge_device(n_qubits=9, grid_shape=(3, 3))


def sledge_4x4() -> DeviceProfile:
    """SLEDGE device with 4x4 grid (16 logical qubits, 48 physical spins).

    This configuration requires the StateVectorSimulator (GPU backend)
    as the MPS backend cannot efficiently handle the 2D entanglement
    structure at this scale.

    Memory requirement: ~1 MB (logical state vector of 2^16 = 65,536).
    """
    return sledge_device(n_qubits=16, grid_shape=(4, 4))


def sledge_5x5() -> DeviceProfile:
    """SLEDGE device with 5x5 grid (25 logical qubits, 75 physical spins).

    This is the largest configuration supported by SiliQun, requiring
    the GPU-accelerated StateVectorSimulator.

    Memory requirement: ~512 MB (logical state vector of 2^25 = 33,554,432).
    Feasible on a single NVIDIA A100 GPU (80 GB HBM2e).
    """
    return sledge_device(n_qubits=25, grid_shape=(5, 5))
