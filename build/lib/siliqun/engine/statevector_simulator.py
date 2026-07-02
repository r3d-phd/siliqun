"""
GPU-accelerated exact state vector simulator for SiliQun.

Operates in the logical (encoded) Hilbert space of dimension 2^n_logical,
avoiding the exponential overhead of the full physical space (2^{3n} for
DFS-encoded qubits). Supports up to 25 logical qubits (5x5 grid) on a
single A100 GPU, requiring only 512 MB of memory.

For DFS-encoded (SLEDGE) devices, exchange interactions on physical spins
are pre-projected into the logical subspace using the DFS encoding
isometry. Leakage out of the encoded subspace is tracked perturbatively
via a separate leakage accumulator.

For non-DFS devices (Donor, SiMOS, GAA), the simulator operates directly
on the physical qubit state vector.

Backend selection:
    - "numpy": CPU reference (NumPy, any system)
    - "cuda":  GPU-accelerated (CuPy + optional cuQuantum)

The simulator exposes the same public API as SiliQunSimulator (MPS-based),
enabling drop-in replacement in the gym environment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from ..backend import active_backend
from ..physics.devices.profiles import DeviceProfile
from ..physics.noise.channels import NoiseParams

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class SVSimConfig:
    """Configuration for the state vector simulator.

    Parameters
    ----------
    dt : float
        Time step for noise application (seconds).
    noise_enabled : bool
        Whether to apply noise channels.
    seed : int
        Random seed for reproducibility.
    use_gpu : bool
        Whether to attempt GPU acceleration.
    dtype : str
        Data type for the state vector: "complex128" or "complex64".
        complex64 halves memory but reduces precision.
    leakage_tracking : bool
        Whether to track DFS leakage perturbatively.
    leakage_order : int
        Perturbation order for leakage tracking (1 or 2).
    """
    dt: float = 1e-9
    noise_enabled: bool = True
    seed: int = 42
    use_gpu: bool = True
    dtype: str = "complex128"
    leakage_tracking: bool = True
    leakage_order: int = 1


# ======================================================================
# DFS Logical Gate Projector
# ======================================================================

class DFSLogicalProjector:
    """Projects physical exchange operations into the DFS logical subspace.

    For an n-logical-qubit system, pre-computes the 2^n x 2^n logical
    representation of all exchange gates that act on physical spins.

    The key identity is:
        U_logical = V^dag @ U_physical @ V
    where V is the encoding isometry (2^{3n} x 2^n).

    For efficiency, single-qubit exchange gates (J_12, J_23 within one
    encoded qubit) are pre-computed as 2x2 matrices and applied via
    tensor contraction. Two-qubit inter-qubit exchanges are pre-computed
    as 4x4 matrices.

    Parameters
    ----------
    n_logical : int
        Number of logical qubits.
    connectivity : list of tuple
        Logical qubit connectivity edges.
    """

    def __init__(self, n_logical: int, connectivity: List[Tuple[int, int]]):
        self.n_logical = n_logical
        self.n_physical = 3 * n_logical
        self.connectivity = connectivity

        # Cache for projected gates
        self._gate_cache = {}

        # Build single-qubit logical gate generators
        self._build_single_qubit_generators()

    def _build_single_qubit_generators(self):
        """Pre-compute the 2x2 logical representations of intra-qubit
        exchange gates (J_12 and J_23).

        For a single encoded qubit:
            J_12(theta) -> logical Z rotation
            J_23(theta) -> logical X rotation

        These are exact: no leakage within the encoded subspace for
        intra-qubit exchanges.
        """
        from ..physics.dfs_encoding import (
            encoded_zero, encoded_one, exchange_12, exchange_23,
        )

        zero_L = encoded_zero()  # 8-dim
        one_L = encoded_one()    # 8-dim
        V_single = np.column_stack([zero_L, one_L])  # (8, 2)

        # Store the single-qubit isometry
        self._V_single = V_single

        # Verify orthonormality
        overlap = V_single.conj().T @ V_single
        assert np.allclose(overlap, np.eye(2), atol=1e-12), \
            "DFS basis states are not orthonormal"

        # Pre-compute generator matrices for J_12 and J_23
        # H_12_logical = V^dag H_12_phys V (2x2 Hermitian)
        # H_23_logical = V^dag H_23_phys V (2x2 Hermitian)
        from ..physics.dfs_encoding import _I, _X, _Y, _Z

        # Physical Heisenberg Hamiltonian for spins 1-2
        H_12_phys = 0.25 * (
            np.kron(np.kron(_X, _X), _I) +
            np.kron(np.kron(_Y, _Y), _I) +
            np.kron(np.kron(_Z, _Z), _I)
        )
        self._H12_logical = V_single.conj().T @ H_12_phys @ V_single

        # Physical Heisenberg Hamiltonian for spins 2-3
        H_23_phys = 0.25 * (
            np.kron(_I, np.kron(_X, _X)) +
            np.kron(_I, np.kron(_Y, _Y)) +
            np.kron(_I, np.kron(_Z, _Z))
        )
        self._H23_logical = V_single.conj().T @ H_23_phys @ V_single

        logger.debug(
            "DFS single-qubit generators built: "
            "H12_logical=%s, H23_logical=%s",
            self._H12_logical, self._H23_logical,
        )

    def logical_exchange_12(self, theta: float) -> np.ndarray:
        """Logical gate from J_12 exchange (intra-qubit, spins 1-2).

        Returns a 2x2 unitary acting on one logical qubit.

        Parameters
        ----------
        theta : float
            Exchange angle J*t/hbar.
        """
        from scipy.linalg import expm
        return expm(-1j * theta * self._H12_logical)

    def logical_exchange_23(self, theta: float) -> np.ndarray:
        """Logical gate from J_23 exchange (intra-qubit, spins 2-3).

        Returns a 2x2 unitary acting on one logical qubit.
        """
        from scipy.linalg import expm
        return expm(-1j * theta * self._H23_logical)

    def logical_inter_qubit_exchange(
        self,
        theta: float,
        qubit_a: int,
        qubit_b: int,
        spin_a_local: int,
        spin_b_local: int,
    ) -> np.ndarray:
        """Logical gate from inter-qubit exchange interaction.

        Projects the exchange between physical spin spin_a_local of
        qubit_a and spin_b_local of qubit_b into the 4x4 logical
        subspace of the two encoded qubits.

        Returns a 4x4 unitary acting on logical qubits (qubit_a, qubit_b).

        Parameters
        ----------
        theta : float
            Exchange angle.
        qubit_a, qubit_b : int
            Logical qubit indices.
        spin_a_local, spin_b_local : int
            Local spin index within each encoded qubit (0, 1, or 2).
        """
        cache_key = (theta, qubit_a, qubit_b, spin_a_local, spin_b_local)
        if cache_key in self._gate_cache:
            return self._gate_cache[cache_key]

        from ..physics.dfs_encoding import _I, _X, _Y, _Z
        from scipy.linalg import expm

        # Build 2-qubit isometry V_2 = V_a kron V_b: (64, 4)
        V_2 = np.kron(self._V_single, self._V_single)  # (64, 4)

        # Build physical Hamiltonian for exchange between
        # spin spin_a_local of qubit A and spin_b_local of qubit B
        # in the 6-physical-spin space (64 x 64)
        H_phys = np.zeros((64, 64), dtype=np.complex128)

        for P in [_X, _Y, _Z]:
            # Build 6-spin operator: I^{a0} ... P^{a_local} ... I^{b0} ... P^{b_local} ...
            ops_a = [_I, _I, _I]
            ops_a[spin_a_local] = P
            ops_b = [_I, _I, _I]
            ops_b[spin_b_local] = P

            term = ops_a[0]
            for op in ops_a[1:]:
                term = np.kron(term, op)
            for op in ops_b:
                term = np.kron(term, op)

            H_phys += 0.25 * term

        # Project to logical subspace
        H_logical = V_2.conj().T @ H_phys @ V_2  # (4, 4)

        # Compute unitary
        U_logical = expm(-1j * theta * H_logical)

        self._gate_cache[cache_key] = U_logical
        return U_logical

    def compute_leakage_rate(
        self,
        theta: float,
        qubit_a: int,
        qubit_b: int,
        spin_a_local: int,
        spin_b_local: int,
    ) -> float:
        """Compute the leakage probability for an inter-qubit exchange.

        Leakage = 1 - ||P_enc U_phys V |psi_L>||^2
        averaged over the logical basis states.

        This gives the fraction of population that leaves the encoded
        subspace due to the physical exchange interaction.

        Parameters
        ----------
        theta : float
            Exchange angle.
        Returns
        -------
        float
            Average leakage probability per gate application.
        """
        from ..physics.dfs_encoding import _I, _X, _Y, _Z
        from scipy.linalg import expm

        # Build physical unitary for 2-encoded-qubit system (64x64)
        V_2 = np.kron(self._V_single, self._V_single)

        H_phys = np.zeros((64, 64), dtype=np.complex128)
        for P in [_X, _Y, _Z]:
            ops_a = [_I, _I, _I]
            ops_a[spin_a_local] = P
            ops_b = [_I, _I, _I]
            ops_b[spin_b_local] = P

            term = ops_a[0]
            for op in ops_a[1:]:
                term = np.kron(term, op)
            for op in ops_b:
                term = np.kron(term, op)
            H_phys += 0.25 * term

        U_phys = expm(-1j * theta * H_phys)

        # Projector onto encoded subspace
        P_enc = V_2 @ V_2.conj().T

        # Average leakage over computational basis
        total_leakage = 0.0
        d_L = 4  # 2-qubit logical dimension
        for i in range(d_L):
            psi_L = np.zeros(d_L, dtype=np.complex128)
            psi_L[i] = 1.0
            psi_P = V_2 @ psi_L
            psi_out = U_phys @ psi_P
            p_enc = float(np.real(psi_out.conj() @ P_enc @ psi_out))
            total_leakage += (1.0 - p_enc)

        return total_leakage / d_L


# ======================================================================
# State Vector Simulator
# ======================================================================

class StateVectorSimulator:
    """Exact state vector simulator for silicon spin qubit systems.

    Maintains the quantum state as a dense complex vector of dimension
    2^n in the logical Hilbert space. For DFS-encoded devices, physical
    exchange gates are projected into the logical subspace, and leakage
    is tracked perturbatively.

    Parameters
    ----------
    device : DeviceProfile
        Physical device specification.
    config : SVSimConfig
        Simulation configuration.
    """

    def __init__(
        self,
        device: DeviceProfile,
        config: Optional[SVSimConfig] = None,
    ):
        self.device = device
        self.config = config or SVSimConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Determine dimensions
        self.n_logical = device.n_qubits
        self.is_dfs = getattr(device, 'dfs_encoded', False)
        self.n_physical = device.n_physical_qubits or device.n_qubits

        # State vector dimension (logical space)
        self._dim = 2 ** self.n_logical
        self._dtype = np.complex128 if self.config.dtype == "complex128" else np.complex64

        # Try GPU backend
        self._use_gpu = False
        self._xp = np  # Default to NumPy
        if self.config.use_gpu:
            try:
                import cupy as cp
                self._xp = cp
                self._use_gpu = True
                logger.info(
                    "StateVectorSimulator using GPU (CuPy), "
                    "dim=%d, memory=%.1f MB",
                    self._dim, self._dim * 16 / 1024**2,
                )
            except ImportError:
                logger.info(
                    "CuPy not available, falling back to NumPy CPU. "
                    "dim=%d", self._dim,
                )

        # Initialize state vector |00...0>
        self._state = self._xp.zeros(self._dim, dtype=self._dtype)
        self._state[0] = 1.0

        # Time tracking
        self._time = 0.0
        self._step_count = 0

        # DFS projector (for SLEDGE devices)
        self._dfs_projector = None
        if self.is_dfs:
            self._dfs_projector = DFSLogicalProjector(
                self.n_logical, device.connectivity,
            )

        # Leakage accumulator
        self._leakage_total = 0.0
        self._leakage_per_qubit = np.zeros(self.n_logical)

        # Noise generators
        self._charge_noise_gen = None
        if self.config.noise_enabled and device.noise_params:
            if device.noise_params.charge_noise_amplitude > 0:
                self._charge_noise_gen = _ChargeNoiseGeneratorSV(
                    n_qubits=self.n_logical,
                    amplitude=device.noise_params.charge_noise_amplitude,
                    dt=self.config.dt,
                    seed=self.config.seed + 1,
                )

        # Metrics history
        self._history = []

        logger.info(
            "StateVectorSimulator initialized: n_logical=%d, dim=%d, "
            "dfs=%s, gpu=%s, memory=%.2f MB",
            self.n_logical, self._dim, self.is_dfs, self._use_gpu,
            self._dim * (16 if self._dtype == np.complex128 else 8) / 1024**2,
        )

    # -- Properties ----------------------------------------------------

    @property
    def n_qubits(self) -> int:
        """Number of logical qubits."""
        return self.n_logical

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @property
    def state_vector(self) -> np.ndarray:
        """Current state vector (as NumPy array)."""
        if self._use_gpu:
            return self._xp.asnumpy(self._state)
        return self._state.copy()

    @property
    def leakage(self) -> float:
        """Total accumulated leakage probability."""
        return self._leakage_total

    @property
    def leakage_per_qubit(self) -> np.ndarray:
        """Per-qubit leakage probabilities."""
        return self._leakage_per_qubit.copy()

    # -- Compatibility properties for gym env --------------------------

    class _SVStateProxy:
        """Proxy object to provide bond_dims-like interface for gym env."""
        def __init__(self, sim):
            self._sim = sim

        @property
        def bond_dims(self):
            """State vector has no bond dimensions; return [dim]."""
            return [self._sim._dim]

        def norm(self):
            xp = self._sim._xp
            return float(xp.real(xp.sqrt(
                xp.sum(xp.abs(self._sim._state) ** 2)
            )))

    @property
    def state(self):
        """Proxy for MPS-like state interface (gym env compatibility)."""
        return self._SVStateProxy(self)

    # -- State initialization ------------------------------------------

    def reset(self, initial_state=None):
        """Reset the simulator to an initial state.

        Parameters
        ----------
        initial_state : ndarray or MPS, optional
            Initial state vector (2^n_logical). If MPS, converts to
            state vector. Defaults to |00...0>.
        """
        if initial_state is not None:
            if hasattr(initial_state, 'to_statevector'):
                sv = initial_state.to_statevector()
            elif hasattr(initial_state, '__len__') and len(initial_state) == self._dim:
                sv = np.asarray(initial_state, dtype=self._dtype)
            else:
                sv = np.zeros(self._dim, dtype=self._dtype)
                sv[0] = 1.0
            self._state = self._xp.asarray(sv, dtype=self._dtype)
        else:
            self._state = self._xp.zeros(self._dim, dtype=self._dtype)
            self._state[0] = 1.0

        self._time = 0.0
        self._step_count = 0
        self._leakage_total = 0.0
        self._leakage_per_qubit = np.zeros(self.n_logical)
        self._history = []

        if self._charge_noise_gen is not None:
            self._charge_noise_gen.reset(self.config.seed + 1)

    # -- Gate application (logical space) ------------------------------

    def _apply_single_qubit_gate(self, gate_2x2: np.ndarray, qubit: int):
        """Apply a 2x2 unitary gate to a single logical qubit.

        Uses efficient tensor reshaping:
            state reshaped to (2^qubit, 2, 2^(n-qubit-1))
            gate applied along axis 1

        Parameters
        ----------
        gate_2x2 : ndarray
            2x2 unitary matrix.
        qubit : int
            Target logical qubit index.
        """
        xp = self._xp
        n = self.n_logical

        # Convert gate to GPU if needed
        G = xp.asarray(gate_2x2, dtype=self._dtype)

        # Reshape state: (2^qubit, 2, 2^(n-qubit-1))
        dim_left = 2 ** qubit
        dim_right = 2 ** (n - qubit - 1)
        psi = xp.reshape(self._state, (dim_left, 2, dim_right))

        # Apply gate: psi_new[a, s', b] = sum_s G[s', s] * psi[a, s, b]
        psi_new = xp.einsum('ij,ajb->aib', G, psi)

        self._state = xp.reshape(psi_new, (self._dim,))

    def _apply_two_qubit_gate(
        self, gate_4x4: np.ndarray, qubit_i: int, qubit_j: int,
    ):
        """Apply a 4x4 unitary gate to two logical qubits.

        Handles both adjacent and non-adjacent qubits via SWAP
        decomposition for non-adjacent pairs.

        Parameters
        ----------
        gate_4x4 : ndarray
            4x4 unitary matrix.
        qubit_i, qubit_j : int
            Target logical qubit indices.
        """
        xp = self._xp
        n = self.n_logical

        # Ensure i < j
        if qubit_i > qubit_j:
            qubit_i, qubit_j = qubit_j, qubit_i
            # Swap the gate indices
            SWAP = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=self._dtype)
            gate_4x4 = SWAP @ gate_4x4 @ SWAP

        G = xp.asarray(gate_4x4.reshape(2, 2, 2, 2), dtype=self._dtype)

        if qubit_j == qubit_i + 1:
            # Adjacent qubits: efficient reshape
            dim_left = 2 ** qubit_i
            dim_right = 2 ** (n - qubit_j - 1)
            psi = xp.reshape(self._state, (dim_left, 2, 2, dim_right))
            psi_new = xp.einsum('ijkl,akld->aijd', G, psi)
            self._state = xp.reshape(psi_new, (self._dim,))
        else:
            # Non-adjacent: use general einsum
            # Reshape to (2, 2, ..., 2) - n indices
            psi = xp.reshape(self._state, tuple([2] * n))

            # Build einsum string for applying gate on qubits i, j
            # Input indices: a0 a1 ... a_{n-1}
            # Gate indices: b_i b_j a_i a_j
            # Output indices: a0 ... b_i ... b_j ... a_{n-1}
            in_idx = list(range(n))
            gate_out_i = n
            gate_out_j = n + 1
            gate_in_i = in_idx[qubit_i]
            gate_in_j = in_idx[qubit_j]

            out_idx = list(in_idx)
            out_idx[qubit_i] = gate_out_i
            out_idx[qubit_j] = gate_out_j

            # Build subscript string
            chars = 'abcdefghijklmnopqrstuvwxyz'
            if n + 2 > len(chars):
                chars = chars + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

            state_sub = ''.join(chars[i] for i in in_idx)
            gate_sub = (chars[gate_out_i] + chars[gate_out_j] +
                        chars[gate_in_i] + chars[gate_in_j])
            out_sub = ''.join(chars[i] for i in out_idx)

            subscripts = f"{gate_sub},{state_sub}->{out_sub}"
            psi_new = xp.einsum(subscripts, G, psi)
            self._state = xp.reshape(psi_new, (self._dim,))

    # -- Standard gate API (compatible with SiliQunSimulator) ----------

    def apply_single_gate(self, gate: np.ndarray, qubit: int):
        """Apply a 2x2 unitary gate to a single qubit."""
        self._apply_single_qubit_gate(gate, qubit)

        # Gate noise
        if self.config.noise_enabled and self.device.noise_params:
            p_err = self.device.noise_params.gate_error_rates.get("single", 0)
            if p_err > 0 and self.rng.random() < p_err:
                pauli_idx = self.rng.randint(1, 4)
                paulis = [
                    np.array([[0, 1], [1, 0]], dtype=np.complex128),
                    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
                    np.array([[1, 0], [0, -1]], dtype=np.complex128),
                ]
                self._apply_single_qubit_gate(paulis[pauli_idx - 1], qubit)

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit_i: int, qubit_j: int):
        """Apply a 4x4 unitary gate to two qubits."""
        self._apply_two_qubit_gate(gate, qubit_i, qubit_j)

        # Gate noise
        if self.config.noise_enabled and self.device.noise_params:
            p_err = self.device.noise_params.gate_error_rates.get("two", 0)
            if p_err > 0 and self.rng.random() < p_err:
                PX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
                PI = np.eye(2, dtype=np.complex128)
                err = np.kron(PX, PI)
                self._apply_two_qubit_gate(err, qubit_i, qubit_j)

    def apply_rx(self, theta: float, qubit: int):
        """Apply Rx(theta) rotation."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        self.apply_single_gate(gate, qubit)

    def apply_ry(self, theta: float, qubit: int):
        """Apply Ry(theta) rotation."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        self.apply_single_gate(gate, qubit)

    def apply_rz(self, theta: float, qubit: int):
        """Apply Rz(theta) rotation."""
        gate = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ], dtype=np.complex128)
        self.apply_single_gate(gate, qubit)

    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        gate = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.complex128)
        self.apply_two_qubit_gate(gate, control, target)
        self._time += self.device.gate_times.get("two", 100e-9)

    def apply_cz(self, qubit_i: int, qubit_j: int):
        """Apply CZ gate."""
        gate = np.diag([1, 1, 1, -1]).astype(np.complex128)
        self.apply_two_qubit_gate(gate, qubit_i, qubit_j)
        self._time += self.device.gate_times.get("two", 100e-9)

    def apply_sqrt_swap(self, qubit_i: int, qubit_j: int):
        """Apply sqrt(SWAP) gate."""
        gate = np.array([
            [1, 0, 0, 0],
            [0, 0.5 * (1 + 1j), 0.5 * (1 - 1j), 0],
            [0, 0.5 * (1 - 1j), 0.5 * (1 + 1j), 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)
        self.apply_two_qubit_gate(gate, qubit_i, qubit_j)
        self._time += self.device.gate_times.get("two", 100e-9)

    def apply_exchange(self, qubit_i: int, qubit_j: int, J: float, t: float):
        """Apply exchange interaction gate.

        For DFS-encoded devices, projects the physical exchange into
        the logical subspace and tracks leakage.

        Parameters
        ----------
        qubit_i, qubit_j : int
            Logical qubit indices.
        J : float
            Exchange coupling strength (angular frequency, rad/s).
        t : float
            Pulse duration (seconds).
        """
        theta = J * t  # Exchange angle

        if self.is_dfs and self._dfs_projector is not None:
            # Determine which physical spins are being exchanged
            # For SLEDGE: inter-qubit exchange couples spin 3 of qubit_i
            # to spin 1 of qubit_j (nearest physical spins)
            gate_4x4 = self._dfs_projector.logical_inter_qubit_exchange(
                theta, qubit_i, qubit_j,
                spin_a_local=2,  # spin 3 (0-indexed)
                spin_b_local=0,  # spin 1 (0-indexed)
            )
            self._apply_two_qubit_gate(gate_4x4, qubit_i, qubit_j)

            # Track leakage
            if self.config.leakage_tracking:
                leak = self._dfs_projector.compute_leakage_rate(
                    theta, qubit_i, qubit_j,
                    spin_a_local=2, spin_b_local=0,
                )
                self._leakage_total += leak
                self._leakage_per_qubit[qubit_i] += leak / 2
                self._leakage_per_qubit[qubit_j] += leak / 2
        else:
            # Non-DFS: use standard exchange gate
            from ..physics.dfs_encoding import partial_swap
            gate = partial_swap(theta)
            self.apply_two_qubit_gate(gate, qubit_i, qubit_j)

        self._time += t

    def apply_exchange_12(self, theta: float, qubit: int):
        """Apply intra-qubit J_12 exchange (logical Z rotation).

        Only valid for DFS-encoded devices.
        """
        if self.is_dfs and self._dfs_projector is not None:
            gate = self._dfs_projector.logical_exchange_12(theta)
            self._apply_single_qubit_gate(gate, qubit)
            self._time += self.device.gate_times.get("single", 10e-9)
        else:
            raise ValueError("apply_exchange_12 only valid for DFS devices")

    def apply_exchange_23(self, theta: float, qubit: int):
        """Apply intra-qubit J_23 exchange (logical X rotation).

        Only valid for DFS-encoded devices.
        """
        if self.is_dfs and self._dfs_projector is not None:
            gate = self._dfs_projector.logical_exchange_23(theta)
            self._apply_single_qubit_gate(gate, qubit)
            self._time += self.device.gate_times.get("single", 10e-9)
        else:
            raise ValueError("apply_exchange_23 only valid for DFS devices")

    def apply_esr(self, theta: float, phi: float, qubit: int):
        """Apply ESR rotation (donor qubits)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([
            [c, -1j * s * np.exp(-1j * phi)],
            [-1j * s * np.exp(1j * phi), c],
        ], dtype=np.complex128)
        self.apply_single_gate(gate, qubit)
        self._time += self.device.gate_times.get("single", 1e-6)

    def apply_edsr(self, theta: float, phi: float, qubit: int):
        """Apply EDSR rotation (SiMOS/GAA qubits)."""
        self.apply_esr(theta, phi, qubit)  # Same matrix, different timing
        self._time += self.device.gate_times.get("single", 200e-9)

    # -- Noise application ---------------------------------------------

    def apply_idle_noise(self, duration: float):
        """Apply decoherence during idle time.

        For state vector simulation, noise is applied as stochastic
        Kraus operators (quantum trajectory method).
        """
        if not self.config.noise_enabled:
            self._time += duration
            return

        noise_params = self.device.noise_params
        if noise_params is None:
            self._time += duration
            return

        n_steps = max(1, int(duration / self.config.dt))
        dt_actual = duration / n_steps

        for _ in range(n_steps):
            # T2* dephasing (dominant noise source)
            if noise_params.t2_star_times:
                for q in range(self.n_logical):
                    t2 = noise_params.t2_star_times[q]
                    if noise_params.dephasing_model == "gaussian":
                        # Gaussian decay: gamma = 1 - exp(-(dt/T2*)^2)
                        gamma = 1.0 - np.exp(-(dt_actual / t2) ** 2)
                    else:
                        # Exponential decay: gamma = 1 - exp(-dt/T2)
                        gamma = 1.0 - np.exp(-dt_actual / t2)

                    if self.is_dfs:
                        # DFS quadratic suppression
                        gamma = gamma ** 2

                    # Apply phase damping as stochastic Z rotation
                    if self.rng.random() < gamma:
                        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
                        self._apply_single_qubit_gate(Z, q)

            # T1 relaxation (very slow for Si, usually negligible)
            if noise_params.t1_times:
                for q in range(self.n_logical):
                    t1 = noise_params.t1_times[q]
                    gamma_t1 = 1.0 - np.exp(-dt_actual / t1)
                    if gamma_t1 > 0 and self.rng.random() < gamma_t1:
                        # Amplitude damping: project to |0>
                        proj = np.array([[1, 0], [0, 0]], dtype=np.complex128)
                        self._apply_single_qubit_gate(proj, q)
                        # Renormalize
                        norm = self._compute_norm()
                        if norm > 1e-15:
                            self._state /= norm

            # Crosstalk (1/r^3 capacitive coupling)
            if noise_params.crosstalk_amplitude > 0 and self.device.qubit_layout:
                self._apply_crosstalk_noise(dt_actual)

        self._time += duration

    def _apply_crosstalk_noise(self, dt: float):
        """Apply 1/r^3 crosstalk-induced dephasing."""
        noise_params = self.device.noise_params
        layout = self.device.qubit_layout

        for qi in range(self.n_logical):
            for qj in range(qi + 1, self.n_logical):
                xi, yi = layout[qi]
                xj, yj = layout[qj]
                r = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if r < 1e-10:
                    continue

                # Crosstalk coupling: A / r^alpha
                alpha = noise_params.crosstalk_decay_exponent
                coupling = noise_params.crosstalk_amplitude / (r ** alpha)

                # Phase accumulation
                phase = coupling * dt * 2 * np.pi
                if abs(phase) > 1e-12:
                    # Apply ZZ interaction
                    ZZ = np.diag([1, -1, -1, 1]).astype(np.complex128)
                    gate = np.diag(np.exp(1j * phase * np.array([1, -1, -1, 1]))).astype(np.complex128)
                    self._apply_two_qubit_gate(gate, qi, qj)

    # -- Observables ---------------------------------------------------

    def _compute_norm(self) -> float:
        """Compute the norm of the state vector."""
        xp = self._xp
        return float(xp.real(xp.sqrt(xp.sum(xp.abs(self._state) ** 2))))

    def expectation_z(self, qubit: int) -> float:
        """Compute <Z_i> for a single qubit."""
        xp = self._xp
        n = self.n_logical

        # Reshape to (2^qubit, 2, 2^(n-qubit-1))
        dim_left = 2 ** qubit
        dim_right = 2 ** (n - qubit - 1)
        psi = xp.reshape(self._state, (dim_left, 2, dim_right))

        # P(0) = sum |psi[a, 0, b]|^2
        p0 = float(xp.real(xp.sum(xp.abs(psi[:, 0, :]) ** 2)))
        p1 = float(xp.real(xp.sum(xp.abs(psi[:, 1, :]) ** 2)))

        return p0 - p1  # <Z> = P(0) - P(1)

    def expectation_zz(self, qubit_i: int, qubit_j: int) -> float:
        """Compute <Z_i Z_j> two-point correlator."""
        xp = self._xp
        n = self.n_logical

        # Reshape to (2, 2, ..., 2) - n indices
        psi = xp.reshape(self._state, tuple([2] * n))

        # Compute <ZZ> = sum_{s} (-1)^{s_i + s_j} |psi_s|^2
        result = 0.0
        for si in range(2):
            for sj in range(2):
                sign = (-1) ** (si + sj)
                # Select slice where qubit_i = si and qubit_j = sj
                slices = [slice(None)] * n
                slices[qubit_i] = si
                slices[qubit_j] = sj
                prob = float(xp.real(xp.sum(xp.abs(psi[tuple(slices)]) ** 2)))
                result += sign * prob

        return result

    def measure_qubit(self, qubit: int) -> int:
        """Projective measurement of a single qubit."""
        xp = self._xp
        n = self.n_logical

        # Compute P(0)
        dim_left = 2 ** qubit
        dim_right = 2 ** (n - qubit - 1)
        psi = xp.reshape(self._state, (dim_left, 2, dim_right))
        p0 = float(xp.real(xp.sum(xp.abs(psi[:, 0, :]) ** 2)))

        # Measurement error
        if self.config.noise_enabled and self.device.noise_params:
            fid = self.device.noise_params.measurement_fidelity
            p0_noisy = fid * p0 + (1 - fid) * (1 - p0)
        else:
            p0_noisy = p0

        # Sample outcome
        outcome = 0 if self.rng.random() < p0_noisy else 1

        # Collapse state
        psi_new = xp.zeros_like(psi)
        psi_new[:, outcome, :] = psi[:, outcome, :]
        self._state = xp.reshape(psi_new, (self._dim,))

        # Renormalize
        norm = self._compute_norm()
        if norm > 1e-15:
            self._state /= norm

        self._time += self.device.gate_times.get("readout", 10e-6)
        return outcome

    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        return [self.measure_qubit(q) for q in range(self.n_logical)]

    def compute_fidelity(self, target) -> float:
        """Compute fidelity |<target|current>|^2.

        Parameters
        ----------
        target : ndarray or MPS
            Target state. If MPS, converts to state vector first.
        """
        xp = self._xp

        if hasattr(target, 'to_statevector'):
            target_sv = target.to_statevector()
        elif hasattr(target, '__getitem__') and hasattr(target, 'bond_dims'):
            # MPS object - contract to state vector
            target_sv = self._mps_to_statevector(target)
        elif isinstance(target, np.ndarray) and target.shape == (self._dim,):
            target_sv = target
        else:
            # Try to use it directly
            target_sv = np.asarray(target).flatten()
            if len(target_sv) != self._dim:
                logger.warning(
                    "Target state dimension %d != %d, returning 0",
                    len(target_sv), self._dim,
                )
                return 0.0

        target_gpu = xp.asarray(target_sv, dtype=self._dtype)
        overlap = xp.sum(xp.conj(target_gpu) * self._state)
        fid = float(xp.real(xp.abs(overlap) ** 2))
        return max(0.0, min(1.0, fid))

    def _mps_to_statevector(self, mps) -> np.ndarray:
        """Convert an MPS to a state vector by full contraction."""
        n = len(mps.bond_dims) + 1 if hasattr(mps, 'bond_dims') else self.n_logical

        # Contract MPS tensors from left to right
        be = active_backend()
        result = be.to_numpy(mps[0])  # (1, d, chi_1)

        for i in range(1, n):
            A = be.to_numpy(mps[i])  # (chi_i, d, chi_{i+1})
            # Contract: result[..., chi_i] @ A[chi_i, d, chi_{i+1}]
            result = np.einsum('...i,ijk->...jk', result, A)

        # Final shape: (1, d, d, ..., d, 1) -> (d^n,)
        sv = result.reshape(-1)

        # Ensure correct dimension
        if len(sv) != self._dim:
            # Pad or truncate
            sv_full = np.zeros(self._dim, dtype=np.complex128)
            sv_full[:min(len(sv), self._dim)] = sv[:self._dim]
            return sv_full

        return sv

    def compute_entanglement_entropy(self, partition: int) -> float:
        """Compute von Neumann entanglement entropy across a bipartition.

        Parameters
        ----------
        partition : int
            Cut between qubit partition-1 and qubit partition.
        """
        xp = self._xp

        # Get state vector on CPU
        if self._use_gpu:
            psi = xp.asnumpy(self._state)
        else:
            psi = self._state

        # Reshape to (2^partition, 2^(n-partition))
        d_left = 2 ** partition
        d_right = 2 ** (self.n_logical - partition)
        psi_mat = psi.reshape(d_left, d_right)

        # SVD
        try:
            _, S, _ = np.linalg.svd(psi_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0

        # Entanglement entropy
        S2 = S ** 2
        S2 = S2[S2 > 1e-15]
        S2 = S2 / S2.sum()
        entropy = -np.sum(S2 * np.log2(S2 + 1e-30))
        return float(entropy)

    # -- Circuit execution ---------------------------------------------

    def execute_circuit(self, circuit: List[Tuple]) -> Dict:
        """Execute a sequence of gate operations.

        Parameters
        ----------
        circuit : list of tuple
            Each tuple is (gate_name, params, qubits).
        """
        gate_map = {
            "rx": lambda p, q: self.apply_rx(p["theta"], q[0]),
            "ry": lambda p, q: self.apply_ry(p["theta"], q[0]),
            "rz": lambda p, q: self.apply_rz(p["theta"], q[0]),
            "esr": lambda p, q: self.apply_esr(p["theta"], p["phi"], q[0]),
            "edsr": lambda p, q: self.apply_edsr(p["theta"], p["phi"], q[0]),
            "cnot": lambda p, q: self.apply_cnot(q[0], q[1]),
            "cz": lambda p, q: self.apply_cz(q[0], q[1]),
            "sqrt_swap": lambda p, q: self.apply_sqrt_swap(q[0], q[1]),
            "exchange": lambda p, q: self.apply_exchange(
                q[0], q[1], p["J"], p["t"],
            ),
            "exchange_12": lambda p, q: self.apply_exchange_12(p["theta"], q[0]),
            "exchange_23": lambda p, q: self.apply_exchange_23(p["theta"], q[0]),
            "idle": lambda p, q: self.apply_idle_noise(p["duration"]),
            "measure": lambda p, q: self.measure_qubit(q[0]),
        }

        measurements = []
        for gate_name, params, qubits in circuit:
            if gate_name not in gate_map:
                raise ValueError(f"Unknown gate: {gate_name}")
            result = gate_map[gate_name](params, qubits)
            if gate_name == "measure":
                measurements.append((qubits[0], result))

        return {
            "final_state": self.state_vector,
            "time": self._time,
            "measurements": measurements,
            "leakage": self._leakage_total,
        }

    # -- Snapshot and metrics ------------------------------------------

    def snapshot(self) -> Dict:
        """Take a snapshot of the current simulator state."""
        metrics = {
            "time": self._time,
            "step": self._step_count,
            "norm": self._compute_norm(),
            "bond_dims": [self._dim],  # Compatibility
            "max_bond": self._dim,
            "z_expectations": [
                self.expectation_z(q) for q in range(self.n_logical)
            ],
            "leakage": self._leakage_total,
            "leakage_per_qubit": self._leakage_per_qubit.tolist(),
            "backend": "gpu" if self._use_gpu else "cpu",
            "memory_mb": self._dim * 16 / 1024**2,
        }
        self._history.append(metrics)
        return metrics

    @property
    def history(self) -> List[Dict]:
        return self._history


# ======================================================================
# Charge noise generator for state vector simulation
# ======================================================================

class _ChargeNoiseGeneratorSV:
    """1/f charge noise generator for state vector simulation.

    Generates correlated noise samples with 1/f power spectral density.
    """

    def __init__(
        self,
        n_qubits: int,
        amplitude: float,
        dt: float,
        seed: int,
    ):
        self.n_qubits = n_qubits
        self.amplitude = amplitude
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        self._phase = np.zeros(n_qubits)

    def reset(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self._phase = np.zeros(self.n_qubits)

    def sample(self) -> np.ndarray:
        """Generate correlated 1/f noise samples."""
        # Simple 1/f noise approximation using random walk
        self._phase += self.rng.randn(self.n_qubits) * self.amplitude * np.sqrt(self.dt)
        return self._phase
