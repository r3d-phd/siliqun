"""
SiliQun Simulation Engine.

Drives the time evolution of silicon spin qubit systems using
Time-Evolving Block Decimation (TEBD) on MPS representations.
Supports both unitary (pure state) and noisy (quantum trajectory)
evolution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from ..backend import active_backend
from ..tensor.mps import MPS
from ..tensor.mpo import MPO
from ..physics.gates import (
    rx, ry, rz, cnot, cz, sqrt_swap,
    esr_rotation, edsr_rotation, exchange_gate,
    two_qubit_gate_to_mpo_tensors,
)
from ..physics.hamiltonian import DeviceParams, build_hamiltonian_mpo
from ..physics.noise.channels import (
    NoiseParams, ChargeNoiseGenerator,
    apply_t1_noise, apply_dephasing_noise,
    apply_charge_noise_dephasing,
)
from ..physics.devices.profiles import DeviceProfile


@dataclass
class SimConfig:
    """Configuration for the simulation engine.

    Parameters
    ----------
    dt : float
        Time step for Trotter decomposition (seconds).
    max_bond_dim : int
        Maximum MPS bond dimension (controls accuracy vs memory).
    svd_cutoff : float
        Singular value cutoff for truncation.
    n_trotter_steps : int
        Number of Trotter steps per gate application.
    noise_enabled : bool
        Whether to apply noise channels.
    n_trajectories : int
        Number of quantum trajectories for noise averaging.
    charge_noise_enabled : bool
        Whether to include 1/f charge noise.
    seed : int
        Random seed for reproducibility.
    """
    dt: float = 1e-9
    max_bond_dim: int = 64
    svd_cutoff: float = 1e-12
    n_trotter_steps: int = 10
    noise_enabled: bool = True
    n_trajectories: int = 1
    charge_noise_enabled: bool = True
    seed: int = 42


class SiliQunSimulator:
    """Main simulation engine for silicon spin qubit systems.

    Manages the quantum state (MPS), applies gates and noise,
    and computes observables.

    Parameters
    ----------
    device : DeviceProfile
        Physical device specification.
    config : SimConfig
        Simulation configuration.
    """

    def __init__(self, device: DeviceProfile, config: Optional[SimConfig] = None):
        self.device = device
        self.config = config or SimConfig()
        self.be = active_backend()
        self.rng = np.random.RandomState(self.config.seed)

        # Initialize quantum state
        self._state: MPS = MPS.computational_basis(
            device.n_qubits, state=0
        )
        self._time: float = 0.0
        self._step_count: int = 0

        # Noise generators
        self._charge_noise: Optional[ChargeNoiseGenerator] = None
        if self.config.charge_noise_enabled and self.config.noise_enabled:
            self._charge_noise = ChargeNoiseGenerator(
                n_qubits=device.n_qubits,
                amplitude=device.noise_params.charge_noise_amplitude,
                dt=self.config.dt,
                seed=self.config.seed + 1,
            )

        # Hamiltonian MPO (cached)
        self._H_mpo: Optional[MPO] = None

        # Metrics history
        self._history: List[Dict] = []

    # ── State access ────────────────────────────────────────────────

    @property
    def state(self) -> MPS:
        """Current quantum state as MPS."""
        return self._state

    @state.setter
    def state(self, new_state: MPS):
        self._state = new_state

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @property
    def n_qubits(self) -> int:
        return self.device.n_qubits

    # ── State initialization ────────────────────────────────────────

    def reset(self, initial_state: Optional[MPS] = None):
        """Reset the simulator to an initial state.

        Parameters
        ----------
        initial_state : MPS, optional
            Custom initial state. Defaults to |00...0⟩.
        """
        if initial_state is not None:
            self._state = initial_state.copy()
        else:
            self._state = MPS.computational_basis(self.n_qubits, state=0)
        self._time = 0.0
        self._step_count = 0
        self._history = []
        if self._charge_noise is not None:
            self._charge_noise.reset(self.config.seed + 1)

    # ── Single-qubit gate application ───────────────────────────────

    def apply_single_gate(self, gate: np.ndarray, qubit: int):
        """Apply a 2×2 unitary gate to a single qubit.

        Parameters
        ----------
        gate : ndarray
            2×2 unitary matrix.
        qubit : int
            Target qubit index.
        """
        be = self.be
        A = self._state[qubit]  # (χ_l, d, χ_r)
        G = be.array(gate)
        # Contract: new_A[a,s',b] = Σ_s G[s',s] A[a,s,b]
        new_A = be.einsum("ij,ajb->aib", G, A)
        self._state[qubit] = new_A

        # Apply gate noise if enabled
        if self.config.noise_enabled:
            p_err = self.device.noise_params.gate_error_rates.get("single", 0)
            if p_err > 0 and self.rng.random() < p_err:
                # Apply random Pauli error
                pauli_idx = self.rng.randint(1, 4)
                from ..physics import gates as g
                paulis = [g.PAULI_X, g.PAULI_Y, g.PAULI_Z]
                self.apply_single_gate(paulis[pauli_idx - 1], qubit)

    def apply_rx(self, theta: float, qubit: int):
        """Apply Rx(θ) rotation to a qubit."""
        self.apply_single_gate(rx(theta), qubit)

    def apply_ry(self, theta: float, qubit: int):
        """Apply Ry(θ) rotation to a qubit."""
        self.apply_single_gate(ry(theta), qubit)

    def apply_rz(self, theta: float, qubit: int):
        """Apply Rz(θ) rotation to a qubit."""
        self.apply_single_gate(rz(theta), qubit)

    def apply_esr(self, theta: float, phi: float, qubit: int):
        """Apply ESR rotation (donor qubits)."""
        self.apply_single_gate(esr_rotation(theta, phi), qubit)
        self._time += self.device.gate_times.get("single", 1e-6)

    def apply_edsr(self, theta: float, phi: float, qubit: int):
        """Apply EDSR rotation (SiMOS/GAA qubits)."""
        self.apply_single_gate(edsr_rotation(theta, phi), qubit)
        self._time += self.device.gate_times.get("single", 200e-9)

    # ── Two-qubit gate application ──────────────────────────────────

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit_i: int, qubit_j: int):
        """Apply a 4×4 unitary gate to two adjacent qubits.

        Uses SVD decomposition to maintain MPS form with bounded
        bond dimension.

        Parameters
        ----------
        gate : ndarray
            4×4 unitary matrix.
        qubit_i, qubit_j : int
            Target qubit indices (must be adjacent: |i-j|=1).
        """
        be = self.be
        assert abs(qubit_i - qubit_j) == 1, \
            f"Two-qubit gates require adjacent qubits, got {qubit_i} and {qubit_j}"

        # Ensure i < j
        if qubit_i > qubit_j:
            qubit_i, qubit_j = qubit_j, qubit_i
            # Swap gate indices: SWAP · G · SWAP
            from ..physics.gates import swap as swap_gate
            S = swap_gate()
            gate = S @ gate @ S

        A_i = self._state[qubit_i]  # (χ_l, d, χ_m)
        A_j = self._state[qubit_j]  # (χ_m, d, χ_r)

        chi_l = A_i.shape[0]
        d = A_i.shape[1]
        chi_r = A_j.shape[2]

        # Contract the two site tensors
        # θ[a, s_i, s_j, b] = Σ_m A_i[a, s_i, m] A_j[m, s_j, b]
        theta = be.einsum("asm,mtb->astb", A_i, A_j)

        # Apply the gate
        # θ'[a, s_i', s_j', b] = Σ_{s_i,s_j} G[s_i's_j', s_i s_j] θ[a, s_i, s_j, b]
        G = be.array(gate.reshape(d, d, d, d))
        theta_new = be.einsum("ijkl,akld->aijd", G, theta)

        # SVD to split back into two tensors
        theta_mat = be.reshape(theta_new, (chi_l * d, d * chi_r))
        U, S_vals, Vh = be.svd(theta_mat, full_matrices=False)

        # Truncate to max_bond_dim
        chi_new = min(len(be.to_numpy(S_vals)), self.config.max_bond_dim)

        # Apply SVD cutoff
        s_np = be.to_numpy(S_vals)
        mask = s_np > self.config.svd_cutoff
        chi_new = min(chi_new, int(mask.sum()))
        chi_new = max(chi_new, 1)

        U = U[:, :chi_new]
        S_vals = S_vals[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb singular values into U (left-canonical)
        US = U * S_vals[None, :]

        # Reshape back to MPS tensors
        new_A_i = be.reshape(US, (chi_l, d, chi_new))
        new_A_j = be.reshape(Vh, (chi_new, d, chi_r))

        self._state[qubit_i] = new_A_i
        self._state[qubit_j] = new_A_j

        # Apply gate noise if enabled
        if self.config.noise_enabled:
            p_err = self.device.noise_params.gate_error_rates.get("two", 0)
            if p_err > 0 and self.rng.random() < p_err:
                from ..physics.gates import PAULI_X, PAULI_I
                err = np.kron(PAULI_X, PAULI_I)
                self.apply_two_qubit_gate(err, qubit_i, qubit_j)

    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        self.apply_two_qubit_gate(cnot(), control, target)
        self._time += self.device.gate_times.get("two", 100e-9)

    def apply_cz(self, qubit_i: int, qubit_j: int):
        """Apply CZ gate."""
        self.apply_two_qubit_gate(cz(), qubit_i, qubit_j)
        self._time += self.device.gate_times.get("two", 100e-9)

    def apply_sqrt_swap(self, qubit_i: int, qubit_j: int):
        """Apply √SWAP gate (native to exchange-coupled spin qubits)."""
        self.apply_two_qubit_gate(sqrt_swap(), qubit_i, qubit_j)
        self._time += self.device.gate_times.get("two", 100e-9)

    def apply_exchange(self, qubit_i: int, qubit_j: int, J: float, t: float):
        """Apply exchange interaction gate with specified coupling and time."""
        gate = exchange_gate(J, t)
        self.apply_two_qubit_gate(gate, qubit_i, qubit_j)
        self._time += t

    # ── Noise application ───────────────────────────────────────────

    def apply_idle_noise(self, duration: float):
        """Apply decoherence during idle time.

        Applies T1 relaxation, T2 dephasing, and charge noise
        for the specified duration.
        """
        if not self.config.noise_enabled:
            self._time += duration
            return

        n_steps = max(1, int(duration / self.config.dt))
        dt_actual = duration / n_steps

        for _ in range(n_steps):
            # T1 relaxation
            if self.device.noise_params.t1_times:
                for q in range(self.n_qubits):
                    t1 = self.device.noise_params.t1_times[q]
                    self._state = apply_t1_noise(
                        self._state, q, dt_actual, t1
                    )

            # T2 dephasing
            if self.device.noise_params.t2_star_times:
                for q in range(self.n_qubits):
                    t2 = self.device.noise_params.t2_star_times[q]
                    self._state = apply_dephasing_noise(
                        self._state, q, dt_actual, t2
                    )

            # 1/f charge noise
            if self._charge_noise is not None:
                noise_vals = self._charge_noise.sample()
                self._state = apply_charge_noise_dephasing(
                    self._state, noise_vals, dt_actual
                )

        self._time += duration

    # ── Observables ─────────────────────────────────────────────────

    def measure_qubit(self, qubit: int) -> int:
        """Projective measurement of a single qubit.

        Returns 0 or 1, and collapses the state.
        """
        be = self.be
        A = self._state[qubit]  # (χ_l, d, χ_r)

        # Compute probability of |0⟩
        # P(0) = Tr(|0⟩⟨0| ρ_qubit)
        # For MPS: contract environment tensors
        p0 = self._compute_local_probability(qubit, 0)
        p1 = 1.0 - p0

        # Apply measurement error
        if self.config.noise_enabled:
            fid = self.device.noise_params.measurement_fidelity
            p0_noisy = fid * p0 + (1 - fid) * p1
            p1_noisy = 1.0 - p0_noisy
        else:
            p0_noisy, p1_noisy = p0, p1

        # Sample outcome
        outcome = 0 if self.rng.random() < p0_noisy else 1

        # Collapse state
        proj = np.zeros((2, 2), dtype=np.complex128)
        proj[outcome, outcome] = 1.0
        self.apply_single_gate(proj, qubit)

        # Renormalize
        norm = self._state.norm()
        if norm > 1e-15:
            self._state[0] = be.array(
                be.to_numpy(self._state[0]) / norm
            )

        self._time += self.device.gate_times.get("readout", 10e-6)
        return outcome

    def measure_all(self) -> List[int]:
        """Measure all qubits. Returns list of outcomes."""
        return [self.measure_qubit(q) for q in range(self.n_qubits)]

    def _compute_local_probability(self, qubit: int, outcome: int) -> float:
        """Compute probability of a specific outcome on one qubit."""
        be = self.be

        # Build left environment
        left_env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(qubit):
            A = self._state[i]
            A_conj = be.conj(A)
            # Contract: env[a',a] * A[a,s,b] * A*[a',s,b']
            left_env = be.einsum("xy,xsb,ysa->ba", left_env, A, A_conj)

        # Site tensor with projection
        A_q = self._state[qubit]
        A_q_conj = be.conj(A_q)
        proj = be.zeros((2,))
        proj_np = np.zeros(2)
        proj_np[outcome] = 1.0
        proj = be.array(proj_np)
        # P = env_L * A[a,s,b] * proj[s] * A*[a',s',b'] * proj[s'] * env_R
        mid = be.einsum("xy,xsb,s,ysa->ba", left_env, A_q, proj, A_q_conj)

        # Build right environment
        right_env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(self.n_qubits - 1, qubit, -1):
            A = self._state[i]
            A_conj = be.conj(A)
            right_env = be.einsum("asb,asc,bc->ac", A, A_conj, right_env)

        # Contract mid with right_env
        prob = float(np.real(be.to_numpy(be.einsum("ab,ab->", mid, right_env))))
        return max(0.0, min(1.0, prob))

    def expectation_z(self, qubit: int) -> float:
        """Compute ⟨Z_i⟩ for a single qubit."""
        p0 = self._compute_local_probability(qubit, 0)
        return 2 * p0 - 1  # ⟨Z⟩ = P(0) - P(1)

    def expectation_zz(self, qubit_i: int, qubit_j: int) -> float:
        """Compute ⟨Z_i Z_j⟩ two-point correlator."""
        be = self.be
        Z = be.array(np.array([[1, 0], [0, -1]], dtype=np.complex128))

        # Build full contraction with Z insertions at sites i and j
        env = be.array(np.ones((1, 1), dtype=np.complex128))
        for k in range(self.n_qubits):
            A = self._state[k]
            A_conj = be.conj(A)
            if k == qubit_i or k == qubit_j:
                env = be.einsum(
                    "xy,xsb,st,yta->ba", env, A, Z, A_conj
                )
            else:
                env = be.einsum("xy,xsb,ysa->ba", env, A, A_conj)

        return float(np.real(be.to_numpy(env[0, 0])))

    def compute_fidelity(self, target: MPS) -> float:
        """Compute fidelity |⟨ψ_target|ψ_current⟩|² between current
        state and a target state."""
        be = self.be
        overlap = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(self.n_qubits):
            A = self._state[i]
            B = target[i]
            B_conj = be.conj(B)
            overlap = be.einsum("xy,xsa,ysb->ab", overlap, A, B_conj)
        fid = float(np.abs(be.to_numpy(overlap[0, 0])) ** 2)
        return fid

    def compute_entanglement_entropy(self, partition: int) -> float:
        """Compute von Neumann entanglement entropy across a bipartition.

        Parameters
        ----------
        partition : int
            Cut between site partition-1 and site partition.
        """
        be = self.be
        # Build the reduced density matrix by contracting left part
        # and computing singular values at the cut
        left = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(partition):
            A = self._state[i]
            A_conj = be.conj(A)
            left = be.einsum("xy,xsa,ysb->ab", left, A, A_conj)

        # The eigenvalues of the reduced density matrix are the
        # squares of the singular values at the bond
        rho_L = be.to_numpy(left)
        eigenvalues = np.linalg.eigvalsh(rho_L)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        eigenvalues = eigenvalues / eigenvalues.sum()

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-30))
        return float(entropy)

    # ── Circuit execution ───────────────────────────────────────────

    def execute_circuit(self, circuit: List[Tuple]) -> Dict:
        """Execute a sequence of gate operations.

        Parameters
        ----------
        circuit : list of tuple
            Each tuple is (gate_name, params, qubits).
            e.g., [("rx", {"theta": np.pi/2}, [0]),
                   ("cnot", {}, [0, 1])]

        Returns
        -------
        dict with keys: final_state, fidelity_history, measurements
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
                q[0], q[1], p["J"], p["t"]
            ),
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
            "final_state": self._state,
            "time": self._time,
            "measurements": measurements,
            "bond_dims": self._state.bond_dims,
        }

    # ── Snapshot and metrics ────────────────────────────────────────

    def snapshot(self) -> Dict:
        """Take a snapshot of the current simulator state."""
        metrics = {
            "time": self._time,
            "step": self._step_count,
            "norm": self._state.norm(),
            "bond_dims": list(self._state.bond_dims),
            "max_bond": max(self._state.bond_dims) if self._state.bond_dims else 1,
            "z_expectations": [
                self.expectation_z(q) for q in range(self.n_qubits)
            ],
        }
        self._history.append(metrics)
        return metrics

    @property
    def history(self) -> List[Dict]:
        return self._history
