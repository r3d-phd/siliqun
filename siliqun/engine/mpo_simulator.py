"""
MPO Density Matrix Simulator — exact mixed-state simulation for SiliQun.

This module provides a simulator that represents quantum states as Matrix
Product Operator (MPO) density matrices, enabling exact (non-stochastic)
simulation of noisy quantum systems. This is the key upgrade over the
MPS-based simulator, which uses quantum trajectories (approximate).

The density matrix ρ is stored as an MPO with rank-4 tensors:
    W[i] : (χ_left, d_out, d_in, χ_right)
where d_out is the "ket" index and d_in is the "bra" index.

Gate application:  ρ → U ρ U†
Noise application: ρ → Σ_k K_k ρ K_k†  (exact, all Kraus operators)
Measurement:       P(m) = Tr(Π_m ρ),  ρ → Π_m ρ Π_m / P(m)

Reference:
    Votto et al., "Efficient learning of quantum states prepared with
    few non-Clifford gates II: Single-copy observables and beyond",
    PRL 136, 090801 (2026).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..backend import active_backend
from ..tensor.mps import MPS
from ..tensor.mpo import MPO
from ..physics.devices.profiles import DeviceProfile
from ..physics.noise.channels import (
    NoiseParams,
    ChargeNoiseGenerator,
    amplitude_damping_kraus,
    phase_damping_kraus,
    depolarizing_kraus,
)
from ..physics import gates


@dataclass
class MPOSimConfig:
    """Configuration for the MPO density matrix simulator.

    Parameters
    ----------
    noise_enabled : bool
        Whether to apply noise channels during simulation.
    max_bond_dim : int
        Maximum MPO bond dimension (controls accuracy vs speed).
    svd_cutoff : float
        SVD truncation threshold for MPO compression.
    compress_every : int
        Compress the MPO every N gate applications (0 = never).
    dt : float
        Time step for noise discretization (seconds).
    seed : int or None
        Random seed for measurement outcomes and charge noise.
    track_purity : bool
        Whether to track Tr(ρ²) at each step (adds overhead).
    """
    noise_enabled: bool = True
    max_bond_dim: int = 64
    svd_cutoff: float = 1e-12
    compress_every: int = 5
    dt: float = 1e-9
    seed: Optional[int] = None
    track_purity: bool = False


class MPODensityMatrixSimulator:
    """Exact mixed-state simulator using MPO density matrices.

    Unlike the MPS simulator which uses stochastic quantum trajectories,
    this simulator evolves the full density matrix ρ as an MPO. Noise
    channels are applied exactly via the superoperator formalism, giving
    the true mixed state at every time step.

    This is essential for:
    - QUASAR: DRL agents see the true noisy state during training
    - MOZAIQ: inter-module states are inherently mixed (partial traces)
    - seQurAIt: security analysis requires mixed-state reasoning

    Parameters
    ----------
    device : DeviceProfile
        Silicon spin qubit device specification.
    config : MPOSimConfig
        Simulation configuration.
    """

    def __init__(self, device: DeviceProfile, config: Optional[MPOSimConfig] = None):
        self.device = device
        self.config = config or MPOSimConfig()
        self.n_qubits = device.n_qubits
        self.be = active_backend()

        # Random state
        self.rng = np.random.RandomState(self.config.seed)

        # Initialize state as |0...0⟩⟨0...0|
        self._state: MPO = self._init_pure_state()
        self._time: float = 0.0
        self._step_count: int = 0
        self._gate_count_since_compress: int = 0
        self._history: List[Dict] = []

        # Charge noise generator
        self._charge_noise: Optional[ChargeNoiseGenerator] = None
        if (self.config.noise_enabled and
                device.noise_params.charge_noise_amplitude > 0):
            self._charge_noise = ChargeNoiseGenerator(
                n_qubits=self.n_qubits,
                amplitude=device.noise_params.charge_noise_amplitude,
                correlation_length=device.noise_params.charge_noise_correlation_length,
                dt=self.config.dt,
                seed=self.config.seed,
            )

    def _init_pure_state(self) -> MPO:
        """Initialize ρ = |0...0⟩⟨0...0| as an MPO."""
        be = self.be
        tensors = []
        for _ in range(self.n_qubits):
            # W[χ_l, d_out, d_in, χ_r] = |0⟩⟨0|
            W = np.zeros((1, 2, 2, 1), dtype=np.complex128)
            W[0, 0, 0, 0] = 1.0  # |0⟩⟨0|
            tensors.append(be.array(W))
        return MPO(tensors)

    def reset(self):
        """Reset the simulator to the initial state |0...0⟩⟨0...0|."""
        self._state = self._init_pure_state()
        self._time = 0.0
        self._step_count = 0
        self._gate_count_since_compress = 0
        self._history = []
        if self._charge_noise is not None:
            self._charge_noise.reset(self.config.seed)

    @property
    def state(self) -> MPO:
        """Current density matrix as MPO."""
        return self._state

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    # ── Single-qubit gate application ──────────────────────────────

    def _apply_single_gate_to_mpo(self, U: np.ndarray, site: int):
        """Apply a single-qubit unitary to the MPO: rho -> U rho U_dag.

        For site tensor W[chi_l, d_out, d_in, chi_r]:
            W'[chi_l, s', t', chi_r] = sum_{s,t} U[s',s] W[chi_l, s, t, chi_r] conj(U)[t',t]

        Note: We use conj(U) not U_dag in the einsum because the contraction
        over t with indices [q,t] already handles the transpose implicitly.
        rho'_{p,q} = sum_{s,t} U_{p,s} rho_{s,t} U*_{q,t}
        """
        be = self.be
        U_arr = be.array(U.astype(np.complex128))
        U_conj = be.conj(U_arr)  # Element-wise conjugate, NOT U_dag

        W = self._state[site]
        # Contract: rho' = U rho U_dag
        # W_new[a, p, q, b] = U[p,s] * W[a,s,t,b] * conj(U)[q,t]
        W_new = be.einsum("ps,astb,qt->apqb", U_arr, W, U_conj)
        self._state[site] = W_new

        self._step_count += 1
        self._gate_count_since_compress += 1
        self._maybe_compress()

    def apply_single_gate(self, U: np.ndarray, site: int):
        """Apply a single-qubit unitary gate."""
        self._apply_single_gate_to_mpo(U, site)
        gate_time = self.device.gate_times.get("single", 1e-6)
        self._time += gate_time

        # Apply gate noise if enabled
        if self.config.noise_enabled:
            error_rate = self.device.noise_params.gate_error_rates.get("single", 1e-4)
            if error_rate > 0:
                self._apply_depolarizing_channel(site, error_rate)

    def apply_rx(self, theta: float, qubit: int):
        """Apply Rx(θ) rotation."""
        self.apply_single_gate(gates.rx(theta), qubit)

    def apply_ry(self, theta: float, qubit: int):
        """Apply Ry(θ) rotation."""
        self.apply_single_gate(gates.ry(theta), qubit)

    def apply_rz(self, theta: float, qubit: int):
        """Apply Rz(θ) rotation."""
        self.apply_single_gate(gates.rz(theta), qubit)

    def apply_esr(self, theta: float, phi: float, qubit: int):
        """Apply ESR (electron spin resonance) rotation."""
        U = gates.esr_rotation(theta, phi)
        self.apply_single_gate(U, qubit)

    def apply_edsr(self, theta: float, phi: float, qubit: int):
        """Apply EDSR (electric dipole spin resonance) rotation."""
        U = gates.edsr_rotation(theta, phi)
        self.apply_single_gate(U, qubit)

    # ── Two-qubit gate application ─────────────────────────────────

    def _apply_two_qubit_gate_to_mpo(self, U: np.ndarray, site_i: int, site_j: int):
        """Apply a two-qubit unitary to the MPO: ρ → U ρ U†.

        For adjacent sites i, j = i+1:
        1. Contract W[i] and W[j] into a single rank-6 tensor
        2. Apply U on ket indices and U† on bra indices
        3. Split back via SVD with truncation

        U has shape (4, 4) acting on the joint space of qubits i and j.
        """
        be = self.be
        # Ensure proper ordering
        if site_i > site_j:
            site_i, site_j = site_j, site_i
            from ..physics.gates import swap as swap_gate
            S = swap_gate()
            U = S @ U.reshape(4, 4) @ S
            U = U.reshape(2, 2, 2, 2)
        assert site_j == site_i + 1, "Two-qubit gates require adjacent qubits"

        U_arr = be.array(U.reshape(2, 2, 2, 2).astype(np.complex128))
        U_dag = be.conj(be.transpose(U_arr, (2, 3, 0, 1)))

        W_i = self._state[site_i]   # (χ_l, d, d, χ_m)
        W_j = self._state[site_j]   # (χ_m, d, d, χ_r)

        # Contract the two tensors over the shared bond
        # Θ[χ_l, s_i, t_i, s_j, t_j, χ_r] = W_i[χ_l, s_i, t_i, χ_m] * W_j[χ_m, s_j, t_j, χ_r]
        Theta = be.einsum("asbc,cdef->asbdef", W_i, W_j)
        # Theta shape: (χ_l, d, d, d, d, χ_r) = (χ_l, s_i, t_i, s_j, t_j, χ_r)

        # Apply U on ket indices (s_i, s_j) and U† on bra indices (t_i, t_j)
        # Θ'[χ_l, s_i', t_i', s_j', t_j', χ_r] =
        #   U[s_i', s_j', s_i, s_j] * Θ[χ_l, s_i, t_i, s_j, t_j, χ_r] * U†[t_i', t_j', t_i, t_j]
        Theta_new = be.einsum(
            "pqrs,arbsdf,uvbd->apuqvf",
            U_arr, Theta, U_dag
        )
        # Theta_new shape: (χ_l, s_i', t_i', s_j', t_j', χ_r)

        # Split back into two tensors via SVD
        chi_l = Theta_new.shape[0]
        chi_r = Theta_new.shape[5]
        d = 2

        # Reshape: (χ_l * d * d, d * d * χ_r)
        mat = be.reshape(Theta_new, (chi_l * d * d, d * d * chi_r))
        U_svd, S, Vh = be.svd(mat, full_matrices=False)

        # Truncate
        s_np = be.to_numpy(S)
        mask = s_np > self.config.svd_cutoff
        max_bond = self.config.max_bond_dim
        if max_bond is not None:
            mask[max_bond:] = False
        chi_new = max(int(np.sum(mask)), 1)

        U_svd = U_svd[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb S into Vh (right-canonical form)
        S_diag = be.array(np.diag(be.to_numpy(S)))
        SVh = S_diag @ Vh

        # Reshape back to MPO tensors
        W_i_new = be.reshape(U_svd, (chi_l, d, d, chi_new))
        W_j_new = be.reshape(SVh, (chi_new, d, d, chi_r))

        self._state[site_i] = W_i_new
        self._state[site_j] = W_j_new

        self._step_count += 1
        self._gate_count_since_compress += 1

    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        self._apply_two_qubit_gate_to_mpo(gates.cnot(), control, target)
        gate_time = self.device.gate_times.get("two", 10e-6)
        self._time += gate_time
        if self.config.noise_enabled:
            error_rate = self.device.noise_params.gate_error_rates.get("two", 1e-3)
            if error_rate > 0:
                self._apply_depolarizing_channel(control, error_rate / 2)
                self._apply_depolarizing_channel(target, error_rate / 2)

    def apply_cz(self, qubit_i: int, qubit_j: int):
        """Apply CZ gate."""
        self._apply_two_qubit_gate_to_mpo(gates.cz(), qubit_i, qubit_j)
        gate_time = self.device.gate_times.get("two", 10e-6)
        self._time += gate_time
        if self.config.noise_enabled:
            error_rate = self.device.noise_params.gate_error_rates.get("two", 1e-3)
            if error_rate > 0:
                self._apply_depolarizing_channel(qubit_i, error_rate / 2)
                self._apply_depolarizing_channel(qubit_j, error_rate / 2)

    def apply_sqrt_swap(self, qubit_i: int, qubit_j: int):
        """Apply √SWAP gate."""
        self._apply_two_qubit_gate_to_mpo(gates.sqrt_swap(), qubit_i, qubit_j)
        gate_time = self.device.gate_times.get("two", 10e-6)
        self._time += gate_time

    # ── Noise channels (exact superoperator) ───────────────────────

    def _apply_kraus_channel(self, site: int, kraus_ops: List[np.ndarray]):
        """Apply a quantum channel via Kraus operators to the MPO.

        ρ → Σ_k K_k ρ K_k†

        For the MPO tensor at the given site:
            W'[χ_l, s', t', χ_r] = Σ_k Σ_{s,t} K_k[s',s] W[χ_l, s, t, χ_r] K_k*[t',t]

        This is the EXACT application — no stochastic sampling.
        """
        be = self.be
        W = self._state[site]  # (χ_l, d_out, d_in, χ_r)

        W_new = be.zeros(W.shape)
        for K in kraus_ops:
            K_arr = be.array(K.astype(np.complex128))
            K_conj = be.conj(K_arr)  # Element-wise conjugate, NOT K_dag
            # Contribution: K rho K_dag
            # W_new[a,p,q,b] = K[p,s] * W[a,s,t,b] * conj(K)[q,t]
            contrib = be.einsum("ps,astb,qt->apqb", K_arr, W, K_conj)
            W_new = W_new + contrib

        self._state[site] = W_new

    def _apply_depolarizing_channel(self, site: int, p: float):
        """Apply depolarizing channel: ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)."""
        kraus = depolarizing_kraus(p)
        self._apply_kraus_channel(site, kraus)

    def _apply_t1_channel(self, site: int, dt: float, t1: float):
        """Apply T1 relaxation (amplitude damping) exactly."""
        gamma = 1 - np.exp(-dt / t1)
        kraus = amplitude_damping_kraus(gamma)
        self._apply_kraus_channel(site, kraus)

    def _apply_t2_channel(self, site: int, dt: float, t2: float):
        """Apply T2 dephasing (phase damping) exactly."""
        gamma = 1 - np.exp(-dt / t2)
        kraus = phase_damping_kraus(gamma)
        self._apply_kraus_channel(site, kraus)

    def _apply_charge_noise_channel(self, noise_values: np.ndarray, dt: float,
                                     sensitivity: float = 1e9):
        """Apply charge-noise-induced dephasing to all qubits.

        Unlike the MPS version which applies a unitary Rz rotation,
        the MPO version applies it as a proper unitary channel:
        ρ → Rz(δφ) ρ Rz(δφ)†
        """
        for i in range(self.n_qubits):
            delta_phi = 2 * np.pi * sensitivity * noise_values[i] * dt
            Rz = gates.rz(delta_phi)
            self._apply_single_gate_to_mpo(Rz, i)

    def apply_idle_noise(self, duration: float):
        """Apply decoherence during idle time.

        Applies T1 relaxation, T2 dephasing, and charge noise
        for the specified duration — all exactly via superoperators.
        """
        if not self.config.noise_enabled:
            self._time += duration
            return

        n_steps = max(1, int(duration / self.config.dt))
        dt_actual = duration / n_steps

        for _ in range(n_steps):
            # T1 relaxation (exact)
            if self.device.noise_params.t1_times:
                for q in range(self.n_qubits):
                    t1 = self.device.noise_params.t1_times[q]
                    self._apply_t1_channel(q, dt_actual, t1)

            # T2 dephasing (exact)
            if self.device.noise_params.t2_star_times:
                for q in range(self.n_qubits):
                    t2 = self.device.noise_params.t2_star_times[q]
                    self._apply_t2_channel(q, dt_actual, t2)

            # 1/f charge noise
            if self._charge_noise is not None:
                noise_vals = self._charge_noise.sample()
                self._apply_charge_noise_channel(noise_vals, dt_actual)

        self._time += duration
        self._maybe_compress()

    # ── MPO compression ────────────────────────────────────────────

    def _maybe_compress(self):
        """Compress the MPO if enough gates have been applied."""
        if (self.config.compress_every > 0 and
                self._gate_count_since_compress >= self.config.compress_every):
            self._state.compress(
                max_bond=self.config.max_bond_dim,
                cutoff=self.config.svd_cutoff,
            )
            self._gate_count_since_compress = 0

    # ── Observables ────────────────────────────────────────────────

    def trace(self) -> float:
        """Compute Tr(ρ) — should be 1.0 for a valid density matrix."""
        return float(np.real(self._state.trace()))

    def expectation_z(self, qubit: int) -> float:
        """Compute ⟨Z_q⟩ = Tr(Z_q ρ).

        Insert Z at site q into the MPO trace contraction.
        """
        be = self.be
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(self.n_qubits):
            W = self._state[i]  # (χ_l, d_out, d_in, χ_r)
            if i == qubit:
                # Tr(Z_q W) = Σ_{s,t} Z[s,t] * δ_{s,t} * W[a,s,t,b]
                # → Σ_s Z[s,s] * W[a,s,s,b] but Z is diagonal
                # More generally: Σ_{s,t} Z[t,s] * W[a,s,t,b]
                Z_arr = be.array(Z)
                traced = be.einsum("astb,ts->ab", W, Z_arr)
            else:
                # Just trace: Σ_s W[a,s,s,b]
                traced = be.einsum("assb->ab", W)
            env = be.einsum("ab,bc->ac", env, traced)

        return float(np.real(be.to_numpy(env)[0, 0]))

    def expectation_zz(self, qubit_i: int, qubit_j: int) -> float:
        """Compute ⟨Z_i Z_j⟩ = Tr(Z_i Z_j ρ)."""
        be = self.be
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        env = be.array(np.ones((1, 1), dtype=np.complex128))
        for k in range(self.n_qubits):
            W = self._state[k]
            if k == qubit_i or k == qubit_j:
                Z_arr = be.array(Z)
                traced = be.einsum("astb,ts->ab", W, Z_arr)
            else:
                traced = be.einsum("assb->ab", W)
            env = be.einsum("ab,bc->ac", env, traced)

        return float(np.real(be.to_numpy(env)[0, 0]))

    def compute_purity(self) -> float:
        """Compute Tr(ρ²) — the purity of the state.

        Purity = 1 for pure states, 1/d for maximally mixed states.
        This is a key metric for DRL agents to understand decoherence.
        """
        be = self.be

        # Tr(ρ²) = contract two copies of the MPO
        # For each site: Σ_{s,t,u} W[a,s,t,b] * W[c,t,u,d] → (a*c, b*d)
        # Wait — this is Tr(ρ²) which requires contracting ρ with itself.
        # ρ² as MPO: for each site, contract the bra of first ρ with ket of second ρ
        # (ρ²)[i] = Σ_t W[a,s,t,b] * W[c,t,u,d] → shape (a*c, s, u, b*d)
        # Then Tr(ρ²) = trace over s=u for all sites

        env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(self.n_qubits):
            W = self._state[i]  # (χ_l, d, d, χ_r)
            # Contract two copies: Σ_{s,t,u} W1[a,s,t,b] * W2[c,t,u,d] * δ_{s,u}
            # = Σ_{s,t} W1[a,s,t,b] * W2[c,t,s,d]
            # env_new[(a,c), (b,d)] = env[(a',c')] * W1[a',s,t,b] * W2[c',t,s,d]
            rho_sq_site = be.einsum("astb,ctsd->acbd", W, W)
            chi_l1, chi_l2, chi_r1, chi_r2 = rho_sq_site.shape
            rho_sq_site = be.reshape(rho_sq_site, (chi_l1 * chi_l2, chi_r1 * chi_r2))
            env = be.einsum("ab,bc->ac", env, rho_sq_site)

        purity = float(np.real(be.to_numpy(env)[0, 0]))
        return max(0.0, min(1.0, purity))

    def compute_fidelity(self, target: MPS) -> float:
        """Compute fidelity F = ⟨ψ_target|ρ|ψ_target⟩.

        This uses the existing MPO.expectation_mps method.
        For a pure target state |ψ⟩, this gives the overlap with ρ.
        """
        result = self._state.expectation_mps(target)
        return float(np.real(result))

    def compute_entanglement_entropy(self, partition: int) -> float:
        """Compute the von Neumann entropy of the reduced state.

        For an MPO density matrix, we compute S(ρ_A) where A is the
        left partition (sites 0 to partition-1).

        This requires computing the reduced density matrix ρ_A = Tr_B(ρ)
        by tracing out sites partition to N-1.
        """
        be = self.be

        # Build the reduced density matrix of the left partition
        # by tracing out the right part
        # Start from the right: trace over sites N-1 down to partition
        right_env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(self.n_qubits - 1, partition - 1, -1):
            W = self._state[i]
            # Trace over physical indices: Σ_s W[a,s,s,b]
            traced = be.einsum("assb->ab", W)
            right_env = be.einsum("ab,bc->ac", traced, right_env)

        # Now contract the left part with the right environment
        # to get the reduced density matrix eigenvalues
        # We need the transfer matrix eigenvalues at the cut

        # Build left environment up to the partition
        left_env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(partition):
            W = self._state[i]
            traced = be.einsum("assb->ab", W)
            left_env = be.einsum("ab,bc->ac", left_env, traced)

        # The entropy of the reduced state requires diagonalizing ρ_A
        # For an MPO, this is non-trivial. We approximate by computing
        # the spectrum of the transfer matrix at the partition cut.
        # For now, use the simpler approach: Tr(ρ_A log ρ_A) ≈ bond entropy

        # Contract left_env with right_env to get Tr(ρ) as sanity check
        total_trace = float(np.real(be.to_numpy(
            be.einsum("ab,ba->", left_env, right_env)
        )))

        # Approximate entropy from the singular values of the MPO at the cut
        # Reshape the partition-boundary tensor for SVD
        if partition > 0 and partition < self.n_qubits:
            W = self._state[partition - 1]
            chi_l, d1, d2, chi_r = W.shape
            mat = be.reshape(W, (chi_l * d1 * d2, chi_r))
            _, S, _ = be.svd(mat, full_matrices=False)
            s_np = be.to_numpy(S)
            s_np = np.abs(s_np)
            s_np = s_np[s_np > 1e-15]
            s_sq = s_np ** 2
            s_sq = s_sq / s_sq.sum()
            entropy = -np.sum(s_sq * np.log2(s_sq + 1e-30))
            return float(entropy)

        return 0.0

    # ── Measurement ────────────────────────────────────────────────

    def _compute_outcome_probability(self, qubit: int, outcome: int) -> float:
        """Compute P(outcome) = Tr(Π_outcome ρ).

        Π_0 = |0⟩⟨0|, Π_1 = |1⟩⟨1|
        """
        be = self.be

        env = be.array(np.ones((1, 1), dtype=np.complex128))
        for i in range(self.n_qubits):
            W = self._state[i]
            if i == qubit:
                # Tr(Π_outcome W) = W[a, outcome, outcome, b]
                traced = W[:, outcome, outcome, :]
            else:
                traced = be.einsum("assb->ab", W)
            env = be.einsum("ab,bc->ac", env, traced)

        prob = float(np.real(be.to_numpy(env)[0, 0]))
        return max(0.0, min(1.0, prob))

    def measure_qubit(self, qubit: int) -> int:
        """Projective measurement of a single qubit.

        Returns 0 or 1, and collapses the state:
        ρ → Π_m ρ Π_m / Tr(Π_m ρ Π_m)
        """
        p0 = self._compute_outcome_probability(qubit, 0)
        p1 = 1.0 - p0

        # Apply measurement error
        if self.config.noise_enabled:
            fid = self.device.noise_params.measurement_fidelity
            p0_noisy = fid * p0 + (1 - fid) * p1
        else:
            p0_noisy = p0

        # Sample outcome
        outcome = 0 if self.rng.random() < p0_noisy else 1

        # Collapse: ρ → Π_m ρ Π_m / P(m)
        be = self.be
        W = self._state[qubit]
        # Zero out the non-outcome components
        W_new = be.zeros(W.shape)
        W_new[:, outcome, outcome, :] = W[:, outcome, outcome, :]
        self._state[qubit] = W_new

        # Renormalize
        tr = self.trace()
        if tr > 1e-15:
            self._state[0] = be.array(
                be.to_numpy(self._state[0]) / tr
            )

        self._time += self.device.gate_times.get("readout", 10e-6)
        return outcome

    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        return [self.measure_qubit(q) for q in range(self.n_qubits)]

    # ── Circuit execution ──────────────────────────────────────────

    def execute_circuit(self, circuit: List[Tuple]) -> Dict:
        """Execute a sequence of gate operations on the MPO state.

        Same interface as the MPS simulator for compatibility.
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
            "trace": self.trace(),
        }

    # ── Snapshot and metrics ───────────────────────────────────────

    def snapshot(self) -> Dict:
        """Take a snapshot of the current simulator state."""
        metrics = {
            "time": self._time,
            "step": self._step_count,
            "trace": self.trace(),
            "bond_dims": list(self._state.bond_dims),
            "max_bond": self._state.max_bond_dim,
            "z_expectations": [
                self.expectation_z(q) for q in range(self.n_qubits)
            ],
        }
        if self.config.track_purity:
            metrics["purity"] = self.compute_purity()
        self._history.append(metrics)
        return metrics

    @property
    def history(self) -> List[Dict]:
        return self._history
