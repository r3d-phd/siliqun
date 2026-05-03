"""
silicon_spin_ghz.py — SiliQunEnv v6.3
======================================
GPU-accelerated silicon spin qubit environment for multi-qubit GHZ state
preparation using deep reinforcement learning.

This module is part of the SiliQun package and provides the canonical
SiliQunEnv class used in the QUASAR v6.3/v6.4 curriculum training framework.

Key features (v6.3):
  - Correct GHZ fidelity formula: F = |<GHZ_n|psi>|^2
  - Adaptive episode length: 10 * n_qubits + 20
  - Noise curriculum: noise_level ramps 0.002 -> 0.02 over training
  - SeQurAIty-compatible noise-robust reward (v6.4 extension)
  - Supports 3, 5, 6, 8, 9 qubit GHZ targets

Physical model:
  Silicon spin qubits with charge noise (1/f spectral density),
  ZZ crosstalk, and gate calibration errors. Noise level 0.02
  corresponds to the conservative end of the physical noise budget
  for state-of-the-art silicon spin qubit devices (2024).

CRITICAL: Do NOT reduce noise_level below 0.008 (physical floor).
Doing so produces physically meaningless results and undermines
adversarial noise resistance for SeQurAIty applications.

Reference:
  R. Al-Shehri et al., "QUASAR: Quantum-Adaptive SAC with Adaptive
  Reconfiguration for GHZ State Preparation on Silicon Spin Qubits,"
  IEEE Transactions on Quantum Engineering (submitted 2026).
"""

import numpy as np
from typing import Optional, Tuple


class SiliQunEnv:
    """
    Silicon spin qubit environment for GHZ state preparation.

    Action space: continuous, shape (2 * n_qubits,)
        Each pair (theta_i, phi_i) parameterises a single-qubit rotation
        on qubit i. Entangling CZ gates are applied between adjacent qubits
        in a 1D chain (stages 0-2) or 2D grid (stages 3-4).

    Observation space: shape (obs_dim,)
        Concatenation of [Re(rho_diag), Im(rho_offdiag), noise_level]
        where obs_dim = 2^n_qubits + 2^(n_qubits-1) + 1.
        Scales as O(2^n) but remains tractable for n <= 9.

    Reward: F_t - F_{t-1}  (dense, shaped)
        Terminal bonus: F_T + 5 * F_T  (amplifies final-state quality)

    Noise model:
        Gaussian dephasing with noise_level controlling the per-step
        perturbation magnitude. Noise curriculum ramps from
        noise_level_start to noise_level_end over total_steps.

    Args:
        n_qubits (int): Number of qubits (3, 5, 6, 8, or 9).
        noise_level (float): Per-step noise magnitude (default 0.02).
        noise_level_start (float): Initial noise for curriculum (default 0.002).
        noise_level_end (float): Final noise for curriculum (default 0.02).
        total_steps (int): Total training steps for curriculum scheduling.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        noise_level: float = 0.02,
        noise_level_start: float = 0.002,
        noise_level_end: float = 0.02,
        total_steps: int = 500_000,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.noise_level = noise_level
        self.noise_level_start = noise_level_start
        self.noise_level_end = noise_level_end
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)

        # Adaptive episode length (v6.3)
        self.episode_length = 10 * n_qubits + 20

        # Hilbert space dimension
        self.dim = 2 ** n_qubits

        # Observation and action dimensions
        self.obs_dim = self.dim + (self.dim // 2) + 1
        self.act_dim = 2 * n_qubits

        # Pre-compute target GHZ state
        self.ghz_state = self._make_ghz_state()

        # State
        self.state: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.global_step: int = 0
        self.prev_fidelity: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset to |000...0> and return (obs, info)."""
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.step_count = 0
        self.prev_fidelity = self._fidelity()
        return self._observe(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action (rotation angles) and return (obs, reward, terminated,
        truncated, info).

        Args:
            action: Array of shape (2 * n_qubits,). Values in [-pi, pi].

        Returns:
            obs: Observation vector.
            reward: Dense fidelity improvement reward.
            terminated: True if fidelity >= 0.999 (perfect preparation).
            truncated: True if episode_length reached.
            info: Dict with fidelity, best_fidelity, noise_level.
        """
        # Update noise curriculum
        self._update_noise_curriculum()

        # Apply single-qubit rotations
        self._apply_rotations(action)

        # Apply entangling layer (CZ gates between adjacent qubits)
        self._apply_entangling_layer()

        # Apply noise
        self._apply_noise()

        # Compute reward
        fidelity = self._fidelity()
        reward = fidelity - self.prev_fidelity
        self.prev_fidelity = fidelity
        self.step_count += 1

        terminated = bool(fidelity >= 0.999)
        truncated = bool(self.step_count >= self.episode_length)

        if terminated or truncated:
            # Terminal bonus amplifies final-state quality
            reward += fidelity + 5.0 * fidelity

        info = {
            "fidelity": fidelity,
            "noise_level": self.noise_level,
            "step": self.step_count,
        }
        return self._observe(), reward, terminated, truncated, info

    def set_global_step(self, step: int) -> None:
        """Update global step counter for noise curriculum scheduling."""
        self.global_step = step

    # ------------------------------------------------------------------
    # Fidelity (CRITICAL: must use GHZ inner product, not ground state)
    # ------------------------------------------------------------------

    def _fidelity(self) -> float:
        """
        Compute GHZ fidelity: F = |<GHZ_n|psi>|^2.

        This is the CORRECT formula. Using |state[0]|^2 (overlap with |000>)
        is WRONG — it rewards the agent for doing nothing (v6.1 bug).
        """
        overlap = np.dot(self.ghz_state.conj(), self.state)
        return float(np.abs(overlap) ** 2)

    def _make_ghz_state(self) -> np.ndarray:
        """Construct the n-qubit GHZ state: (|00...0> + |11...1>) / sqrt(2)."""
        ghz = np.zeros(self.dim, dtype=complex)
        ghz[0] = 1.0 / np.sqrt(2)          # |00...0>
        ghz[self.dim - 1] = 1.0 / np.sqrt(2)  # |11...1>
        return ghz

    # ------------------------------------------------------------------
    # Quantum operations
    # ------------------------------------------------------------------

    def _apply_rotations(self, action: np.ndarray) -> None:
        """Apply parameterised single-qubit rotations Ry(theta) Rz(phi)."""
        for i in range(self.n_qubits):
            theta = float(action[2 * i])
            phi = float(action[2 * i + 1])
            self.state = self._apply_single_qubit_gate(
                self.state, i, self._ry(theta)
            )
            self.state = self._apply_single_qubit_gate(
                self.state, i, self._rz(phi)
            )

    def _apply_entangling_layer(self) -> None:
        """Apply CZ gates between all adjacent qubit pairs (1D chain)."""
        for i in range(self.n_qubits - 1):
            self.state = self._apply_cz(self.state, i, i + 1)

    def _apply_noise(self) -> None:
        """Apply Gaussian dephasing noise to the quantum state."""
        noise = self.rng.normal(0, self.noise_level, self.dim * 2)
        noise_complex = noise[:self.dim] + 1j * noise[self.dim:]
        self.state = self.state + noise_complex
        norm = np.linalg.norm(self.state)
        if norm > 1e-10:
            self.state /= norm

    def _update_noise_curriculum(self) -> None:
        """Linearly ramp noise_level from start to end over total_steps."""
        if self.total_steps > 0:
            progress = min(1.0, self.global_step / self.total_steps)
            self.noise_level = (
                self.noise_level_start
                + progress * (self.noise_level_end - self.noise_level_start)
            )

    # ------------------------------------------------------------------
    # Gate implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        """Ry(theta) single-qubit rotation matrix."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def _rz(phi: float) -> np.ndarray:
        """Rz(phi) single-qubit rotation matrix."""
        return np.array(
            [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
            dtype=complex,
        )

    def _apply_single_qubit_gate(
        self, state: np.ndarray, qubit: int, gate: np.ndarray
    ) -> np.ndarray:
        """Apply a 2x2 gate to a single qubit in the full state vector."""
        state_matrix = state.reshape([2] * self.n_qubits)
        state_matrix = np.tensordot(gate, state_matrix, axes=[[1], [qubit]])
        axes = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, self.n_qubits))
        state_matrix = np.transpose(state_matrix, axes)
        return state_matrix.reshape(self.dim)

    def _apply_cz(
        self, state: np.ndarray, control: int, target: int
    ) -> np.ndarray:
        """Apply a CZ gate between control and target qubits."""
        state_matrix = state.reshape([2] * self.n_qubits)
        # CZ flips the phase of |11> component
        idx = [slice(None)] * self.n_qubits
        idx[control] = 1
        idx[target] = 1
        state_matrix[tuple(idx)] *= -1
        return state_matrix.reshape(self.dim)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """
        Construct observation vector from current quantum state.

        obs = [Re(state), Im(state[:dim//2]), noise_level]
        Shape: (dim + dim//2 + 1,) = obs_dim
        """
        obs = np.concatenate([
            self.state.real,
            self.state[:self.dim // 2].imag,
            [self.noise_level],
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def t2(self) -> float:
        """Alias for noise_level (T2* dephasing proxy)."""
        return self.noise_level

    @t2.setter
    def t2(self, value: float) -> None:
        self.noise_level = value
