"""
siliqun.envs
============
Core Gym-compatible environment for silicon spin qubit control.

SiliQunEnv simulates a chain of n silicon spin qubits under:
  - Single-qubit Rx/Ry/Rz control pulses (continuous action space)
  - ZZ Ising nearest-neighbour coupling (SiMOS nominal: J = 0.05 rad/ns)
  - Charge noise (Gaussian, σ_charge = 0.01)
  - Phonon dephasing (T1 = 10 µs, T2 = 2 µs)

The agent's goal is to drive the quantum state from |0...0⟩ to a
specified target state (GHZ, W, Cluster-linear, Dicke-k, or custom)
with fidelity F = |⟨ψ_target|ψ⟩|² ≥ 0.99.

This module is self-contained and has no dependency on QUASAR, ANDROMEDA,
or any other project. All physics is implemented from first principles.

Author: Raad Al-Shehri | KAU FCIT PhD
"""

from __future__ import annotations

import math
import cmath
import random
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Target state factory
# ---------------------------------------------------------------------------

def _ghz_state(n: int) -> np.ndarray:
    """GHZ state: (|0...0⟩ + |1...1⟩) / √2"""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0]      = 1.0 / math.sqrt(2)
    state[dim - 1] = 1.0 / math.sqrt(2)
    return state


def _w_state(n: int) -> np.ndarray:
    """W state: uniform superposition of all single-excitation basis states."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    amp = 1.0 / math.sqrt(n)
    for i in range(n):
        idx = 1 << (n - 1 - i)
        state[idx] = amp
    return state


def _cluster_linear_state(n: int) -> np.ndarray:
    """1D cluster state (linear graph): |+⟩^⊗n stabilised by CZ gates."""
    plus = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
    state = plus.copy()
    for _ in range(n - 1):
        state = np.kron(state, plus)
    # Apply CZ between adjacent pairs
    dim = 2 ** n
    for i in range(n - 1):
        for idx in range(dim):
            bi = (idx >> (n - 1 - i)) & 1
            bj = (idx >> (n - 2 - i)) & 1
            if bi == 1 and bj == 1:
                state[idx] *= -1.0
    norm = np.linalg.norm(state)
    return state / norm


def _dicke_state(n: int, k: int) -> np.ndarray:
    """Dicke state |D_n^k⟩: equal superposition of all n-qubit states with
    exactly k excitations."""
    from math import comb
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    amp = 1.0 / math.sqrt(comb(n, k))
    for idx in range(dim):
        if bin(idx).count("1") == k:
            state[idx] = amp
    return state


_TARGET_REGISTRY: Dict[str, Any] = {
    "ghz":          lambda n: _ghz_state(n),
    "w":            lambda n: _w_state(n),
    "cluster_linear": lambda n: _cluster_linear_state(n),
    "dicke_k1":     lambda n: _dicke_state(n, 1),
    "dicke_k2":     lambda n: _dicke_state(n, 2),
    "dicke_k3":     lambda n: _dicke_state(n, 3),
}


def get_target_state(name: str, n_qubits: int) -> np.ndarray:
    """Return the target state vector for a given name and qubit count."""
    key = name.lower().strip()
    if key not in _TARGET_REGISTRY:
        raise ValueError(
            f"Unknown target state '{name}'. "
            f"Available: {sorted(_TARGET_REGISTRY.keys())}"
        )
    return _TARGET_REGISTRY[key](n_qubits)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

class SiSpinNoiseModel:
    """
    Silicon spin qubit noise model.

    Parameters
    ----------
    stage : int
        Noise stage 0–5. Higher = more noise.
        Stage 0: noiseless (ideal).
        Stage 1: charge noise only (σ = 0.002).
        Stage 2: charge noise (σ = 0.005) + T2 dephasing (T2 = 10 µs).
        Stage 3: charge noise (σ = 0.008) + T2 = 5 µs + T1 = 50 µs.
        Stage 4: charge noise (σ = 0.01)  + T2 = 2 µs + T1 = 10 µs.
        Stage 5: charge noise (σ = 0.015) + T2 = 1 µs + T1 = 5 µs.
    dt : float
        Time step in nanoseconds (default 1.0 ns).
    """

    _PARAMS = {
        0: dict(sigma_charge=0.000, T1_us=None,  T2_us=None),
        1: dict(sigma_charge=0.002, T1_us=None,  T2_us=None),
        2: dict(sigma_charge=0.005, T1_us=None,  T2_us=10.0),
        3: dict(sigma_charge=0.008, T1_us=50.0,  T2_us=5.0),
        4: dict(sigma_charge=0.010, T1_us=10.0,  T2_us=2.0),
        5: dict(sigma_charge=0.015, T1_us=5.0,   T2_us=1.0),
    }

    def __init__(self, stage: int = 5, dt: float = 1.0):
        if stage not in self._PARAMS:
            raise ValueError(f"Noise stage must be 0–5, got {stage}.")
        p = self._PARAMS[stage]
        self.sigma_charge = p["sigma_charge"]
        self.T1_us        = p["T1_us"]
        self.T2_us        = p["T2_us"]
        self.dt           = dt          # ns
        self.dt_us        = dt * 1e-3   # µs

    def apply(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Apply noise to the quantum state vector in-place and return it."""
        if self.sigma_charge > 0:
            state = self._charge_noise(state, n_qubits)
        if self.T2_us is not None:
            state = self._dephasing(state, n_qubits)
        if self.T1_us is not None:
            state = self._relaxation(state, n_qubits)
        norm = np.linalg.norm(state)
        if norm > 1e-12:
            state /= norm
        return state

    def _charge_noise(self, state: np.ndarray, n: int) -> np.ndarray:
        """Gaussian charge noise: random phase kicks on each qubit."""
        dim = 2 ** n
        for q in range(n):
            delta = np.random.normal(0.0, self.sigma_charge)
            for idx in range(dim):
                bit = (idx >> (n - 1 - q)) & 1
                state[idx] *= cmath.exp(1j * delta * (1 - 2 * bit))
        return state

    def _dephasing(self, state: np.ndarray, n: int) -> np.ndarray:
        """T2 dephasing: exponential decay of off-diagonal coherences."""
        gamma2 = 1.0 / self.T2_us if self.T2_us else 0.0
        factor = math.exp(-gamma2 * self.dt_us)
        dim = 2 ** n
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    # Approximate: scale off-diagonal elements
                    pass  # Applied via random phase model above
        # Simpler: random dephasing angle per qubit
        for q in range(n):
            phi = np.random.normal(0.0, math.sqrt(2 * gamma2 * self.dt_us))
            for idx in range(dim):
                bit = (idx >> (n - 1 - q)) & 1
                state[idx] *= cmath.exp(1j * phi * (1 - 2 * bit))
        return state

    def _relaxation(self, state: np.ndarray, n: int) -> np.ndarray:
        """T1 relaxation: amplitude damping toward |0...0⟩."""
        gamma1 = 1.0 / self.T1_us if self.T1_us else 0.0
        p_decay = 1.0 - math.exp(-gamma1 * self.dt_us)
        dim = 2 ** n
        for q in range(n):
            for idx in range(dim):
                bit = (idx >> (n - 1 - q)) & 1
                if bit == 1:
                    if np.random.random() < p_decay:
                        # Flip qubit q from |1⟩ to |0⟩
                        new_idx = idx ^ (1 << (n - 1 - q))
                        state[new_idx] += state[idx]
                        state[idx] = 0.0
        return state


# ---------------------------------------------------------------------------
# SiliQunEnv — the core Gym-compatible environment
# ---------------------------------------------------------------------------

class SiliQunEnv:
    """
    Silicon Spin Qubit Control Environment.

    A Gym-compatible continuous-action environment for quantum state
    preparation on a chain of n silicon spin qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (2–8 supported).
    target_state : str
        Target state name: 'ghz', 'w', 'cluster_linear',
        'dicke_k1', 'dicke_k2', 'dicke_k3'.
    noise_stage : int
        Noise level 0 (noiseless) to 5 (realistic SiMOS noise).
    max_steps : int
        Maximum episode length.
    J : float
        ZZ Ising coupling strength in rad/ns (default 0.05).
    dt : float
        Time step in ns (default 1.0).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_qubits:     int            = 3,
        target_state: str            = "ghz",
        noise_stage:  int            = 5,
        max_steps:    int            = 200,
        J:            float          = 0.05,
        dt:           float          = 1.0,
        seed:         Optional[int]  = None,
        reward_weights: Tuple[float, float, float, float] = (1.0, 0.1, 0.05, 0.01),
    ):
        if n_qubits < 2 or n_qubits > 8:
            raise ValueError(f"n_qubits must be 2–8, got {n_qubits}.")

        self.n_qubits     = n_qubits
        self.target_state = target_state.lower().strip()
        self.noise_stage  = noise_stage
        self.max_steps    = max_steps
        self._J           = J
        self._dt          = dt
        self.reward_weights = reward_weights

        self.dim     = 2 ** n_qubits
        self.act_dim = 3 * n_qubits          # Rx, Ry, Rz per qubit
        # obs = Re(ψ) + Im(ψ) + |SWDFT|[:n] + [F]
        self.obs_dim = 2 * self.dim + n_qubits + 1

        self.target   = get_target_state(target_state, n_qubits)
        self._noise   = SiSpinNoiseModel(stage=noise_stage, dt=dt)

        self._rng     = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Gym-compatible space shapes (used by SAC agent)
        self.observation_space_shape = (self.obs_dim,)
        self.action_space_shape      = (self.act_dim,)

        # Episode state
        self.state:    np.ndarray = np.zeros(self.dim, dtype=complex)
        self._steps:   int        = 0
        self._best_F:  float      = 0.0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset to |0...0⟩ and return (obs, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            np.random.seed(seed)
        self.state         = np.zeros(self.dim, dtype=complex)
        self.state[0]      = 1.0 + 0j          # |0...0⟩
        self._steps        = 0
        self._best_F       = 0.0
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply control pulses and advance one time step.

        Parameters
        ----------
        action : np.ndarray, shape (3*n_qubits,)
            Rotation angles [θx_0, θy_0, θz_0, θx_1, ...] in radians.
            Expected range: [-π, π].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.clip(action, -math.pi, math.pi)

        # 1. Apply single-qubit rotations
        for q in range(self.n_qubits):
            tx = float(action[3 * q])
            ty = float(action[3 * q + 1])
            tz = float(action[3 * q + 2])
            self._apply_single_qubit(q, tx, ty, tz)

        # 2. Apply ZZ coupling
        self._apply_zz_coupling()

        # 3. Apply noise
        self.state = self._noise.apply(self.state, self.n_qubits)

        # 4. Normalise
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state /= norm

        self._steps += 1
        F = self._fidelity()
        if F > self._best_F:
            self._best_F = F

        # Shaped reward
        w_F, w_S, w_str, w_log = self.reward_weights
        reward = (
            w_F   * F
            + w_S   * (F - self._best_F + F)   # improvement bonus
            + w_str * float(F >= 0.99)           # success bonus
            + w_log * math.log(max(F, 1e-8) + 1)
        )

        terminated = bool(F >= 0.999)
        truncated  = bool(self._steps >= self.max_steps)

        info = {
            "fidelity":  F,
            "best_F":    self._best_F,
            "n_qubits":  self.n_qubits,
            "target":    self.target_state,
            "step":      self._steps,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        F = self._fidelity()
        print(
            f"  SiliQunEnv | {self.n_qubits}Q/{self.target_state} | "
            f"step={self._steps}/{self.max_steps} | F={F:.4f}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        re    = self.state.real.astype(np.float32)
        im    = self.state.imag.astype(np.float32)
        swdft = np.abs(
            np.fft.fft(np.abs(self.state))[: self.n_qubits]
        ).astype(np.float32)
        fid   = np.array([self._fidelity()], dtype=np.float32)
        return np.concatenate([re, im, swdft, fid])

    def _fidelity(self) -> float:
        return float(np.abs(np.dot(self.target.conj(), self.state)) ** 2)

    def _apply_single_qubit(
        self, qubit: int, tx: float, ty: float, tz: float
    ) -> None:
        Rx = np.array(
            [
                [math.cos(tx / 2),         -1j * math.sin(tx / 2)],
                [-1j * math.sin(tx / 2),    math.cos(tx / 2)     ],
            ],
            dtype=complex,
        )
        Ry = np.array(
            [
                [math.cos(ty / 2),  -math.sin(ty / 2)],
                [math.sin(ty / 2),   math.cos(ty / 2)],
            ],
            dtype=complex,
        )
        Rz = np.array(
            [
                [cmath.exp(-1j * tz / 2), 0                      ],
                [0,                        cmath.exp(1j * tz / 2)],
            ],
            dtype=complex,
        )
        U_q = Rz @ Ry @ Rx
        ops = [np.eye(2, dtype=complex)] * self.n_qubits
        ops[qubit] = U_q
        U_full = ops[0]
        for op in ops[1:]:
            U_full = np.kron(U_full, op)
        self.state = U_full @ self.state

    def _apply_zz_coupling(self) -> None:
        """Apply ZZ Ising coupling between adjacent qubits (SiMOS nominal)."""
        dim = 2 ** self.n_qubits
        for i in range(self.n_qubits - 1):
            for idx in range(dim):
                zi = 1 - 2 * ((idx >> (self.n_qubits - 1 - i)) & 1)
                zj = 1 - 2 * ((idx >> (self.n_qubits - 2 - i)) & 1)
                self.state[idx] *= cmath.exp(
                    -1j * self._J * self._dt * zi * zj / 2
                )
