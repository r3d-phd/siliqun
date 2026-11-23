# -*- coding: utf-8 -*-
"""bench_env_v5: 7-gate discrete benchmark environment for the survey
(Section 12, "Empirical Benchmark: A Standardised Comparison on SiMOS
Hardware").

Reconstruction of the survey benchmark environment built on SiliQun.
The original ``bench_env_v5`` implementation was hosted in the removed
``quasar-v6`` repository; this reconstruction follows the published
specification in the survey exactly:

* SiMOS silicon spin qubit simulator with hardware-accurate Lindblad
  noise (SiliQun ``simos`` device profile, exact state-vector backend).
* Task: prepare the two-qubit GHZ state |GHZ_2> = (|00> + |11>)/sqrt(2)
  from |00>.
* Discrete action space of six single-qubit rotations and one
  two-qubit entangling gate (7 actions total).
* Shaped dense reward (SiliQunEnv ``reward_type="shaped"``:
  log-fidelity potential-based shaping + success bonus).
* Fidelity threshold 0.99; episode length 200 gates.

Action layout (token -> gate):
    0: RX(pi/2) on qubit 0      1: RX(pi/2) on qubit 1
    2: RY(pi/2) on qubit 0      3: RY(pi/2) on qubit 1
    4: RZ(pi/2) on qubit 0      5: RZ(pi/2) on qubit 1
    6: CNOT(0 -> 1)

The optimal solution uses exactly three gates
(RX(pi/2) q0 -> CNOT -> RZ(pi/2) q1); no two-gate circuit reaches
F = 1, consistent with the survey statement that "the optimal circuit
requires at least three gates".

Also provides:
* ``BenchHistoryEnv``  -- observation extended with the last two
                          one-hot actions (flat-hierarchical variant).
* ``BenchRelaxationEnv`` -- Box(7,) logits over the 7 gates with
                          argmax interpretation (continuous
                          relaxation used for the DDPG row).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from siliqun.engine.gym_env import make_siliqun_env

N_TOKENS = 7
GATE_NAMES = [
    "RX(pi/2) q0",
    "RX(pi/2) q1",
    "RY(pi/2) q0",
    "RY(pi/2) q1",
    "RZ(pi/2) q0",
    "RZ(pi/2) q1",
    "CNOT(0->1)",
]

# token -> (gate_idx, [qubit_logits], theta_fraction)
# SiliQunEnv Box layout (n=2): [gate_logits(5), qubit_logits(2), theta, phi]
# gate idx: 0=rx, 1=ry, 2=rz, 3=cnot, 4=cz ; theta = box[9]*pi
_TOKEN_MAP = {
    0: (0, [1.0, 0.0], 0.5),
    1: (0, [0.0, 1.0], 0.5),
    2: (1, [1.0, 0.0], 0.5),
    3: (1, [0.0, 1.0], 0.5),
    4: (2, [1.0, 0.0], 0.5),
    5: (2, [0.0, 1.0], 0.5),
    6: (3, [1.0, 1.0], 0.0),  # CNOT: top-2 qubit logits -> edge (0,1)
}


class BenchTokenEnv(gym.Wrapper):
    """Discrete(7) wrapper over SiliQunEnv (2 qubits, SiMOS, GHZ target).

    Benchmark-specific adjustment: SiliQunEnv's native "shaped" reward
    initialises the potential to 0, which yields a constant ~+21.6
    first-step offset (log(0.25) - log(1e-10)) on every episode that
    drowns the per-step signal. We re-initialise the potential to the
    start-state fidelity at each reset, as intended by
    potential-based shaping.
    """

    def __init__(self, inner: gym.Env):
        super().__init__(inner)
        assert inner.n_qubits == 2, "bench_env_v5 is a 2-qubit task"
        self.action_space = spaces.Discrete(N_TOKENS)
        self.observation_space = inner.observation_space
        self.tokens = GATE_NAMES

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env._prev_fidelity = float(
            self.env.sim.compute_fidelity(self.env._target_state)
        )
        return obs, info

    def _box(self, token: int) -> np.ndarray:
        gate_idx, q_logits, theta = _TOKEN_MAP[int(token)]
        a = np.zeros(9, dtype=np.float32)
        a[gate_idx] = 1.0
        a[5 + 0] = q_logits[0]
        a[5 + 1] = q_logits[1]
        a[7] = theta  # theta = box[7]*pi -> pi/2 (layout: gates 0-4, qubits 5-6, theta 7, phi 8)
        return a

    def step(self, action):
        return self.env.step(self._box(int(action)))


def _install_fast_path(env):
    """Benchmark-local fast observation/fidelity path (no shared-library changes).

    The native SiliQunEnv obs path performs ~6 small numpy operations per
    step (2x expectation_z, expectation_zz, an SVD-based entropy, and two
    fidelity overlaps), which dominates wall time on the 2-qubit task.
    For n=2 with q0 as the most-significant bit and psi = [a, b, c, d]
    (basis 00, 01, 10, 11), every observation component is a closed-form
    function of the 4 amplitudes:

        <Z0> = |a|^2 + |b|^2 - |c|^2 - |d|^2
        <Z1> = |a|^2 + |c|^2 - |b|^2 - |d|^2
        <ZZ> = |a|^2 - |b|^2 - |c|^2 + |d|^2
        S(cut 1) = von Neumann entropy of rho_left = M M^dagger, M = [[a,b],[c,d]]
        F        = |<target|psi>|^2 (target is a dense vector in sv mode)

    These reproduce the native methods (statevector_simulator.py) exactly;
    the 2x2 entropy uses eigvalsh of rho_left instead of an SVD of the
    reshaped state (identical spectra).
    """
    sim = env.sim
    tgt = np.asarray(env._target_state, dtype=np.complex128)
    max_time = env.max_steps * max(env.device.gate_times.values())
    max_entropy = env.n_qubits / 2.0  # sv mode, per gym_env._get_obs

    def fast_obs():
        psi = np.asarray(sim._state)
        p = np.abs(psi) ** 2
        z0 = float(p[0] + p[1] - p[2] - p[3])
        z1 = float(p[0] + p[2] - p[1] - p[3])
        zz = float(p[0] - p[1] - p[2] + p[3])
        m = psi.reshape(2, 2)
        w = np.linalg.eigvalsh(m @ m.conj().T)
        w = w[w > 1e-15]
        if w.size:
            w = w / w.sum()
            ent = float(-np.sum(w * np.log2(w + 1e-30)))
        else:
            ent = 0.0
        overlap = np.dot(tgt.conj(), psi)
        fid = float(np.real(np.abs(overlap) ** 2))
        t = min(float(sim.time) / max(max_time, 1e-15), 1.0)
        return np.array(
            [z0, z1, zz, min(ent / max(max_entropy, 1e-10), 1.0), fid, t, 1.0],
            dtype=np.float32,
        )

    def fast_fidelity(_target):
        overlap = np.dot(tgt.conj(), np.asarray(sim._state))
        return max(0.0, min(1.0, float(np.real(np.abs(overlap) ** 2))))

    env._get_obs = fast_obs
    sim.compute_fidelity = fast_fidelity


def make_bench_env(seed: int | None = None, use_gpu: bool = False) -> BenchTokenEnv:
    """Factory: SiliQunEnv(2q, SiMOS, GHZ, SV exact, Lindblad noise)
    wrapped as the discrete 7-gate benchmark environment."""
    inner = make_siliqun_env(
        n_qubits=2,
        device="simos",
        target="ghz",
        sim_mode="sv",
        noise=True,
        max_steps=200,
        fidelity_threshold=0.99,
        reward_type="shaped",
        use_gpu=use_gpu,
        seed=seed,
    )
    _install_fast_path(inner)
    return BenchTokenEnv(inner)


class BenchHistoryEnv(gym.Wrapper):
    """Observation extended with one-hot of the last two actions.

    Used for the flat-hierarchical variant (PPO-hier): a minimal
    temporal abstraction over the last K-1 decisions without a
    separate manager agent.
    """

    HISTORY = 2

    def __init__(self, inner: gym.Env):
        super().__init__(inner)
        base = inner.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(base + 2 * N_TOKENS,), dtype=np.float32,
        )
        self._last = [np.zeros(N_TOKENS, dtype=np.float32) for _ in range(self.HISTORY)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last = [np.zeros(N_TOKENS, dtype=np.float32) for _ in range(self.HISTORY)]
        return self._extend(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        onehot = np.zeros(N_TOKENS, dtype=np.float32)
        onehot[int(action)] = 1.0
        self._last = [onehot] + self._last[: self.HISTORY - 1]
        return self._extend(obs), reward, terminated, truncated, info

    def _extend(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, *self._last]).astype(np.float32)


class BenchRelaxationEnv(gym.Wrapper):
    """Box(7,) logits over the 7 gates; argmax selects the gate.

    Continuous relaxation used for the DDPG row. A deterministic DDPG
    policy must push one logit above the others; with a 7-action
    discrete problem the policy gradient is effectively undefined,
    which is the mechanism analysed in the survey.
    """

    def __init__(self, inner: gym.Env):
        super().__init__(inner)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(N_TOKENS,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        token = int(np.argmax(np.asarray(action, dtype=np.float32)))
        return self.env.step(token)
