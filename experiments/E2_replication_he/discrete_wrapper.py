# -*- coding: utf-8 -*-
"""Discrete-token wrapper for SiliQunEnv (4-qubit linear chain).

Maps Discrete(23) -> the env's Box(11,) continuous action:
    Box layout (n=4): [gate_logits(5), qubit_logits(4), theta, phi]
    gate argmax -> ["rx", "ry", "rz", "cnot", "cz"]
    qubit argmax (single-q) / top-2 (two-q, edge-constrained)
    theta = box[9] * pi, phi = box[10] * pi

Token layout (family per paper sec3 subsec:gymnasium; the paper's
'11 tokens' compresses the 4 rotation types x 4 qubit indices into
4 named types -- here each (type, qubit) pair is an explicit token):
    0-15   single-qubit rotations, per qubit q in 0..3:
              4q+0: RX(pi/2)   4q+1: RX(pi)
              4q+2: RY(pi/2)   4q+3: RY(pi)
    16-21  two-qubit gates on adjacent pairs:
              16: CNOT(0,1)  17: CZ(0,1)
              18: CNOT(1,2)  19: CZ(1,2)
              20: CNOT(2,3)  21: CZ(2,3)
    22     idle: RZ(0) on qubit 0  (identity rotation; the env still
         advances one gate window, so idle decoherence applies)

n_qubits must be 4; pairs are the linear-chain edges (0-1, 1-2, 2-3).
"""
import numpy as np
import gymnasium as gym


class DiscreteTokenEnv(gym.Wrapper):
    def __init__(self, env, n_qubits: int = 4):
        super().__init__(env)
        assert env.n_qubits == 4, "wrapper is for the 4-qubit chain"
        self.n_qubits = n_qubits
        self.action_space = gym.spaces.Discrete(23)
        self.observation_space = env.observation_space
        self.token_map = {}
        q = 0
        for base in (0, 4, 8, 12):
            for k, (gate, ang) in enumerate(
                [(0, 0.5), (0, 1.0), (1, 0.5), (1, 1.0)]
            ):
                self.token_map[base + k] = ("single", gate, q, ang)
            q += 1
        for i, (gate, (a, b)) in enumerate(
            [(3, (0, 1)), (4, (0, 1)), (3, (1, 2)),
             (4, (1, 2)), (3, (2, 3)), (4, (2, 3))]
        ):
            self.token_map[16 + i] = ("pair", gate, (a, b), None)
        self.token_map[22] = ("idle", 2, 0, 0.0)

    def _box(self, token: int) -> np.ndarray:
        a = np.zeros(11, dtype=np.float32)
        kind, gate, q, ang = self.token_map[int(token)]
        a[gate] = 1.0  # one-hot gate_logits (argmax is exact for +-1)
        if kind == "pair":
            a[5 + q[0]] = 1.0
            a[5 + q[1]] = 1.0
        else:
            a[5 + q] = 1.0
        a[9] = ang  # theta = box[9]*pi
        return a

    def step(self, action):
        return self.env.step(self._box(int(action)))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
