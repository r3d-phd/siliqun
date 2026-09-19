"""Small ideal state-vector simulator for the standalone V2 baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import Circuit, Gate


@dataclass(frozen=True)
class SimulationResult:
    """Result of an ideal circuit simulation."""

    n_qubits: int
    statevector: np.ndarray

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.statevector) ** 2


def fidelity(left: np.ndarray, right: np.ndarray) -> float:
    """Return pure-state fidelity after checking equal nonzero vector norms."""

    left = np.asarray(left, dtype=complex).reshape(-1)
    right = np.asarray(right, dtype=complex).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("state vectors must have the same shape")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("state vectors must have nonzero norm")
    return float(abs(np.vdot(left, right)) ** 2 / (left_norm**2 * right_norm**2))


class StateVectorSimulator:
    """Ideal simulator with a bounded local state-vector allocation.

    Qubit zero is the least significant computational-basis bit. The bound is
    an allocation safeguard for this educational baseline, not a simulator
    scalability claim.
    """

    def __init__(self, n_qubits: int, max_qubits: int = 20) -> None:
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        if n_qubits > max_qubits:
            raise ValueError("n_qubits exceeds this baseline allocation bound")
        self.n_qubits = n_qubits
        self.max_qubits = max_qubits

    def run(self, circuit: Circuit) -> SimulationResult:
        if circuit.n_qubits != self.n_qubits:
            raise ValueError("circuit and simulator qubit counts differ")
        state = np.zeros(1 << self.n_qubits, dtype=complex)
        state[0] = 1.0
        for gate in circuit.gates:
            self._apply_gate(state, gate)
        return SimulationResult(n_qubits=self.n_qubits, statevector=state)

    def _apply_gate(self, state: np.ndarray, gate: Gate) -> None:
        if gate.name in {"x", "y", "z", "h", "rx", "ry", "rz"}:
            self._apply_single(state, gate.targets[0], _single_matrix(gate))
            return
        if gate.name == "cx":
            self._apply_cx(state, *gate.targets)
            return
        if gate.name == "cz":
            self._apply_cz(state, *gate.targets)
            return
        if gate.name == "swap":
            self._apply_swap(state, *gate.targets)
            return
        raise ValueError(f"unsupported gate: {gate.name}")

    def _apply_single(self, state: np.ndarray, target: int, matrix: np.ndarray) -> None:
        mask = 1 << target
        for base in range(0, len(state), mask << 1):
            for offset in range(mask):
                low, high = base + offset, base + offset + mask
                low_value, high_value = state[low], state[high]
                state[low] = matrix[0, 0] * low_value + matrix[0, 1] * high_value
                state[high] = matrix[1, 0] * low_value + matrix[1, 1] * high_value

    @staticmethod
    def _apply_cx(state: np.ndarray, control: int, target: int) -> None:
        control_mask, target_mask = 1 << control, 1 << target
        for index in range(len(state)):
            if index & control_mask and not index & target_mask:
                flipped = index | target_mask
                state[index], state[flipped] = state[flipped], state[index]

    @staticmethod
    def _apply_cz(state: np.ndarray, control: int, target: int) -> None:
        mask = (1 << control) | (1 << target)
        for index in range(len(state)):
            if index & mask == mask:
                state[index] *= -1

    @staticmethod
    def _apply_swap(state: np.ndarray, left: int, right: int) -> None:
        left_mask, right_mask = 1 << left, 1 << right
        for index in range(len(state)):
            if not index & left_mask and index & right_mask:
                swapped = (index | left_mask) & ~right_mask
                state[index], state[swapped] = state[swapped], state[index]


def _single_matrix(gate: Gate) -> np.ndarray:
    if gate.name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if gate.name == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if gate.name == "z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if gate.name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    theta = gate.parameters[0]
    if gate.name == "rx":
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    if gate.name == "ry":
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    if gate.name == "rz":
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    raise ValueError(f"unsupported single-qubit gate: {gate.name}")
