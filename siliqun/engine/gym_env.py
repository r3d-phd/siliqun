"""
Gymnasium environment wrapper for SiliQun.

Provides a standard Gymnasium interface for DRL-based quantum
control of silicon spin qubit systems. This is the bridge between
SiliQun and QUASAR/MOZAIQ/SeQurAIty.

Observation space:
    - Qubit Z-expectations (n_qubits)
    - Pairwise ZZ correlators (n_qubits - 1)
    - Entanglement entropies at each cut (n_qubits - 1)
    - Current fidelity to target (1)
    - Elapsed time / max_time (1)
    - Bond dimension / max_bond (1)
    Total: 3*n_qubits - 1

Action space:
    Continuous: gate parameters for the native gate set.
    The action vector encodes which gate to apply, to which qubit(s),
    and with what parameters.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False

from .simulator import SiliQunSimulator, SimConfig
from ..physics.devices.profiles import DeviceProfile, get_device_profile
from ..tensor.mps import MPS


if HAS_GYM:
    BASE_CLASS = gym.Env
else:
    BASE_CLASS = object


class SiliQunEnv(BASE_CLASS):
    """Gymnasium environment for silicon spin qubit control.

    The agent controls a silicon spin qubit device by choosing
    gate operations to prepare a target quantum state with
    maximum fidelity.

    Parameters
    ----------
    device : DeviceProfile or str
        Device specification or device type name.
    n_qubits : int, optional
        Number of qubits (used if device is a string).
    target_state : str or MPS
        Target state to prepare. Options: "ghz", "bell", "w",
        "random", or a custom MPS.
    max_steps : int
        Maximum number of gate applications per episode.
    fidelity_threshold : float
        Fidelity threshold for success.
    config : SimConfig, optional
        Simulation configuration.
    reward_type : str
        Reward function type: "dense", "sparse", or "shaped".
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        device: DeviceProfile | str = "donor",
        n_qubits: int = 2,
        target_state: str = "bell",
        max_steps: int = 200,
        fidelity_threshold: float = 0.99,
        config: Optional[SimConfig] = None,
        reward_type: str = "dense",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Device setup
        if isinstance(device, str):
            self.device = get_device_profile(device, n_qubits)
        else:
            self.device = device
            n_qubits = device.n_qubits

        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.fidelity_threshold = fidelity_threshold
        self.reward_type = reward_type
        self.render_mode = render_mode

        # Simulator
        self.config = config or SimConfig(
            noise_enabled=True,
            max_bond_dim=32,
        )
        self.sim = SiliQunSimulator(self.device, self.config)

        # Target state
        self._target_name = target_state
        self._target_state: Optional[MPS] = None
        self._build_target(target_state)

        # Action space: [gate_type(5), qubit_select(n), theta(1), phi(1)]
        # gate_type: 0=Rx, 1=Ry, 2=Rz, 3=CNOT, 4=CZ
        n_gates = 5
        act_dim = n_gates + n_qubits + 2  # gate_logits + qubit_logits + theta + phi
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        # Observation space
        obs_dim = 3 * n_qubits - 1  # Z + ZZ + entropy + fidelity + time + bond
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Episode tracking
        self._step_count = 0
        self._prev_fidelity = 0.0
        self._best_fidelity = 0.0

    def _build_target(self, target_state):
        """Build the target quantum state."""
        if isinstance(target_state, MPS):
            self._target_state = target_state
        elif target_state == "bell":
            self._target_state = MPS.bell_state(self.n_qubits)
        elif target_state == "ghz":
            self._target_state = MPS.ghz_state(self.n_qubits)
        elif target_state == "w":
            self._target_state = MPS.w_state(self.n_qubits)
        elif target_state == "random":
            self._target_state = MPS.random_state(
                self.n_qubits, bond_dim=4
            )
        else:
            raise ValueError(f"Unknown target state: {target_state}")

    def _get_obs(self) -> np.ndarray:
        """Compute the observation vector."""
        obs = []

        # Z expectations (n_qubits values, range [-1, 1])
        for q in range(self.n_qubits):
            obs.append(self.sim.expectation_z(q))

        # ZZ correlators (n_qubits - 1 values)
        for q in range(self.n_qubits - 1):
            obs.append(self.sim.expectation_zz(q, q + 1))

        # Entanglement entropies (n_qubits - 1 values, normalized)
        max_entropy = np.log2(min(
            self.config.max_bond_dim, 2 ** (self.n_qubits // 2)
        ))
        for q in range(1, self.n_qubits):
            S = self.sim.compute_entanglement_entropy(q)
            obs.append(min(S / max(max_entropy, 1e-10), 1.0))

        # Current fidelity
        fid = self.sim.compute_fidelity(self._target_state)
        obs.append(fid)

        # Elapsed time fraction
        max_time = self.max_steps * max(
            self.device.gate_times.values()
        )
        obs.append(min(self.sim.time / max(max_time, 1e-15), 1.0))

        # Bond dimension fraction
        max_bd = max(self.sim.state.bond_dims) if self.sim.state.bond_dims else 1
        obs.append(min(max_bd / self.config.max_bond_dim, 1.0))

        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> Tuple[str, Dict, list]:
        """Decode a continuous action vector into a gate operation."""
        n = self.n_qubits
        n_gates = 5

        # Split action vector
        gate_logits = action[:n_gates]
        qubit_logits = action[n_gates:n_gates + n]
        theta = float(action[n_gates + n]) * np.pi      # [-π, π]
        phi = float(action[n_gates + n + 1]) * np.pi    # [-π, π]

        # Select gate type (argmax)
        gate_idx = int(np.argmax(gate_logits))
        gate_names = ["rx", "ry", "rz", "cnot", "cz"]
        gate_name = gate_names[gate_idx]

        # Select qubit(s)
        if gate_idx <= 2:  # Single-qubit gate
            qubit = int(np.argmax(qubit_logits))
            return gate_name, {"theta": theta}, [qubit]
        else:  # Two-qubit gate
            # Select two adjacent qubits
            sorted_qubits = np.argsort(qubit_logits)[::-1]
            q0 = int(sorted_qubits[0])
            q1 = int(sorted_qubits[1])
            # Ensure adjacency
            if abs(q0 - q1) != 1:
                q1 = min(q0 + 1, n - 1) if q0 < n - 1 else q0 - 1
            return gate_name, {}, [min(q0, q1), max(q0, q1)]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.config.seed = seed
            self.sim.rng = np.random.RandomState(seed)

        self.sim.reset()
        self._step_count = 0
        self._prev_fidelity = 0.0
        self._best_fidelity = 0.0

        obs = self._get_obs()
        info = {"fidelity": 0.0, "time": 0.0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        self._step_count += 1

        # Decode and apply action
        gate_name, params, qubits = self._decode_action(action)

        gate_map = {
            "rx": lambda: self.sim.apply_rx(params["theta"], qubits[0]),
            "ry": lambda: self.sim.apply_ry(params["theta"], qubits[0]),
            "rz": lambda: self.sim.apply_rz(params["theta"], qubits[0]),
            "cnot": lambda: self.sim.apply_cnot(qubits[0], qubits[1]),
            "cz": lambda: self.sim.apply_cz(qubits[0], qubits[1]),
        }
        gate_map[gate_name]()

        # Compute fidelity
        fidelity = self.sim.compute_fidelity(self._target_state)
        self._best_fidelity = max(self._best_fidelity, fidelity)

        # Compute reward
        reward = self._compute_reward(fidelity)
        self._prev_fidelity = fidelity

        # Check termination
        terminated = fidelity >= self.fidelity_threshold
        truncated = self._step_count >= self.max_steps

        # Build observation
        obs = self._get_obs()

        info = {
            "fidelity": fidelity,
            "best_fidelity": self._best_fidelity,
            "time": self.sim.time,
            "step": self._step_count,
            "gate": gate_name,
            "bond_dims": list(self.sim.state.bond_dims),
            "success": terminated,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, fidelity: float) -> float:
        """Compute the reward signal."""
        if self.reward_type == "sparse":
            return 1.0 if fidelity >= self.fidelity_threshold else 0.0

        elif self.reward_type == "dense":
            # Fidelity improvement + bonus for high fidelity
            delta_f = fidelity - self._prev_fidelity
            bonus = 0.0
            if fidelity >= self.fidelity_threshold:
                bonus = 10.0
            elif fidelity >= 0.95:
                bonus = 2.0
            return delta_f + bonus - 0.01  # small step penalty

        elif self.reward_type == "shaped":
            # Potential-based shaping: Φ(s') - Φ(s)
            # Φ(s) = log(fidelity + ε)
            eps = 1e-10
            phi_new = np.log(fidelity + eps)
            phi_old = np.log(self._prev_fidelity + eps)
            shaping = phi_new - phi_old

            # Success bonus
            bonus = 10.0 if fidelity >= self.fidelity_threshold else 0.0

            return shaping + bonus - 0.005

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def render(self):
        """Render the current state."""
        if self.render_mode == "ansi":
            lines = [
                f"SiliQun | {self.device.name} | Step {self._step_count}/{self.max_steps}",
                f"Fidelity: {self._prev_fidelity:.6f} (best: {self._best_fidelity:.6f})",
                f"Time: {self.sim.time:.2e} s",
                f"Bond dims: {self.sim.state.bond_dims}",
                f"⟨Z⟩: {[f'{self.sim.expectation_z(q):.3f}' for q in range(self.n_qubits)]}",
            ]
            return "\n".join(lines)
        return None

    def close(self):
        """Clean up resources."""
        pass
