"""
Gymnasium environment wrapper for SiliQun.

Provides a standard Gymnasium interface for DRL-based quantum
control of silicon spin qubit systems. This is the bridge between
SiliQun and QUASAR/MOZAIQ/seQurAIt.

Supports two simulation backends:
    - MPS (pure state + quantum trajectories): fast, approximate noise
    - MPO (density matrix): exact noise, essential for mixed-state tasks

Observation space:
    - Qubit Z-expectations (n_qubits)
    - Pairwise ZZ correlators (n_qubits - 1)
    - Entanglement entropies at each cut (n_qubits - 1)
    - Current fidelity to target (1)
    - Elapsed time / max_time (1)
    - Bond dimension / max_bond (1)
    Total: 3*n_qubits - 1

    MPO mode adds (if include_purity=True):
    - State purity Tr(rho^2) (1)
    Total: 3*n_qubits

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
from .mpo_simulator import MPODensityMatrixSimulator, MPOSimConfig
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
    config : SimConfig or MPOSimConfig, optional
        Simulation configuration.
    reward_type : str
        Reward function type: "dense", "sparse", or "shaped".
    sim_mode : str
        Simulation mode: "mps" (pure state) or "mpo" (density matrix).
    include_purity : bool
        Whether to include state purity in observations (MPO mode only).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        device: DeviceProfile | str = "donor",
        n_qubits: int = 2,
        target_state: str = "bell",
        max_steps: int = 200,
        fidelity_threshold: float = 0.99,
        config: Optional[SimConfig | MPOSimConfig] = None,
        reward_type: str = "dense",
        render_mode: Optional[str] = None,
        sim_mode: str = "mps",
        include_purity: bool = False,
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
        self.sim_mode = sim_mode
        self.include_purity = include_purity and (sim_mode == "mpo")

        # Simulator -- select backend based on sim_mode
        if sim_mode == "mpo":
            self.config = config if isinstance(config, MPOSimConfig) else MPOSimConfig(
                noise_enabled=True,
                max_bond_dim=32,
            )
            self.sim = MPODensityMatrixSimulator(self.device, self.config)
        else:
            self.config = config if isinstance(config, SimConfig) else SimConfig(
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
        if self.include_purity:
            obs_dim += 1  # Add purity channel for MPO mode
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
        """Compute the observation vector.

        Works for both MPS and MPO simulators since they share
        the same observable interface.
        """
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
        if self.sim_mode == "mpo":
            max_bd = self.sim.state.max_bond_dim if hasattr(self.sim.state, 'max_bond_dim') else 1
        else:
            max_bd = max(self.sim.state.bond_dims) if self.sim.state.bond_dims else 1
        obs.append(min(max_bd / self.config.max_bond_dim, 1.0))

        # Purity (MPO mode only)
        if self.include_purity:
            purity = self.sim.compute_purity()
            obs.append(purity)

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
            if hasattr(self.config, 'seed'):
                self.config.seed = seed
            self.sim.rng = np.random.RandomState(seed)

        self.sim.reset()
        self._step_count = 0
        self._prev_fidelity = 0.0
        self._best_fidelity = 0.0

        obs = self._get_obs()
        info = {
            "fidelity": 0.0,
            "time": 0.0,
            "sim_mode": self.sim_mode,
        }
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
            "sim_mode": self.sim_mode,
        }

        # MPO-specific info
        if self.sim_mode == "mpo":
            info["trace"] = self.sim.trace()
            if self.include_purity:
                info["purity"] = self.sim.compute_purity()

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
            mode_label = "MPO" if self.sim_mode == "mpo" else "MPS"
            lines = [
                f"SiliQun [{mode_label}] | {self.device.name} | Step {self._step_count}/{self.max_steps}",
                f"Fidelity: {self._prev_fidelity:.6f} (best: {self._best_fidelity:.6f})",
                f"Time: {self.sim.time:.2e} s",
                f"Bond dims: {self.sim.state.bond_dims}",
                f"<Z>: {[f'{self.sim.expectation_z(q):.3f}' for q in range(self.n_qubits)]}",
            ]
            if self.sim_mode == "mpo":
                lines.append(f"Tr(rho): {self.sim.trace():.6f}")
                if self.include_purity:
                    lines.append(f"Purity: {self.sim.compute_purity():.6f}")
            return "\n".join(lines)
        return None

    def close(self):
        """Clean up resources."""
        pass


# -- Convenience factory functions ------------------------------------------

def make_siliqun_env(
    n_qubits: int = 2,
    device: str = "donor",
    target: str = "bell",
    sim_mode: str = "mps",
    noise: bool = True,
    max_bond_dim: int = 32,
    max_steps: int = 200,
    fidelity_threshold: float = 0.99,
    reward_type: str = "dense",
    include_purity: bool = False,
    seed: Optional[int] = None,
) -> SiliQunEnv:
    """Create a SiliQun Gymnasium environment with convenient defaults.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    device : str
        Device type: "donor", "simos", or "gaa".
    target : str
        Target state: "bell", "ghz", "w", or "random".
    sim_mode : str
        "mps" for pure-state (fast, approximate noise) or
        "mpo" for density matrix (slower, exact noise).
    noise : bool
        Whether to enable noise simulation.
    max_bond_dim : int
        Maximum bond dimension.
    max_steps : int
        Maximum steps per episode.
    fidelity_threshold : float
        Fidelity threshold for success.
    reward_type : str
        "dense", "sparse", or "shaped".
    include_purity : bool
        Include purity in observations (MPO mode only).
    seed : int, optional
        Random seed.

    Returns
    -------
    SiliQunEnv
        Configured Gymnasium environment.

    Examples
    --------
    >>> env = make_siliqun_env(n_qubits=2, sim_mode="mpo", target="bell")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    if sim_mode == "mpo":
        config = MPOSimConfig(
            noise_enabled=noise,
            max_bond_dim=max_bond_dim,
            seed=seed,
        )
    else:
        config = SimConfig(
            noise_enabled=noise,
            max_bond_dim=max_bond_dim,
            seed=seed or 42,
        )

    return SiliQunEnv(
        device=device,
        n_qubits=n_qubits,
        target_state=target,
        max_steps=max_steps,
        fidelity_threshold=fidelity_threshold,
        config=config,
        reward_type=reward_type,
        sim_mode=sim_mode,
        include_purity=include_purity,
    )
