"""
Gymnasium environment wrapper for SiliQun.

Provides a standard Gymnasium interface for DRL-based quantum
control of silicon spin qubit systems. This is the bridge between
SiliQun and QUASAR/MOZAIQ/seQurAIt.

Supports two simulation backends:
    - MPS (pure state + quantum trajectories): fast, approximate noise
    - MPO (density matrix): exact noise, essential for mixed-state tasks

Supports four device architectures:
    - Donor (P:Si): ESR-driven, linear chain
    - SiMOS: EDSR-driven, linear chain
    - GAA: all-electric, linear chain
    - SLEDGE: exchange-only, DFS-encoded, 2D grid topologies
              (Weinstein et al., Nature 615, 817-822, 2023)

Observation space (standard):
    - Qubit Z-expectations (n_qubits)
    - Pairwise ZZ correlators (n_edges for 2D, n_qubits-1 for linear)
    - Entanglement entropies (n_cuts)
    - Current fidelity to target (1)
    - Elapsed time / max_time (1)
    - Bond dimension / max_bond (1)

Observation space (SLEDGE/DFS adds):
    - Total leakage probability (1)
    - Per-qubit gauge population (n_qubits)

Action space:
    Standard: gate parameters for the native gate set.
    SLEDGE: exchange angles for each connectivity edge (sequential).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
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
from .statevector_simulator import StateVectorSimulator, SVSimConfig
from .mpo_simulator import MPODensityMatrixSimulator, MPOSimConfig
from ..physics.devices.profiles import DeviceProfile, get_device_profile
from ..tensor.mps import MPS

# Optional DFS imports (only needed for SLEDGE)
try:
    from ..physics.dfs_encoding import (
        DFSEncoder, encoded_zero, encoded_one,
        compute_leakage, compute_gauge_population,
        exchange_quality_factor,
    )
    from ..physics.sequential_pulsing import (
        PulseScheduler, SequentialActionSpace, ExchangePulse,
    )
    HAS_DFS = True
except ImportError:
    HAS_DFS = False


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
    grid_shape : tuple, optional
        (rows, cols) for 2D grid topology (SLEDGE devices).
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
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()

        # Device setup
        if isinstance(device, str):
            if device == "sledge" and grid_shape is not None:
                self.device = get_device_profile(
                    device, n_qubits, grid_shape=grid_shape
                )
            else:
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
        self.grid_shape = grid_shape or getattr(self.device, 'grid_shape', None)

        # DFS encoding setup (SLEDGE devices)
        self.is_dfs = getattr(self.device, 'dfs_encoded', False)
        self.is_sequential = getattr(self.device, 'sequential_pulsing', False)
        self._dfs_encoder = None
        self._seq_action_space = None

        if self.is_dfs and HAS_DFS:
            self._dfs_encoder = DFSEncoder(n_qubits)
            self._seq_action_space = SequentialActionSpace(
                n_qubits=n_qubits,
                connectivity=self.device.connectivity,
                sequential=self.is_sequential,
                pulse_duration=self.device.gate_times.get('single', 10e-9),
                idle_duration=self.device.gate_times.get('idle', 10e-9),
            )

        # Connectivity info for observation construction
        self._edges = self.device.connectivity
        self._n_edges = len(self._edges)

        # Determine entanglement cuts based on topology
        if self.grid_shape is not None:
            # For 2D grids, use row-based bipartitions
            rows, cols = self.grid_shape
            self._ent_cuts = []
            for r in range(1, rows):
                # Cut between row r-1 and row r
                self._ent_cuts.append(r * cols)
            # Also add column-based cuts
            for c in range(1, cols):
                self._ent_cuts.append(c)  # Approximate: cut at qubit c
            self._n_cuts = len(self._ent_cuts)
        else:
            # Linear chain: cut at each bond
            self._ent_cuts = list(range(1, n_qubits))
            self._n_cuts = n_qubits - 1

        # Simulator -- select backend based on sim_mode
        # Auto-select state vector backend for large 2D grids (>9 qubits)
        if sim_mode == "auto":
            if n_qubits > 9 and self.grid_shape is not None:
                sim_mode = "sv"
            else:
                sim_mode = "mps"
            self.sim_mode = sim_mode

        if sim_mode == "mpo":
            self.config = config if isinstance(config, MPOSimConfig) else MPOSimConfig(
                noise_enabled=True,
                max_bond_dim=32,
            )
            self.sim = MPODensityMatrixSimulator(self.device, self.config)
        elif sim_mode == "sv":
            self.config = config if isinstance(config, SVSimConfig) else SVSimConfig(
                noise_enabled=True,
                use_gpu=True,
            )
            self.sim = StateVectorSimulator(self.device, self.config)
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

        # Action space
        if self.is_dfs and self._seq_action_space is not None:
            # SLEDGE: one exchange angle per edge
            act_dim = self._seq_action_space.action_dim
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(act_dim,),
                dtype=np.float32,
            )
        else:
            # Standard: gate_type(5) + qubit_select(n) + theta(1) + phi(1)
            n_gates = 5
            act_dim = n_gates + n_qubits + 2
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(act_dim,),
                dtype=np.float32,
            )

        # Observation space
        obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Episode tracking
        self._step_count = 0
        self._prev_fidelity = 0.0
        self._best_fidelity = 0.0
        self._total_leakage = 0.0

    def _compute_obs_dim(self) -> int:
        """Compute the observation dimension based on device type.

        Standard (linear):
            Z(n) + ZZ(n-1) + entropy(n-1) + fidelity(1) + time(1) + bond(1)
            = 3n + 1

        Standard (2D grid):
            Z(n) + ZZ(n_edges) + entropy(n_cuts) + fidelity(1) + time(1) + bond(1)
            = n + n_edges + n_cuts + 3

        SLEDGE/DFS adds:
            + leakage(1) + gauge_pop(n)
            = n + n_edges + n_cuts + 3 + 1 + n

        MPO adds:
            + purity(1)
        """
        n = self.n_qubits
        dim = n + self._n_edges + self._n_cuts + 3  # Z + ZZ + entropy + scalars

        if self.is_dfs:
            dim += 1 + n  # total leakage + per-qubit gauge population

        if self.include_purity:
            dim += 1

        return dim

    def _build_target(self, target_state):
        """Build the target quantum state.

        For SV mode, builds state vectors directly (more efficient
        for large systems). For MPS/MPO mode, builds MPS targets.
        """
        if self.sim_mode == "sv":
            # State vector mode: build dense target
            self._target_state = self._build_sv_target(target_state)
        else:
            # MPS/MPO mode: build MPS target
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

    def _build_sv_target(self, target_state) -> np.ndarray:
        """Build a dense state vector target for SV mode."""
        n = self.n_qubits
        dim = 2 ** n
        if isinstance(target_state, np.ndarray):
            return target_state
        elif target_state == "bell" or target_state == "ghz":
            # GHZ: (|00...0> + |11...1>) / sqrt(2)
            sv = np.zeros(dim, dtype=np.complex128)
            sv[0] = 1.0 / np.sqrt(2)
            sv[dim - 1] = 1.0 / np.sqrt(2)
            return sv
        elif target_state == "w":
            # W: equal superposition of single-excitation states
            sv = np.zeros(dim, dtype=np.complex128)
            for k in range(n):
                idx = 1 << (n - 1 - k)
                sv[idx] = 1.0 / np.sqrt(n)
            return sv
        elif target_state == "random":
            sv = np.random.randn(dim) + 1j * np.random.randn(dim)
            sv /= np.linalg.norm(sv)
            return sv.astype(np.complex128)
        else:
            raise ValueError(f"Unknown target state: {target_state}")

    def _get_obs(self) -> np.ndarray:
        """Compute the observation vector.

        Topology-aware: uses connectivity edges for ZZ correlators
        and topology-appropriate cuts for entanglement entropy.
        """
        obs = []

        # Z expectations (n_qubits values, range [-1, 1])
        for q in range(self.n_qubits):
            obs.append(self.sim.expectation_z(q))

        # ZZ correlators (n_edges values - topology-aware)
        for qi, qj in self._edges:
            obs.append(self.sim.expectation_zz(qi, qj))

        # Entanglement entropies (n_cuts values, normalized)
        if self.sim_mode == "sv":
            max_entropy = self.n_qubits / 2.0  # max possible entropy
        else:
            max_entropy = np.log2(min(
                getattr(self.config, 'max_bond_dim', 32),
                2 ** (self.n_qubits // 2),
            ))
        for cut_pos in self._ent_cuts:
            S = self.sim.compute_entanglement_entropy(cut_pos)
            obs.append(min(S / max(max_entropy, 1e-10), 1.0))

        # Current fidelity
        fid = self.sim.compute_fidelity(self._target_state)
        obs.append(fid)

        # Elapsed time fraction
        max_time = self.max_steps * max(
            self.device.gate_times.values()
        )
        obs.append(min(self.sim.time / max(max_time, 1e-15), 1.0))

        # Bond dimension / memory fraction
        if self.sim_mode == "sv":
            # SV mode: report memory utilization (always 1.0 for exact)
            obs.append(1.0)
        elif self.sim_mode == "mpo":
            max_bd = self.sim.state.max_bond_dim if hasattr(self.sim.state, 'max_bond_dim') else 1
            obs.append(min(max_bd / getattr(self.config, 'max_bond_dim', 32), 1.0))
        else:
            max_bd = max(self.sim.state.bond_dims) if self.sim.state.bond_dims else 1
            obs.append(min(max_bd / getattr(self.config, 'max_bond_dim', 32), 1.0))

        # DFS-specific observations
        if self.is_dfs and self._dfs_encoder is not None:
            # Total leakage probability
            obs.append(self._total_leakage)

            # Per-qubit gauge population (approximated from state)
            # For simulation, we track this from the DFS encoder
            for q in range(self.n_qubits):
                obs.append(0.0)  # Placeholder - updated during step

        # Purity (MPO mode only)
        if self.include_purity:
            purity = self.sim.compute_purity()
            obs.append(purity)

        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> Tuple[str, Dict, list]:
        """Decode a continuous action vector into a gate operation.

        For standard devices: selects gate type, qubit, and parameters.
        For SLEDGE: converts exchange angles to sequential pulse schedule.
        """
        if self.is_dfs and self._seq_action_space is not None:
            return self._decode_sledge_action(action)
        else:
            return self._decode_standard_action(action)

    def _decode_standard_action(
        self, action: np.ndarray
    ) -> Tuple[str, Dict, list]:
        """Decode standard gate-based action."""
        n = self.n_qubits
        n_gates = 5

        # Split action vector
        gate_logits = action[:n_gates]
        qubit_logits = action[n_gates:n_gates + n]
        theta = float(action[n_gates + n]) * np.pi      # [-pi, pi]
        phi = float(action[n_gates + n + 1]) * np.pi    # [-pi, pi]

        # Select gate type (argmax)
        gate_idx = int(np.argmax(gate_logits))
        gate_names = ["rx", "ry", "rz", "cnot", "cz"]
        gate_name = gate_names[gate_idx]

        # Select qubit(s)
        if gate_idx <= 2:  # Single-qubit gate
            qubit = int(np.argmax(qubit_logits))
            return gate_name, {"theta": theta}, [qubit]
        else:  # Two-qubit gate
            # Select two qubits connected by an edge
            sorted_qubits = np.argsort(qubit_logits)[::-1]
            q0 = int(sorted_qubits[0])
            q1 = int(sorted_qubits[1])
            # Ensure the pair is in the connectivity graph
            edge = (min(q0, q1), max(q0, q1))
            if edge not in self._edges:
                # Fall back to nearest connected neighbour
                for qi, qj in self._edges:
                    if qi == q0 or qj == q0:
                        q1 = qj if qi == q0 else qi
                        break
                else:
                    q0, q1 = self._edges[0]
            return gate_name, {}, [min(q0, q1), max(q0, q1)]

    def _decode_sledge_action(
        self, action: np.ndarray
    ) -> Tuple[str, Dict, list]:
        """Decode SLEDGE exchange-only action.

        The action vector contains one exchange angle per edge,
        scaled from [-1, 1] to [-pi, pi].

        Returns a special gate name "exchange_sequence" with the
        full pulse schedule in params.
        """
        # Scale action from [-1, 1] to [-pi, pi]
        angles = action * np.pi

        # Build pulse schedule
        schedule = self._seq_action_space.action_to_schedule(angles)
        total_time = self._seq_action_space.schedule_to_total_time(schedule)

        return "exchange_sequence", {
            "schedule": schedule,
            "angles": angles,
            "total_time": total_time,
        }, list(range(self.n_qubits))

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
        self._total_leakage = 0.0

        obs = self._get_obs()
        info = {
            "fidelity": 0.0,
            "time": 0.0,
            "sim_mode": self.sim_mode,
            "device_type": self.device.device_type,
            "dfs_encoded": self.is_dfs,
        }
        if self.grid_shape is not None:
            info["grid_shape"] = self.grid_shape
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

        if gate_name == "exchange_sequence":
            # SLEDGE: apply sequential exchange pulses
            self._apply_exchange_sequence(params)
        else:
            # Standard: apply single gate
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
            "device_type": self.device.device_type,
        }

        # DFS-specific info
        if self.is_dfs:
            # For SV mode, get leakage from the simulator directly
            if self.sim_mode == "sv" and hasattr(self.sim, 'leakage'):
                self._total_leakage = self.sim.leakage
            info["leakage"] = self._total_leakage
            info["dfs_encoded"] = True
            if gate_name == "exchange_sequence":
                info["n_pulses"] = sum(
                    len(layer.pulses)
                    for layer in params.get("schedule", [])
                )
                info["circuit_depth"] = len(params.get("schedule", []))
                info["total_pulse_time"] = params.get("total_time", 0.0)

        # Grid topology info
        if self.grid_shape is not None:
            info["grid_shape"] = self.grid_shape
            info["n_edges"] = self._n_edges

        # MPO-specific info
        if self.sim_mode == "mpo":
            info["trace"] = self.sim.trace()
            if self.include_purity:
                info["purity"] = self.sim.compute_purity()

        return obs, reward, terminated, truncated, info

    def _apply_exchange_sequence(self, params: Dict):
        """Apply a sequential exchange pulse schedule (SLEDGE mode).

        Each pulse in the schedule is applied as an exchange gate
        on the specified qubit pair, with idle decoherence applied
        to all other qubits during the pulse.
        """
        schedule = params.get("schedule", [])
        angles = params.get("angles", np.array([]))

        for layer in schedule:
            for pulse in layer.pulses:
                # Apply exchange gate between the two qubits
                qi, qj = pulse.qubit_i, pulse.qubit_j
                theta = pulse.angle

                # Use the simulator's exchange gate
                # theta = J * t / hbar, so t = theta / J
                # Use device exchange frequency (J/h) to compute pulse time
                J_freq = getattr(self, '_exchange_freq', 100e6)  # Hz
                J_coupling = J_freq * 2 * np.pi  # angular frequency
                pulse_t = abs(theta) / J_coupling if J_coupling > 0 else 10e-9
                if hasattr(self.sim, 'apply_exchange'):
                    self.sim.apply_exchange(qi, qj, J_coupling, pulse_t)
                else:
                    # Fallback: decompose exchange into CNOT + rotations
                    # SWAP(theta) ~ Rz(theta/2) CNOT Rz(-theta/2) CNOT
                    self.sim.apply_rz(theta / 2, qi)
                    self.sim.apply_cnot(qi, qj)
                    self.sim.apply_rz(-theta / 2, qi)
                    self.sim.apply_cnot(qi, qj)

                # Apply idle decoherence during pulse duration
                if hasattr(self.sim, 'apply_idle_noise'):
                    idle_t = pulse.duration or pulse_t or 10e-9
                    self.sim.apply_idle_noise(idle_t)

    def _compute_reward(self, fidelity: float) -> float:
        """Compute the reward signal.

        For SLEDGE/DFS devices, includes a leakage penalty.
        """
        if self.reward_type == "sparse":
            base_reward = 1.0 if fidelity >= self.fidelity_threshold else 0.0

        elif self.reward_type == "dense":
            # Fidelity improvement + bonus for high fidelity
            delta_f = fidelity - self._prev_fidelity
            bonus = 0.0
            if fidelity >= self.fidelity_threshold:
                bonus = 10.0
            elif fidelity >= 0.95:
                bonus = 2.0
            base_reward = delta_f + bonus - 0.01  # small step penalty

        elif self.reward_type == "shaped":
            # Potential-based shaping: Phi(s') - Phi(s)
            # Phi(s) = log(fidelity + eps)
            eps = 1e-10
            phi_new = np.log(fidelity + eps)
            phi_old = np.log(self._prev_fidelity + eps)
            shaping = phi_new - phi_old

            # Success bonus
            bonus = 10.0 if fidelity >= self.fidelity_threshold else 0.0
            base_reward = shaping + bonus - 0.005

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        # DFS leakage penalty
        if self.is_dfs:
            leakage_penalty = -5.0 * self._total_leakage
            base_reward += leakage_penalty

        return base_reward

    def render(self):
        """Render the current state."""
        if self.render_mode == "ansi":
            mode_label = "MPO" if self.sim_mode == "mpo" else "MPS"
            dev_label = self.device.name
            if self.grid_shape:
                dev_label += f" ({self.grid_shape[0]}x{self.grid_shape[1]})"

            lines = [
                f"SiliQun [{mode_label}] | {dev_label} | Step {self._step_count}/{self.max_steps}",
                f"Fidelity: {self._prev_fidelity:.6f} (best: {self._best_fidelity:.6f})",
                f"Time: {self.sim.time:.2e} s",
                f"Bond dims: {self.sim.state.bond_dims}",
                f"<Z>: {[f'{self.sim.expectation_z(q):.3f}' for q in range(self.n_qubits)]}",
            ]

            if self.is_dfs:
                lines.append(f"Leakage: {self._total_leakage:.6f}")

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
    sim_mode: str = "auto",
    noise: bool = True,
    max_bond_dim: int = 32,
    max_steps: int = 200,
    fidelity_threshold: float = 0.99,
    reward_type: str = "dense",
    include_purity: bool = False,
    seed: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    use_gpu: bool = True,
) -> SiliQunEnv:
    """Create a SiliQun Gymnasium environment with convenient defaults.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    device : str
        Device type: "donor", "simos", "gaa", or "sledge".
    target : str
        Target state: "bell", "ghz", "w", or "random".
    sim_mode : str
        "auto" (selects best backend automatically),
        "mps" for pure-state (fast, approximate noise),
        "mpo" for density matrix (slower, exact noise), or
        "sv" for GPU-accelerated exact state vector.
    noise : bool
        Whether to enable noise simulation.
    max_bond_dim : int
        Maximum bond dimension (MPS/MPO modes only).
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
    grid_shape : tuple, optional
        (rows, cols) for 2D grid topology (SLEDGE devices).
    use_gpu : bool
        Whether to use GPU acceleration (SV mode only).

    Returns
    -------
    SiliQunEnv
        Configured Gymnasium environment.

    Examples
    --------
    >>> env = make_siliqun_env(n_qubits=2, sim_mode="mps", target="bell")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)

    # SLEDGE 3x3 grid (9 logical qubits, 27 physical spins)
    >>> env = make_siliqun_env(
    ...     n_qubits=9, device="sledge", target="ghz",
    ...     grid_shape=(3, 3), max_steps=500
    ... )

    # SLEDGE 5x5 grid (25 logical qubits) - requires GPU
    >>> env = make_siliqun_env(
    ...     n_qubits=25, device="sledge", target="ghz",
    ...     grid_shape=(5, 5), sim_mode="sv", max_steps=1000
    ... )
    """
    if sim_mode == "mpo":
        config = MPOSimConfig(
            noise_enabled=noise,
            max_bond_dim=max_bond_dim,
            seed=seed,
        )
    elif sim_mode == "sv":
        config = SVSimConfig(
            noise_enabled=noise,
            use_gpu=use_gpu,
            seed=seed or 42,
        )
    elif sim_mode == "auto":
        # Auto-select: use SV for large 2D grids, MPS otherwise
        if n_qubits > 9 and grid_shape is not None:
            config = SVSimConfig(
                noise_enabled=noise,
                use_gpu=use_gpu,
                seed=seed or 42,
            )
        else:
            config = SimConfig(
                noise_enabled=noise,
                max_bond_dim=max_bond_dim,
                seed=seed or 42,
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
        grid_shape=grid_shape,
    )
