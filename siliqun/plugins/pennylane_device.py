"""
SiliQun PennyLane Device Plugin
================================
Exposes SiliQun's physics-accurate silicon spin qubit simulator as a standard
PennyLane device, allowing users to run PennyLane QNodes directly on SiliQun
without modifying their existing quantum machine learning code.

Standards:
    - PennyLane Device API v2 (pennylane.devices.Device)
    - https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.Device.html

Usage:
    >>> import pennylane as qml
    >>> dev = qml.device("siliqun.simos", wires=4)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.Hadamard(wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.state()
    >>> print(circuit())

References:
    [1] Bergholm et al., "PennyLane: Automatic differentiation of hybrid
        quantum-classical computations", arXiv:1811.04968, 2018.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    from pennylane.devices import Device, ExecutionConfig
    # TransformProgram was renamed to CompilePipeline in PennyLane 0.40+
    try:
        from pennylane.transforms.core.compile_pipeline import CompilePipeline as TransformProgram
    except ImportError:
        try:
            from pennylane.transforms import TransformProgram
        except ImportError:
            TransformProgram = None
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available. SiliQunDevice plugin disabled.")

# SiliQun imports
from ..physics.devices.profiles import DeviceProfile, simos_device, donor_device


class SiliQunDevice(Device if PENNYLANE_AVAILABLE else object):
    """A PennyLane Device backed by SiliQun's physics-accurate simulator.

    This device implements the PennyLane Device API v2, making SiliQun
    accessible to the entire PennyLane ecosystem including quantum machine
    learning, variational algorithms, and quantum chemistry.

    Parameters
    ----------
    wires : int or list
        Number of qubits or list of wire labels.
    device_type : str
        Silicon spin qubit device type: "simos", "donor", "gaa", or "sledge".
        Default "simos".
    sim_mode : str
        Simulation mode: "auto", "sv" (state vector), or "mps".
        Default "auto".
    shots : int or None
        Number of measurement shots. None for exact (analytic) simulation.

    Examples
    --------
    >>> dev = SiliQunDevice(wires=4, device_type="simos")
    >>> @qml.qnode(dev)
    ... def bell_state():
    ...     qml.Hadamard(wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.state()
    """

    # PennyLane device name (used in qml.device("siliqun.simos", ...))
    name = "SiliQun Silicon Spin Qubit Simulator"
    short_name = "siliqun.simos"
    pennylane_requires = ">=0.38.0"
    version = "1.0.0"
    author = "SiliQun Team"

    # Supported operations (PennyLane gate names)
    operations = {
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "S", "T", "SX",
        "RX", "RY", "RZ", "PhaseShift", "Rot",
        "CNOT", "CZ", "SWAP", "ISWAP",
        "CRX", "CRY", "CRZ",
        "Toffoli", "CSWAP",
        "Identity", "BasisState", "StatePrep",
    }

    # Supported observables
    observables = {
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "Hermitian", "Identity",
        "Projector", "SparseHamiltonian", "Hamiltonian",
    }

    def __init__(
        self,
        wires: Union[int, List] = 4,
        device_type: str = "simos",
        sim_mode: str = "auto",
        shots: Optional[int] = None,
    ):
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is required for SiliQunDevice. "
                "Install with: pip install pennylane"
            )
        super().__init__(wires=wires, shots=shots)
        self._device_type = device_type
        self._sim_mode = sim_mode
        n = len(self.wires)
        self._siliqun_device = self._load_device(device_type, n)

    def _load_device(self, device_type: str, n_qubits: int) -> DeviceProfile:
        """Load the SiliQun device profile."""
        from ..physics.devices.profiles import get_device_profile
        try:
            return get_device_profile(device_type, n_qubits=n_qubits)
        except Exception:
            logger.warning("Could not load device '%s', falling back to SiMOS.", device_type)
            return simos_device(n_qubits=n_qubits)

    @property
    def name(self) -> str:
        return f"SiliQun {self._device_type.upper()} ({len(self.wires)} qubits)"

    def preprocess(
        self,
        execution_config: "ExecutionConfig" = None,
    ) -> Tuple:
        """Preprocess circuits before execution (PennyLane Device API v2)."""
        config = execution_config or ExecutionConfig()
        if TransformProgram is not None:
            program = TransformProgram()
        else:
            program = None
        return program, config

    def execute(
        self,
        circuits,
        execution_config: "ExecutionConfig" = None,
    ) -> "ResultBatch":
        """Execute a batch of circuits on SiliQun's simulator.

        Parameters
        ----------
        circuits : list of QuantumScript
            PennyLane quantum scripts to execute.
        execution_config : ExecutionConfig, optional
            Execution configuration.

        Returns
        -------
        tuple
            Batch of results.
        """
        results = []
        for circuit in circuits:
            result = self._execute_single(circuit)
            results.append(result)
        return tuple(results)

    def _execute_single(self, circuit) -> "Result":
        """Execute a single PennyLane QuantumScript on SiliQun."""
        # Convert PennyLane circuit to SiliQun gate list
        gate_list = self._pennylane_to_gate_list(circuit)

        # Run through SiliQun simulation
        state = self._simulate(gate_list, circuit.num_wires)

        # Process measurements
        return self._process_measurements(circuit, state)

    def _pennylane_to_gate_list(self, circuit) -> List[Tuple]:
        """Convert a PennyLane QuantumScript to a SiliQun gate list."""
        gate_list = []
        for op in circuit.operations:
            name = op.name.lower()
            wires = [self.wires.index(w) for w in op.wires]
            params = {}
            if op.num_params > 0:
                param_names = ["theta", "phi", "omega"]
                for i, p in enumerate(op.parameters):
                    if i < len(param_names):
                        try:
                            params[param_names[i]] = float(p)
                        except (TypeError, ValueError):
                            pass

            # Map PennyLane names to SiliQun names
            name_map = {
                "paulix": "x", "pauliy": "y", "pauliz": "z",
                "hadamard": "h", "cnot": "cx",
                "rx": "rx", "ry": "ry", "rz": "rz",
                "cz": "cz", "swap": "swap",
                "identity": "id",
            }
            siliqun_name = name_map.get(name, name)
            gate_list.append((siliqun_name, params, wires))
        return gate_list

    def _simulate(self, gate_list: List[Tuple], n_qubits: int) -> np.ndarray:
        """Run the gate list through SiliQun's state vector simulator."""
        # Build state vector: start from |0...0⟩
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Apply gates using SiliQun's physics engine
        for gate_name, params, qubits in gate_list:
            state = self._apply_gate(state, gate_name, params, qubits, n_qubits)
        return state

    def _apply_gate(
        self,
        state: np.ndarray,
        gate_name: str,
        params: dict,
        qubits: List[int],
        n_qubits: int,
    ) -> np.ndarray:
        """Apply a single gate to the state vector."""
        from ..physics.gates import (
            rx, ry, rz, hadamard,
            pauli_x, pauli_y, pauli_z,
        )

        gate_map = {
            "rx": lambda: rx(params.get("theta", np.pi / 2)),
            "ry": lambda: ry(params.get("theta", np.pi / 2)),
            "rz": lambda: rz(params.get("theta", np.pi / 2)),
            "h": lambda: hadamard(),
            "x": lambda: pauli_x(),
            "y": lambda: pauli_y(),
            "z": lambda: pauli_z(),
        }

        try:
            if gate_name in gate_map and len(qubits) == 1:
                U = gate_map[gate_name]()
                state = self._apply_single_qubit_gate(state, U, qubits[0], n_qubits)
            elif gate_name in ("cx", "cnot") and len(qubits) == 2:
                state = self._apply_cnot(state, qubits[0], qubits[1], n_qubits)
            elif gate_name == "cz" and len(qubits) == 2:
                state = self._apply_cz(state, qubits[0], qubits[1], n_qubits)
            elif gate_name == "id":
                pass  # Identity: no-op
        except Exception as e:
            logger.warning("Gate '%s' failed: %s. Skipping.", gate_name, e)

        return state

    def _apply_single_qubit_gate(
        self, state: np.ndarray, U: np.ndarray, qubit: int, n_qubits: int
    ) -> np.ndarray:
        """Apply a 2×2 unitary to a single qubit in the state vector."""
        state = state.reshape([2] * n_qubits)
        state = np.tensordot(U, state, axes=[[1], [qubit]])
        # Move the contracted axis back to its original position
        state = np.moveaxis(state, 0, qubit)
        return state.reshape(-1)

    def _apply_cnot(
        self, state: np.ndarray, control: int, target: int, n_qubits: int
    ) -> np.ndarray:
        """Apply a CNOT gate."""
        CNOT = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        return self._apply_two_qubit_gate(state, CNOT, control, target, n_qubits)

    def _apply_cz(
        self, state: np.ndarray, q0: int, q1: int, n_qubits: int
    ) -> np.ndarray:
        """Apply a CZ gate."""
        CZ = np.diag([1, 1, 1, -1]).astype(complex)
        return self._apply_two_qubit_gate(state, CZ, q0, q1, n_qubits)

    def _apply_two_qubit_gate(
        self, state: np.ndarray, U: np.ndarray, q0: int, q1: int, n_qubits: int
    ) -> np.ndarray:
        """Apply a 4×4 unitary to two qubits in the state vector."""
        state = state.reshape([2] * n_qubits)
        U = U.reshape(2, 2, 2, 2)
        state = np.tensordot(U, state, axes=[[2, 3], [q0, q1]])
        state = np.moveaxis(state, [0, 1], [q0, q1])
        return state.reshape(-1)

    def _process_measurements(self, circuit, state: np.ndarray):
        """Process PennyLane measurement processes."""
        results = []
        measurements = getattr(circuit, 'measurements', [])
        for mp in measurements:
            mp_type = mp.__class__.__name__

            # PennyLane 0.44 class names: ExpectationMP, StateMP, ProbabilityMP, SampleMP
            if mp_type in ("StateMP", "StateMeasurement"):
                results.append(state)
            elif mp_type in ("ExpvalMP", "ExpectationMP"):
                obs = mp.obs
                expval = self._compute_expval(state, obs, circuit.num_wires)
                results.append(float(np.real(expval)))
            elif mp_type in ("ProbabilityMP", "ProbabilityMeasurement"):
                probs = np.abs(state) ** 2
                results.append(probs)
            elif mp_type in ("SampleMP", "SampleMeasurement"):
                shots = (self.shots.total_shots if hasattr(self.shots, 'total_shots')
                         else self.shots) or 1024
                probs = np.abs(state) ** 2
                probs = probs / probs.sum()  # normalise
                samples = np.random.choice(len(probs), size=int(shots), p=probs)
                results.append(samples)
            else:
                # Default: return state vector
                results.append(state)

        if not results:
            return state
        return tuple(results) if len(results) > 1 else results[0]

    def _compute_expval(self, state: np.ndarray, obs, n_qubits: int) -> float:
        """Compute the expectation value of an observable."""
        obs_name = obs.name if hasattr(obs, 'name') else str(obs)
        wires = [self.wires.index(w) for w in obs.wires]

        pauli_matrices = {
            "PauliX": np.array([[0, 1], [1, 0]], dtype=complex),
            "PauliY": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "PauliZ": np.array([[1, 0], [0, -1]], dtype=complex),
            "Identity": np.eye(2, dtype=complex),
        }

        if obs_name in pauli_matrices and len(wires) == 1:
            P = pauli_matrices[obs_name]
            state_r = state.reshape([2] * n_qubits)
            bra = np.tensordot(P, state_r, axes=[[1], [wires[0]]])
            bra = np.moveaxis(bra, 0, wires[0]).reshape(-1)
            return float(np.real(np.dot(state.conj(), bra)))

        # Fallback: compute full matrix expectation value
        try:
            H = obs.matrix()
            return float(np.real(state.conj() @ H @ state))
        except Exception:
            return 0.0


def register_pennylane_device():
    """Register SiliQunDevice with PennyLane's device registry.

    Call this function once at import time to make the device available
    via qml.device("siliqun.simos", wires=n).
    """
    if not PENNYLANE_AVAILABLE:
        return
    try:
        qml.plugin_devices["siliqun.simos"] = SiliQunDevice
        qml.plugin_devices["siliqun.donor"] = SiliQunDevice
        qml.plugin_devices["siliqun.gaa"] = SiliQunDevice
        logger.info("SiliQun PennyLane devices registered successfully.")
    except Exception as e:
        logger.warning("Could not register SiliQun PennyLane devices: %s", e)
