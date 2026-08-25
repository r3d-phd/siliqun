"""
SiliQun REST API Server.

Exposes the generalised SiliQun silicon spin qubit infrastructure as a
RESTful HTTP API, suitable for deployment on Aziz HPC or any cloud server.

Endpoints
---------
GET  /health                    Server health and version
GET  /devices                   List available device profiles
POST /simulate/circuit          Execute a gate circuit (statevector or Lindblad)
POST /simulate/pulse            Execute a pulse sequence (Lindblad only)
POST /compile                   Compile a gate circuit to pulse sequence
POST /tomography/state          Run quantum state tomography
POST /tomography/process        Run quantum process tomography
POST /fidelity                  Compute state or process fidelity
GET  /jobs/{job_id}             Get async job status and result
POST /jobs/cancel/{job_id}      Cancel a running job

Usage
-----
    uvicorn siliqun.api.server:app --host 0.0.0.0 --port 8000

Or via the CLI:
    python -m siliqun.api

Authentication
--------------
API key authentication via X-API-Key header.
Set SILIQUN_API_KEY environment variable to enable (disabled by default).

Rate Limiting
-------------
Configurable via SILIQUN_MAX_QUBITS (default: 20) and
SILIQUN_MAX_SHOTS (default: 100000).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

SILIQUN_VERSION = "2.0.0-generalised"
MAX_QUBITS = int(os.environ.get("SILIQUN_MAX_QUBITS", 20))
MAX_SHOTS = int(os.environ.get("SILIQUN_MAX_SHOTS", 100000))
API_KEY = os.environ.get("SILIQUN_API_KEY", None)

app = FastAPI(
    title="SiliQun API",
    description=(
        "Near-realistic silicon spin qubit infrastructure platform. "
        "Provides circuit simulation, pulse-level simulation, "
        "gate-to-pulse compilation, and quantum tomography."
    ),
    version=SILIQUN_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job store: job_id -> {status, result, error, created_at}
_job_store: Dict[str, Dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if SILIQUN_API_KEY is set."""
    if API_KEY is not None:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GateInstruction(BaseModel):
    """A single gate instruction in a circuit."""
    gate: str = Field(..., description="Gate name (rx, ry, rz, h, cnot, cz, ...)")
    qubits: List[int] = Field(..., description="Target qubit indices")
    params: Dict[str, float] = Field(default={}, description="Gate parameters (e.g., theta)")

    class Config:
        schema_extra = {
            "example": {"gate": "rx", "qubits": [0], "params": {"theta": 1.5708}}
        }


class DeviceConfig(BaseModel):
    """Device configuration for simulation."""
    device: str = Field(
        default="simos",
        description="Device profile: donor, simos, gaa, sledge"
    )
    n_qubits: int = Field(default=4, ge=1, le=20, description="Number of qubits")
    T1: Optional[float] = Field(default=None, description="T1 relaxation time (s)")
    T2_star: Optional[float] = Field(default=None, description="T2* dephasing time (s)")
    charge_noise: Optional[float] = Field(default=None, description="Charge noise amplitude")
    J_max_hz: Optional[float] = Field(default=None, description="Max exchange coupling (Hz)")

    @validator("device")
    def validate_device(cls, v):
        valid = {"donor", "simos", "gaa", "sledge"}
        if v.lower() not in valid:
            raise ValueError(f"device must be one of {valid}")
        return v.lower()


class CircuitSimRequest(BaseModel):
    """Request for circuit simulation."""
    circuit: List[GateInstruction] = Field(..., description="Gate circuit")
    device: DeviceConfig = Field(default=DeviceConfig(), description="Device configuration")
    sim_mode: str = Field(
        default="auto",
        description="Simulation mode: auto, sv (statevector), mps, lindblad"
    )
    n_shots: int = Field(default=1000, ge=1, le=100000, description="Measurement shots")
    observables: List[str] = Field(
        default=["Z"] * 4,
        description="Pauli observables to measure (e.g., ['ZI', 'IZ', 'ZZ'])"
    )
    return_density_matrix: bool = Field(
        default=False,
        description="Return full density matrix (only for Lindblad mode)"
    )
    return_statevector: bool = Field(
        default=False,
        description="Return full statevector (only for SV mode)"
    )

    class Config:
        schema_extra = {
            "example": {
                "circuit": [
                    {"gate": "h", "qubits": [0], "params": {}},
                    {"gate": "cnot", "qubits": [0, 1], "params": {}},
                ],
                "device": {"device": "simos", "n_qubits": 2},
                "sim_mode": "auto",
                "n_shots": 1000,
                "observables": ["ZI", "IZ", "ZZ"],
            }
        }


class PulseSimRequest(BaseModel):
    """Request for pulse-level simulation (Lindblad only)."""
    pulses: List[Dict[str, Any]] = Field(..., description="List of pulse dictionaries")
    device: DeviceConfig = Field(default=DeviceConfig())
    dt: float = Field(default=1e-10, description="Time step (s)")
    return_density_matrix: bool = Field(default=True)


class CompileRequest(BaseModel):
    """Request to compile a gate circuit to a pulse sequence."""
    circuit: List[GateInstruction] = Field(..., description="Gate circuit to compile")
    device: DeviceConfig = Field(default=DeviceConfig())
    schedule_mode: str = Field(default="sequential", description="sequential or parallel")
    qasm: Optional[str] = Field(default=None, description="OpenQASM 2.0 string (alternative to circuit)")


class TomographyRequest(BaseModel):
    """Request for state tomography."""
    circuit: List[GateInstruction] = Field(..., description="Circuit to characterise")
    device: DeviceConfig = Field(default=DeviceConfig())
    n_shots: int = Field(default=10000, ge=100, le=100000)
    method: str = Field(default="linear", description="linear or mle")
    target_state: Optional[List[List[float]]] = Field(
        default=None,
        description="Target state as flat list of [re, im] pairs for fidelity"
    )


class FidelityRequest(BaseModel):
    """Request to compute state fidelity."""
    rho: List[List[float]] = Field(..., description="Density matrix as flat [re, im] list")
    sigma: List[List[float]] = Field(..., description="Target density matrix")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    max_qubits: int
    max_shots: int
    devices: List[str]


class DeviceInfo(BaseModel):
    name: str
    n_qubits_max: int
    T1_us: float
    T2_star_us: float
    J_max_mhz: float
    gate_time_1q_ns: float
    gate_time_2q_ns: float
    native_gates: List[str]
    dfs_encoded: bool


class SimulationResult(BaseModel):
    job_id: str
    status: str
    n_qubits: int
    device: str
    sim_mode: str
    expectations: Dict[str, float]
    fidelity: Optional[float] = None
    purity: Optional[float] = None
    elapsed_s: float
    density_matrix: Optional[List] = None
    statevector: Optional[List] = None
    warnings: List[str] = []


class CompilationResponse(BaseModel):
    job_id: str
    n_gates: int
    n_exchange_pulses: int
    n_drive_pulses: int
    total_fidelity: float
    total_duration_ns: float
    fidelity_breakdown: Dict[str, float]
    warnings: List[str]
    pulse_sequence: Optional[List[Dict]] = None


class TomographyResponse(BaseModel):
    job_id: str
    n_qubits: int
    purity: float
    fidelity_to_target: Optional[float]
    method: str
    density_matrix: Optional[List] = None
    pauli_expectations: Optional[Dict[str, float]] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    elapsed_s: Optional[float]
    result: Optional[Any] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_device_profile(config: DeviceConfig):
    """Load a device profile from the SiliQun physics module."""
    from siliqun.physics.devices.profiles import get_device_profile
    device = get_device_profile(config.device, n_qubits=config.n_qubits)

    # Override with user-specified parameters
    if config.T1 is not None:
        device.noise_params.t1_times = [config.T1] * config.n_qubits
    if config.T2_star is not None:
        device.noise_params.t2_star_times = [config.T2_star] * config.n_qubits
    if config.charge_noise is not None:
        device.noise_params.charge_noise_amplitude = config.charge_noise
    if config.J_max_hz is not None:
        device.hamiltonian_params.exchange_couplings = [
            config.J_max_hz
        ] * max(0, config.n_qubits - 1)

    return device


def _circuit_to_tuples(circuit: List[GateInstruction]):
    """Convert GateInstruction list to (gate_name, params, qubits) tuples."""
    return [(g.gate, g.params, g.qubits) for g in circuit]


def _run_circuit_simulation(request: CircuitSimRequest) -> Dict:
    """Execute circuit simulation (runs in thread pool)."""
    t0 = time.time()
    device = _get_device_profile(request.device)
    circuit = _circuit_to_tuples(request.circuit)
    warnings = []

    # Choose simulator
    sim_mode = request.sim_mode
    n_qubits = request.device.n_qubits

    if sim_mode == "auto":
        # Auto-select: SV for ≤9 qubits, MPS for larger
        sim_mode = "sv" if n_qubits <= 9 else "mps"

    if sim_mode in ("sv", "statevector"):
        from siliqun.engine.statevector_simulator import StatevectorSimulator
        sim = StatevectorSimulator(device=device)
        result = sim.run(circuit)
        sv = result.statevector
        rho = np.outer(sv, sv.conj())
    elif sim_mode == "lindblad":
        from siliqun.pulse.lindblad import LindbladSimulator
        from siliqun.compiler.gate_compiler import GateToPulseCompiler
        compiler = GateToPulseCompiler(device)
        compiled = compiler.compile(circuit)
        warnings.extend(compiled.warnings)
        sim = LindbladSimulator(device=device, n_qubits=n_qubits)
        result = sim.evolve(compiled.sequence)
        rho = result.rho_final
    else:  # mps
        from siliqun.engine.simulator import MPSSimulator
        sim = MPSSimulator(device=device)
        result = sim.run(circuit)
        sv = result.statevector
        rho = np.outer(sv, sv.conj())

    # Compute observables
    from siliqun.tomography.tomography import pauli_expectation, _tensor_pauli, purity as _purity
    expectations = {}
    for obs in request.observables:
        labels = list(obs.upper())
        if len(labels) != n_qubits:
            warnings.append(f"Observable '{obs}' has wrong length for {n_qubits} qubits")
            continue
        op = _tensor_pauli(labels)
        expectations[obs] = pauli_expectation(rho, op)

    p = _purity(rho)

    output = {
        "n_qubits": n_qubits,
        "device": request.device.device,
        "sim_mode": sim_mode,
        "expectations": expectations,
        "purity": p,
        "elapsed_s": time.time() - t0,
        "warnings": warnings,
    }

    if request.return_density_matrix:
        output["density_matrix"] = rho.tolist()
    if request.return_statevector and sim_mode in ("sv", "statevector"):
        output["statevector"] = sv.tolist()

    return output


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check server health and capabilities."""
    return HealthResponse(
        status="online",
        version=SILIQUN_VERSION,
        timestamp=time.time(),
        max_qubits=MAX_QUBITS,
        max_shots=MAX_SHOTS,
        devices=["donor", "simos", "gaa", "sledge"],
    )


@app.get("/devices", tags=["Devices"])
async def list_devices():
    """List available silicon spin qubit device profiles with parameters."""
    # Device parameters from Burkard et al., Rev. Mod. Phys. 95, 025003 (2023)
    devices = {
        "donor": DeviceInfo(
            name="donor",
            n_qubits_max=6,
            T1_us=1e6,        # ~1 second (exceptional T1)
            T2_star_us=100,   # ~100 µs
            J_max_mhz=1.0,    # ~1 MHz exchange
            gate_time_1q_ns=200,
            gate_time_2q_ns=2000,
            native_gates=["rx", "ry", "rz", "cnot"],
            dfs_encoded=False,
        ),
        "simos": DeviceInfo(
            name="simos",
            n_qubits_max=9,
            T1_us=10000,      # ~10 ms
            T2_star_us=20,    # ~20 µs
            J_max_mhz=50.0,   # ~50 MHz exchange
            gate_time_1q_ns=50,
            gate_time_2q_ns=200,
            native_gates=["rx", "ry", "rz", "cz", "sqrt_swap"],
            dfs_encoded=False,
        ),
        "gaa": DeviceInfo(
            name="gaa",
            n_qubits_max=12,
            T1_us=5000,
            T2_star_us=10,
            J_max_mhz=100.0,
            gate_time_1q_ns=30,
            gate_time_2q_ns=100,
            native_gates=["rx", "ry", "rz", "cz"],
            dfs_encoded=False,
        ),
        "sledge": DeviceInfo(
            name="sledge",
            n_qubits_max=6,
            T1_us=100000,
            T2_star_us=1000,
            J_max_mhz=10.0,
            gate_time_1q_ns=500,
            gate_time_2q_ns=5000,
            native_gates=["exchange", "rx", "ry", "rz"],
            dfs_encoded=True,
        ),
    }
    return devices


@app.post("/simulate/circuit", tags=["Simulation"])
async def simulate_circuit(
    request: CircuitSimRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    """Execute a gate circuit on a silicon spin qubit simulator.

    Supports three simulation modes:
    - **auto**: Automatically selects SV (≤9 qubits) or MPS (>9 qubits)
    - **sv**: GPU-accelerated statevector simulation (≤25 qubits)
    - **mps**: Matrix Product State simulation (any qubit count)
    - **lindblad**: Full density matrix simulation with noise (≤12 qubits)

    Returns Pauli expectation values and optionally the full density matrix.
    """
    if request.device.n_qubits > MAX_QUBITS:
        raise HTTPException(
            status_code=400,
            detail=f"n_qubits={request.device.n_qubits} exceeds MAX_QUBITS={MAX_QUBITS}"
        )

    job_id = str(uuid.uuid4())[:8]
    _job_store[job_id] = {
        "status": "running",
        "created_at": time.time(),
        "result": None,
        "error": None,
    }

    def _run():
        try:
            result = _run_circuit_simulation(request)
            result["job_id"] = job_id
            result["status"] = "completed"
            _job_store[job_id]["status"] = "completed"
            _job_store[job_id]["result"] = result
        except Exception as e:
            logger.exception("Simulation failed: %s", e)
            _job_store[job_id]["status"] = "failed"
            _job_store[job_id]["error"] = str(e)

    background_tasks.add_task(_run)

    return {"job_id": job_id, "status": "submitted"}


@app.post("/compile", response_model=CompilationResponse, tags=["Compilation"])
async def compile_circuit(
    request: CompileRequest,
    _: str = Depends(verify_api_key),
):
    """Compile a gate circuit into a silicon spin qubit pulse sequence.

    Uses the exchange interaction as the native two-qubit primitive.
    Returns per-gate fidelity estimates (Q-Forge validated) and
    the full pulse schedule.
    """
    from siliqun.compiler.gate_compiler import GateToPulseCompiler

    device = _get_device_profile(request.device)
    compiler = GateToPulseCompiler(device, schedule_mode=request.schedule_mode)

    if request.qasm:
        result = compiler.compile_qasm(request.qasm)
    else:
        circuit = _circuit_to_tuples(request.circuit)
        result = compiler.compile(circuit)

    # Serialise pulse sequence
    pulse_list = []
    for p in result.sequence.pulses:
        pulse_dict = {
            "type": type(p).__name__,
            "t_start": getattr(p, "t_start", 0.0),
            "duration": getattr(p, "duration", 0.0),
        }
        if hasattr(p, "qubit_i"):
            pulse_dict.update({
                "qubit_i": p.qubit_i,
                "qubit_j": p.qubit_j,
                "J_hz": p.J,
                "shape": p.shape,
            })
        elif hasattr(p, "qubit"):
            pulse_dict.update({
                "qubit": p.qubit,
                "amplitude": p.amplitude,
                "phase": p.phase,
            })
        pulse_list.append(pulse_dict)

    job_id = str(uuid.uuid4())[:8]

    return CompilationResponse(
        job_id=job_id,
        n_gates=len(result.compiled_gates),
        n_exchange_pulses=result.n_exchange_pulses,
        n_drive_pulses=result.n_drive_pulses,
        total_fidelity=result.total_fidelity,
        total_duration_ns=result.total_duration * 1e9,
        fidelity_breakdown=result.fidelity_breakdown(),
        warnings=result.warnings,
        pulse_sequence=pulse_list,
    )


@app.post("/tomography/state", response_model=TomographyResponse, tags=["Tomography"])
async def state_tomography(
    request: TomographyRequest,
    _: str = Depends(verify_api_key),
):
    """Run quantum state tomography on a circuit.

    Reconstructs the density matrix from Pauli expectation value measurements.
    Supports linear inversion and Maximum Likelihood Estimation (MLE).
    """
    from siliqun.tomography import StateTomography

    device = _get_device_profile(request.device)
    n_qubits = request.device.n_qubits

    # Run circuit simulation to get density matrix
    sim_request = CircuitSimRequest(
        circuit=request.circuit,
        device=request.device,
        sim_mode="lindblad" if n_qubits <= 12 else "sv",
        return_density_matrix=True,
    )
    sim_result = _run_circuit_simulation(sim_request)
    rho_list = sim_result.get("density_matrix")
    if rho_list is None:
        raise HTTPException(status_code=500, detail="Simulation did not return density matrix")

    rho = np.array(rho_list)

    # Run state tomography
    qst = StateTomography(n_qubits=n_qubits, method=request.method)
    expectations = qst.simulate_measurements(rho, n_shots=request.n_shots)

    target = None
    if request.target_state is not None:
        target_flat = [complex(r, i) for r, i in request.target_state]
        d = 2 ** n_qubits
        target = np.array(target_flat).reshape(d, d)

    result = qst.reconstruct(expectations, target_state=target, n_shots=request.n_shots)

    job_id = str(uuid.uuid4())[:8]

    return TomographyResponse(
        job_id=job_id,
        n_qubits=n_qubits,
        purity=result.purity,
        fidelity_to_target=result.fidelity_to_target,
        method=result.method,
        density_matrix=result.rho.tolist(),
        pauli_expectations=result.pauli_expectations,
    )


@app.post("/fidelity", tags=["Metrics"])
async def compute_fidelity(
    request: FidelityRequest,
    _: str = Depends(verify_api_key),
):
    """Compute quantum state fidelity F(rho, sigma).

    Accepts density matrices as lists of [real, imag] pairs.
    """
    from siliqun.tomography import fidelity as _fidelity

    def _parse_matrix(data):
        flat = [complex(r, i) for r, i in data]
        d = int(np.sqrt(len(flat)))
        return np.array(flat).reshape(d, d)

    rho = _parse_matrix(request.rho)
    sigma = _parse_matrix(request.sigma)
    f = _fidelity(rho, sigma)
    return {"fidelity": f}


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job(job_id: str):
    """Get the status and result of an async simulation job."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _job_store[job_id]
    elapsed = time.time() - job["created_at"] if job["status"] == "running" else None
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        elapsed_s=elapsed,
        result=job.get("result"),
        error=job.get("error"),
    )


@app.post("/jobs/cancel/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str):
    """Cancel a running simulation job."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    _job_store[job_id]["status"] = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "siliqun.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
