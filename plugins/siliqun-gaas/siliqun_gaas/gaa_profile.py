"""
siliqun_gaas.gaa_profile
========================
Minimum plugin contract for gate-all-around (GAA) silicon spin qubits.

This file (28 lines of platform-specific code) is the only profile
class a hardware integrator must write to register a new technology
with SiliQun.  The Lindblad solver, Gymnasium interface, and RL agent
are provided by the SiliQun core — no modifications required.

Source: Tanamoto and Ono, Phys. Rev. Applied 23, 034001 (2025).
DOI:    10.1103/PhysRevApplied.23.034001
"""

import math, json, pathlib
import numpy as np
from siliqun.core.abc import TechnologyProfile, CalibrationRecord

_I  = np.eye(2, dtype=complex)
_Sx = np.array([[0,1],[1,0]], dtype=complex)/2
_Sy = np.array([[0,-1j],[1j,0]], dtype=complex)/2
_Sz = np.array([[1,0],[0,-1]], dtype=complex)/2
_Sm = np.array([[0,1],[0,0]], dtype=complex)

class GAAProfile(TechnologyProfile):
    technology_name  = "GAA-Nominal"
    gymnasium_env_id = "SiliQun-GAA-v1"
    _N = 2

    def device_parameters(self):
        return {"T1":5e-4,"T2":1e-4,"tau1":5e-9,"tau2":50e-9,"p2":0.015,"n_qubits":self._N}

    def drift_hamiltonian(self):
        return np.zeros((4,4), dtype=complex)

    def control_hamiltonians(self):
        return [
            np.kron(_Sx,_I), np.kron(_Sy,_I),
            np.kron(_I,_Sx), np.kron(_I,_Sy),
            np.kron(_Sz,_Sz)+np.kron(_Sx,_Sx)+np.kron(_Sy,_Sy),
        ]

    def noise_channels(self):
        p=self.device_parameters(); T1,T2,tau2,p2=p["T1"],p["T2"],p["tau2"],p["p2"]
        g1=1/T1; gp=1/T2-1/(2*T1); gd=p2/(4*tau2)
        L=[]
        for q in range(self._N):
            ops=[_I]*self._N; ops[q]=_Sm
            Lq=ops[0];  [Lq:=np.kron(Lq,o) for o in ops[1:]]
            L.append(math.sqrt(g1)*Lq)
            ops[q]=2*_Sz; Lq=ops[0]; [Lq:=np.kron(Lq,o) for o in ops[1:]]
            L.append(math.sqrt(gp/2)*Lq)
            for op in [2*_Sx,2*_Sy,2*_Sz]:
                ops[q]=op; Lq=ops[0]; [Lq:=np.kron(Lq,o) for o in ops[1:]]
                L.append(math.sqrt(gd/4)*Lq)
        return L

    def gate_library(self):
        return ["Rx","Ry","Rz","CNOT","CZ"]

    @property
    def calibration_record(self):
        data=json.loads((pathlib.Path(__file__).parent/"gaa_calibration.json").read_text())
        return CalibrationRecord(**data)
