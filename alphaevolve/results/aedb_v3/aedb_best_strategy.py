# Best strategy found by AEDB Orchestrator v3
# Combined score: 0.700922
# Base fidelity:  0.698409
# Final (500):    0.710864
# Model:          ollama-gpu
# Generation:     19
# Source:         evolution
# N qubits:       3
# Target gates:   ['ghz']
# LLM calls:      100
# DEHB evals:     90
# BRFD outer:     705
# Time:           3291.8s

from fitness import Gate

def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    import numpy as np
    gates = []
    
    if target_gate == "bell":
        gates.append(Gate("h", [0]))
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "cnot":
        gates.append(Gate("cnot", [0, 1]))
    elif target_gate == "ghz":
        if nn_correlation < 0.5:
            gates.append(Gate("h", [0]))
            for i in range(n_qubits - 1):
                gates.append(Gate("cnot", [i, i + 1]))
        else:
            phase_corr = np.pi * nn_correlation / corr_length_nm
            spacing_factor = qubit_spacing_nm / corr_length_nm
            
            # Merge consecutive single-qubit rotations
            rz_phase = 0.0
            for i in range(n_qubits):
                if (nn_correlation == 1 or i % 2 == 0) and i < n_qubits - 1:
                    rz_phase += phase_corr / (n_qubits // 2) * spacing_factor
                if i > 0 and (nn_correlation == 1 or i % 2 == 1):
                    rz_phase -= phase_corr * spacing_factor / 4
            
            for i in range(n_qubits):
                gates.append(Gate("rz", [i], {"theta": rz_phase}))
            
            # Additional CNOTs
            for i in range(1, n_qubits):
                gates.append(Gate("cnot", [(i - 1) % n_qubits, i]))
    
    return gates
