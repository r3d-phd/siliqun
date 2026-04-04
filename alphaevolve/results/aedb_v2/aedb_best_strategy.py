# Best strategy found by AEDB Orchestrator v2
# Combined score: 0.768399
# Base fidelity:  0.909664
# Final (500):    0.912683
# Model:          ollama-gpu
# Generation:     10
# Source:         evolution
# LLM calls:     30
# DEHB evals:    216
# BRFD steps:    108
# Time:          707.6s

from fitness import Gate

def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    gates = []
    
    if nn_correlation > 0.7:
        for i in range(n_qubits):
            angle = np.pi * (i + 1) / n_qubits
            ry_angle = np.arccos(nn_correlation) * angle
            gates.append(Gate("ry", [i], {"angle": ry_angle}))
        
        if target_gate == "bell":
            gates.append(Gate("h", [0]))
            gates.append(Gate("cnot", [0, 1]))
            gates.append(Gate("h", [1]))
            gates.append(Gate("cnot", [1, 0]))
            gates.append(Gate("h", [0]))
        elif target_gate == "cnot":
            gates.append(Gate("cnot", [0, 1]))
        elif target_gate == "ghz":
            gates.append(Gate("h", [0]))
            for i in range(n_qubits - 1):
                gates.append(Gate("cnot", [i, i + 1]))
        
        for i in range(n_qubits-1, -1, -1):
            angle = np.pi * (i + 1) / n_qubits
            ry_angle = np.arccos(nn_correlation) * angle
            gates.append(Gate("ry", [i], {"angle": ry_angle}))
    else:
        if target_gate == "bell":
            gates.append(Gate("h", [0]))
            gates.append(Gate("cnot", [0, 1]))
        elif target_gate == "cnot":
            gates.append(Gate("cnot", [0, 1]))
        elif target_gate == "ghz":
            gates.append(Gate("h", [0]))
            for i in range(n_qubits - 1):
                gates.append(Gate("cnot", [i, i + 1]))
    
    return gates
