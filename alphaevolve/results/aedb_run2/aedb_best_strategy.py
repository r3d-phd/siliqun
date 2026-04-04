# Best strategy found by AEDB Orchestrator
# Combined fitness: 0.751724
# Final fitness (500 samples): 0.912683
# Model: 
# Generation: 0
# Source: seed:standard
# Total LLM calls: 18
# Total DEHB evals: 80
# Total BRFD steps: 40
# Total time: 304.0s

from fitness import Gate

def generate_gate_sequence(target_gate, n_qubits, nn_correlation, qubit_spacing_nm, corr_length_nm):
    """Standard textbook decomposition."""
    gates = []
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
