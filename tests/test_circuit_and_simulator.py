from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from siliqun import Circuit, StateVectorSimulator, fidelity


ROOT = Path(__file__).resolve().parents[1]


class CircuitAndSimulatorTest(unittest.TestCase):
    def test_bell_state_probabilities(self) -> None:
        circuit = Circuit(2).h(0).cx(0, 1)
        result = StateVectorSimulator(2).run(circuit)
        self.assertTrue(np.allclose(result.probabilities, [0.5, 0.0, 0.0, 0.5]))
        expected = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        self.assertAlmostEqual(fidelity(result.statevector, expected), 1.0, places=12)

    def test_openqasm_subset_matches_direct_circuit(self) -> None:
        source = """
        OPENQASM 3.0;
        include \"stdgates.inc\";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        rz(pi/2) q[1];
        """
        parsed = Circuit.from_openqasm3(source)
        direct = Circuit(2).h(0).cx(0, 1).rz(math.pi / 2, 1)
        first = StateVectorSimulator(2).run(parsed)
        second = StateVectorSimulator(2).run(direct)
        self.assertAlmostEqual(fidelity(first.statevector, second.statevector), 1.0, places=12)

    def test_documented_qasm_example_produces_bell_state(self) -> None:
        source = (ROOT / "examples" / "bell_state.qasm").read_text()
        result = StateVectorSimulator(2).run(Circuit.from_openqasm3(source))
        self.assertTrue(np.allclose(result.probabilities, [0.5, 0.0, 0.0, 0.5]))

    def test_parser_rejects_nonbaseline_control_flow(self) -> None:
        source = "OPENQASM 3.0; qubit[1] q; if (true) x q[0];"
        with self.assertRaises(ValueError):
            Circuit.from_openqasm3(source)

    def test_allocation_bound_is_explicit(self) -> None:
        with self.assertRaises(ValueError):
            StateVectorSimulator(3, max_qubits=2)


if __name__ == "__main__":
    unittest.main()
