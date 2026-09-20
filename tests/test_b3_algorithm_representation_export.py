from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools_run_v2_b3_algorithm_representation_export import fixed_representations


class B3AlgorithmRepresentationExportTest(unittest.TestCase):
    def test_static_suite_is_fixed_native_and_bounded(self) -> None:
        representations = dict(fixed_representations())
        self.assertEqual(
            list(representations),
            [
                "deutsch_jozsa_two_qubit_balanced_native",
                "grover_two_qubit_fixed_phase_oracle_native",
                "shor_n15_order4_compiled_orbit_component_native",
            ],
        )
        self.assertEqual([circuit.n_qubits for circuit in representations.values()], [2, 2, 3])
        self.assertTrue(all({gate.name for gate in circuit.gates} <= {"rx", "ry", "rz", "cz"} for circuit in representations.values()))
        self.assertTrue(all(circuit.gates for circuit in representations.values()))
        self.assertGreater(len(representations["shor_n15_order4_compiled_orbit_component_native"].gates), 20)

    def test_static_specification_and_readiness_pass(self) -> None:
        subprocess.run([sys.executable, "tools_validate_v2_b3_algorithm_representation.py"], cwd=ROOT, check=True)
        static = json.loads((ROOT / "docs" / "V2_B3_ALGORITHM_REPRESENTATION_STATIC_SPECIFICATION_RECEIPT_V1.json").read_text())
        readiness = json.loads((ROOT / "docs" / "V2_B3_ALGORITHM_REPRESENTATION_READINESS_RECEIPT_V1.json").read_text())
        self.assertEqual(static["status"], "PASS")
        self.assertEqual(static["row_count"], 3)
        self.assertTrue(all(set(row["native_gate_names"]) <= {"rx", "ry", "rz", "cz"} for row in static["rows"]))
        self.assertEqual(readiness["status"], "PASS")
        self.assertTrue(all(readiness["checks"].values()))

    def test_persisted_export_retains_only_declared_fields(self) -> None:
        receipt = json.loads((ROOT / "docs" / "V2_B3_ALGORITHM_REPRESENTATION_EXPORT_RECEIPT_V1.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["suite_count"], 3)
        self.assertTrue(all(value is False for value in receipt["retention"].values()))
        self.assertTrue(all(value is False for value in receipt["scope"].values()))
        self.assertTrue(all(len(row["siliqun_ideal_probability_sha256"]) == 64 for row in receipt["suite"]))


if __name__ == "__main__":
    unittest.main()
