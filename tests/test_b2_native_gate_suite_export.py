from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools_run_v2_b2_native_gate_suite_export import fixed_circuits


class B2NativeGateSuiteExportTest(unittest.TestCase):
    def test_fixed_suite_has_declared_non_algorithm_coverage(self) -> None:
        circuits = dict(fixed_circuits())
        self.assertEqual(len(circuits), 5)
        # Coverage C1: non-B1 x-axis rotation angle.
        self.assertEqual([gate.name for gate in circuits["native_single_rx_pi_over_3"].gates], ["rx"])
        self.assertAlmostEqual(circuits["native_single_rx_pi_over_3"].gates[0].parameters[0], math.pi / 3)
        # Coverage C2: signed y-axis angle and virtual-z composition.
        second = circuits["native_single_ry_minus_pi_over_4_rz_pi_over_5"].gates
        self.assertEqual([gate.name for gate in second], ["ry", "rz"])
        self.assertAlmostEqual(second[0].parameters[0], -math.pi / 4)
        self.assertAlmostEqual(second[1].parameters[0], math.pi / 5)
        # Coverage C3: relative phase applied after x-basis preparation.
        phase_after_x = circuits["native_single_rx_pi_over_2_rz_pi_over_2"].gates
        self.assertEqual([gate.name for gate in phase_after_x], ["rx", "rz"])
        self.assertAlmostEqual(phase_after_x[0].parameters[0], math.pi / 2)
        self.assertAlmostEqual(phase_after_x[1].parameters[0], math.pi / 2)
        # Coverage C4: q0/q1 ordering and CZ phase followed by local rotations.
        ordered = circuits["native_two_qubit_ordered_cz_interference"].gates
        self.assertEqual([gate.name for gate in ordered], ["ry", "rx", "cz", "rz", "ry", "rx"])
        self.assertEqual(ordered[0].targets, (0,))
        self.assertEqual(ordered[1].targets, (1,))
        self.assertEqual(ordered[2].targets, (0, 1))
        self.assertAlmostEqual(ordered[3].parameters[0], math.pi / 7)
        self.assertAlmostEqual(ordered[4].parameters[0], math.pi / 4)
        self.assertAlmostEqual(ordered[5].parameters[0], math.pi / 6)
        # Coverage C5: global phase is intentionally tested only through probabilities.
        self.assertAlmostEqual(circuits["native_single_rz_two_pi_probability_invariant"].gates[0].parameters[0], 2 * math.pi)

    def test_persisted_export_retains_only_declared_fields(self) -> None:
        receipt = json.loads((ROOT / "docs" / "V2_B2_NATIVE_GATE_SUITE_EXPORT_RECEIPT_V1.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["suite_count"], 5)
        self.assertEqual(receipt["profile"]["calibration_status"], "literature_parameterised")
        self.assertTrue(all(value is False for value in receipt["retention"].values()))
        self.assertTrue(all(value is False for value in receipt["scope"].values()))
        rendered = json.dumps(receipt["suite"], sort_keys=True).lower()
        for forbidden in ("statevector", "probabilities", "amplitude", "phase_rad", "duration_s", "fidelity", "reward"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(len({row["siliqun_ideal_probability_sha256"] for row in receipt["suite"]}), 5)

    def test_readiness_receipt_passes(self) -> None:
        subprocess.run([sys.executable, "tools_validate_v2_b2_native_gate_suite.py"], cwd=ROOT, check=True)
        receipt = json.loads((ROOT / "docs" / "V2_B2_NATIVE_GATE_SUITE_READINESS_RECEIPT_V1.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
