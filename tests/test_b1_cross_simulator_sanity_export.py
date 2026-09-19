from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools_run_v2_b1_cross_simulator_sanity_export import build_export


class B1CrossSimulatorExportTest(unittest.TestCase):
    def test_fixed_export_retains_only_declared_fields(self) -> None:
        receipt = build_export()
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["suite_count"], 2)
        self.assertEqual(receipt["profile"]["calibration_status"], "literature_parameterised")
        self.assertTrue(all(value is False for value in receipt["retention"].values()))
        self.assertTrue(all(value is False for value in receipt["scope"].values()))
        rendered = json.dumps(receipt["suite"], sort_keys=True).lower()
        for forbidden in ("statevector", "probabilities", "amplitude", "phase_rad", "duration_s", "fidelity", "reward"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(
            [row["circuit_id"] for row in receipt["suite"]],
            ["native_single_rx_pi_over_2", "native_two_qubit_cz_phase"],
        )
        self.assertEqual([len(row["operation_categories"]) for row in receipt["suite"]], [1, 4])

    def test_readiness_receipt_passes(self) -> None:
        subprocess.run([sys.executable, "tools_validate_v2_b1_cross_simulator_sanity.py"], cwd=ROOT, check=True)
        receipt = json.loads((ROOT / "docs" / "V2_B1_CROSS_SIMULATOR_SANITY_READINESS_RECEIPT_V1.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
