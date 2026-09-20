from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class B3AlgorithmRepresentationAuditTest(unittest.TestCase):
    def test_source_audit_passes_after_export(self) -> None:
        subprocess.run([sys.executable, "tools_audit_v2_b3_algorithm_representation_export.py"], cwd=ROOT, check=True)
        receipt = json.loads((ROOT / "docs" / "V2_B3_ALGORITHM_REPRESENTATION_SOURCE_AUDIT_RECEIPT_V1.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
