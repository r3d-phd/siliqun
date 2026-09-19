from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class D1SiliconEligibilityTest(unittest.TestCase):
    def test_d1_validator_preserves_refusal(self) -> None:
        subprocess.run(
            [sys.executable, "tools_validate_d1_silicon_eligibility.py"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
        )
        receipt = json.loads((ROOT / "docs" / "V2_D1_SILICON_ELIGIBILITY_VALIDATION_RECEIPT_V1.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
