from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DRLPreparationDispositionTest(unittest.TestCase):
    def test_disposition_validator_passes(self) -> None:
        subprocess.run(
            [sys.executable, "tools_validate_drl_preparation_disposition.py"],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
        )
        result = json.loads((ROOT / "docs" / "V2_DRL_PREPARATION_DISPOSITION_VALIDATION_RECEIPT_V1.json").read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
