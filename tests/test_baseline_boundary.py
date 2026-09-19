from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED_TOKENS = tuple(
    "".join(parts)
    for parts in (
        ("gym", "nasium"),
        ("stable", "_baselines"),
        ("andro", "meda"),
        ("qua", "sar"),
        ("simo", "ra"),
    )
)


class BaselineBoundaryTest(unittest.TestCase):
    def test_manifest_records_orphan_source_boundary(self) -> None:
        manifest = json.loads((ROOT / "BASELINE_MANIFEST.json").read_text())
        self.assertEqual(manifest["artifact_type"], "SILIQUN_V2_STANDALONE_BASELINE_MANIFEST")
        self.assertEqual(manifest["legacy_source"]["commit"], "14a49bf957b7aeacffc1bd15ba91293d6fd31b83")
        self.assertIn("named-device calibration claims", manifest["scope"]["excluded"])

    def test_package_avoids_excluded_legacy_terms(self) -> None:
        for path in (ROOT / "src" / "siliqun").rglob("*.py"):
            content = path.read_text().lower()
            for token in BANNED_TOKENS:
                self.assertNotIn(token, content, f"{token} appears in {path}")

    def test_documentation_states_calibration_boundary(self) -> None:
        readme = (ROOT / "README.md").read_text().lower()
        boundary = (ROOT / "docs" / "LEGACY_BOUNDARY.md").read_text().lower()
        self.assertIn("named-device", readme)
        self.assertIn("not a claim", boundary)


if __name__ == "__main__":
    unittest.main()
