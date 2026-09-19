from __future__ import annotations

import unittest

from siliqun import Circuit, simos_nominal_profile
from siliqun.contracts import (
    SafePPOCalibrationPluginSpec,
    TranslationRequest,
    prohibit_runtime_operation,
    translate,
)


class DRLPreparationContractsTest(unittest.TestCase):
    def test_native_translation_yields_non_executable_receipt(self) -> None:
        request = TranslationRequest(Circuit(2).rz(0.25, 0).cz(0, 1), simos_nominal_profile())
        receipt = translate(request)
        manifest = receipt.to_manifest()
        self.assertEqual(manifest["profile_id"], "simos-nominal-literature-v1")
        self.assertEqual(manifest["calibration_status"], "literature_parameterised")
        self.assertEqual(len(manifest["pulse_schedule"]["pulses"]), 2)
        self.assertIn("not a device command", manifest["claim_ceiling"])

    def test_unsupported_gate_refuses_translation(self) -> None:
        with self.assertRaises(ValueError):
            translate(TranslationRequest(Circuit(1).h(0), simos_nominal_profile()))

    def test_wrong_baseline_identity_refuses_translation(self) -> None:
        with self.assertRaises(ValueError):
            TranslationRequest(
                Circuit(2).rz(0.25, 0),
                simos_nominal_profile(),
                baseline_root_commit="not-the-v2-root",
            ).validate()

    def test_static_plugin_spec_is_safely_bounded(self) -> None:
        spec = SafePPOCalibrationPluginSpec()
        spec.validate()
        self.assertEqual(spec.algorithm, "PPO")
        self.assertEqual(spec.maximum_qubits, 5)
        self.assertEqual(spec.command_mode, "disabled")
        self.assertEqual(spec.te_pws_status, "reserved_not_implemented")

    def test_plugin_rejects_unauthorized_mode_and_size(self) -> None:
        with self.assertRaises(ValueError):
            SafePPOCalibrationPluginSpec(mode="shadow_recommendation").validate()
        with self.assertRaises(ValueError):
            SafePPOCalibrationPluginSpec(maximum_qubits=6).validate()

    def test_runtime_operations_are_explicitly_refused(self) -> None:
        for operation in ("train_policy", "call_provider_api", "submit_job", "modify_supervisory_policy"):
            with self.subTest(operation=operation):
                with self.assertRaises(RuntimeError):
                    prohibit_runtime_operation(operation)
        with self.assertRaises(ValueError):
            prohibit_runtime_operation("unknown")


if __name__ == "__main__":
    unittest.main()
