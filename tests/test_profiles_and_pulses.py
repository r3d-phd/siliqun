from __future__ import annotations

import unittest

from siliqun import Circuit, GateToPulseCompiler, TechnologyProfile, simos_nominal_profile


class ProfileAndPulseTest(unittest.TestCase):
    def test_nominal_profile_has_explicit_claim_ceiling(self) -> None:
        profile = simos_nominal_profile()
        self.assertEqual(profile.calibration_status, "literature_parameterised")
        self.assertFalse(profile.is_named_device)
        self.assertTrue(profile.supports("cz"))
        self.assertGreaterEqual(len(profile.citations), 1)

    def test_named_device_profile_requires_evidence_fields(self) -> None:
        with self.assertRaises(ValueError):
            TechnologyProfile(
                identifier="device-under-test",
                technology_family="silicon",
                qubit_count=2,
                connectivity=((0, 1),),
                native_gates=("cz",),
                parameters={"two_qubit_duration_s": 1e-9},
                citations=("doi:example",),
                calibration_status="named_device",
            )

    def test_native_pulse_mapping_is_profile_bound(self) -> None:
        profile = simos_nominal_profile()
        schedule = GateToPulseCompiler(profile).compile_gate(Circuit(2).cz(0, 1).gates[0])
        self.assertEqual(schedule.profile_id, profile.identifier)
        self.assertEqual(schedule.pulses[0].kind, "exchange")
        self.assertEqual(schedule.pulses[0].targets, (0, 1))
        self.assertGreater(schedule.duration_s, 0)

    def test_non_native_mapping_is_refused(self) -> None:
        profile = simos_nominal_profile()
        with self.assertRaises(ValueError):
            GateToPulseCompiler(profile).compile_gate(Circuit(1).h(0).gates[0])


if __name__ == "__main__":
    unittest.main()
