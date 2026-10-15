"""Synthetic patient-completeness checks: archive membership alone is insufficient."""
import unittest

from src.inspect_archive import complete_patients


class CompletenessTests(unittest.TestCase):
    def test_missing_scan_excludes_entire_patient_and_preserves_string_ids(self):
        rows = [{"PatientID": "001", "ImageID": "a"},
                {"PatientID": "001", "ImageID": "b"},
                {"PatientID": "1", "ImageID": "c"}]
        self.assertEqual(set(complete_patients(rows, {"a", "c"})), {"1"})
        self.assertEqual(set(complete_patients(rows, {"a", "b", "c"})), {"001", "1"})

    def test_missing_identifier_fails_completeness_check(self):
        with self.assertRaises(ValueError):
            complete_patients([{"PatientID": "nan", "ImageID": "a"}], {"a"})
