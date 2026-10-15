"""Synthetic fixtures only; no patient data."""
import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from src.audit_metadata import audit


class AuditTests(unittest.TestCase):
    def test_grouping_duplicates_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["PatientID", "ImageID", "StudyID", "StudyDate_DICOM", "Labels"])
                writer.writerows([
                    ["001", "a", "s1", "20160101", "['test_label']"],
                    ["001", "a", "s1", "2016-01-01", "['test_label']"],
                    ["1", "a", "s2", "20160230", "not-a-list"],
                    ["1", "b", "s2", "", "[]"],
                ])
            result = audit(path)
            self.assertEqual(result["unique_patients"], 2)  # Preserve string IDs.
            self.assertEqual(result["duplicate_image_rows"], 2)
            self.assertEqual(result["images_assigned_to_multiple_patients"], 1)
            self.assertEqual(result["unique_images_per_patient"]["median"], 1.5)
            self.assertEqual(result["dates"]["unparseable"], 1)
            self.assertEqual(result["dates"]["missing"], 1)
            self.assertEqual(result["labels"]["unparseable_rows"], 1)
            self.assertNotIn("001", str(result))

    def test_compressed_minimal_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("PatientID,ImageID\nsynthetic,x\n")
            result = audit(path)
            self.assertEqual(result["rows"], 1)
            self.assertFalse(result["dates"]["column_present"])
            self.assertIsNone(result["unique_studies_per_patient"])

    def test_reject_missing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text("ImageID\nx\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit(path)

    def test_missing_markers_are_not_patients_or_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text("PatientID,ImageID,StudyDate_DICOM,Labels,MethodLabel\n"
                            "NaN,a,NaN,nan,\n"
                            "001,b,20160101,[],Physician\n", encoding="utf-8")
            result = audit(path)
            self.assertEqual(result["unique_patients"], 1)
            self.assertEqual(result["dates"]["missing"], 1)
            self.assertEqual(result["dates"]["unparseable"], 0)
            self.assertEqual(result["labels"]["missing_rows"], 1)
            self.assertEqual(result["labels"]["unparseable_rows"], 0)

    def test_reject_malformed_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text("PatientID,ImageID\np,x,extra\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit(path)


if __name__ == "__main__":
    unittest.main()
