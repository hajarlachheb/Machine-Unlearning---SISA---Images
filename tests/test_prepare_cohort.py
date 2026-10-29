import csv
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from src.pilot_data import assignment, preprocess_pixels
from src.prepare_cohort import Source, prepare, select_records


class ExpandedCohortTests(unittest.TestCase):
    def config(self):
        return {"seed": 1, "image_size": 16, "shards": 1, "slices": 1,
                "split_percentages": [70, 15, 15], "allowed_projections": ["PA", "AP"],
                "excluded_labels": ["exclude", "suboptimal study"], "fixed_labels": ["a", "b"]}

    def test_patient_is_complete_only_after_combining_sources(self):
        rows = [{"PatientID": "p", "ImageID": name, "Projection": "PA", "Labels": "['a']"}
                for name in ("one", "two")]
        self.assertEqual(len(select_records(rows, {"one"}, self.config())[0]), 0)
        self.assertEqual(len(select_records(rows, {"one", "two"}, self.config())[0]), 2)

    def test_duplicate_folder_basenames_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path = Path(folder)
            (path / "nested").mkdir()
            (path / "same.png").write_bytes(b"x")
            (path / "nested" / "same.png").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                Source("54", path, stack)

    def test_mixed_sources_prepare_pixels_preserve_old_cache_and_reject_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.config()
            pixels = np.arange(24, dtype=np.uint16).reshape(4, 6)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            payload = buffer.getvalue()
            rows = []
            for image, patient, directory in [("1.png", "p1", "6"), ("2.png", "p1", "54"),
                                               ("3.png", "p2", "6"), ("4.png", "p2", "99"),
                                               ("5.png", "p3", "6")]:
                rows.append({"ImageID": image, "PatientID": patient, "ImageDir": directory,
                    "Projection": "PA", "Labels": "['a', 'b']", "Rows_DICOM": "4", "Columns_DICOM": "6",
                    "PhotometricInterpretation_DICOM": "MONOCHROME2", "StudyID": "s", "StudyDate_DICOM": "20100101",
                    "PatientSex_DICOM": "F", "MethodLabel": "Physician"})
            meta = root / "meta.csv"
            with meta.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            archive = root / "6.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in ("1.png", "3.png", "5.png"):
                    bundle.writestr(name, payload)
            extracted = root / "54"
            extracted.mkdir()
            (extracted / "2.png").write_bytes(payload)
            old = root / "old"
            old.mkdir()
            (old / "config.json").write_text(json.dumps(config))
            (old / "manifest.json").write_text(json.dumps([{"patient": "p3", "image": "5.png", "labels": ["a", "b"],
                "projection": "PA", "study": "s", "date": "20100101", **assignment("p3", config)}]))
            np.save(old / "images.npy", preprocess_pixels(pixels, "MONOCHROME2", 16)[None])
            summary = prepare(meta, [("6", archive), ("54", extracted)], root / "out", config, old)
            self.assertEqual(summary["eligible_images"], 3)
            self.assertEqual(summary["eligible_patients"], 2)
            self.assertEqual(summary["previous_images_preserved_and_pixels_identical"], 1)
            self.assertTrue(all(n == 0 for n in summary["patient_overlap"].values()))
            data = np.load(root / "out" / "images.npy")
            self.assertEqual(data.shape, (3, 1, 16, 16))
            (extracted / "2.png").unlink()
            with self.assertRaisesRegex(ValueError, "1 missing"):
                prepare(meta, [("6", archive), ("54", extracted)], root / "bad", config)


if __name__ == "__main__":
    unittest.main()
