import unittest
from src.compare_cohorts import align_test_rows


class CohortComparisonTests(unittest.TestCase):
    def row(self, image, patient):
        return {"image": image, "patient": patient, "study": "s", "labels": ["a"], "split": "test"}

    def test_alignment_follows_image_identity_not_position(self):
        a, b, c = self.row("a", "p1"), self.row("b", "p2"), self.row("c", "p3")
        _, _, matched, additional = align_test_rows([a, b], [c, b, a])
        self.assertEqual(matched, [2, 1])
        self.assertEqual(additional, [0])

    def test_rejects_missing_or_changed_original_test_records(self):
        a, b = self.row("a", "p1"), self.row("b", "p2")
        with self.assertRaisesRegex(ValueError, "missing"):
            align_test_rows([a], [b])
        with self.assertRaisesRegex(ValueError, "changed"):
            align_test_rows([a], [{**a, "labels": ["b"]}, b])

    def test_additional_group_must_contain_new_patients(self):
        a, b = self.row("a", "p1"), self.row("b", "p1")
        with self.assertRaisesRegex(ValueError, "original test patient"):
            align_test_rows([a], [a, b])


if __name__ == "__main__":
    unittest.main()
