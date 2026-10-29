"""Multilabel semantics, train-only selection and held-out metric edge cases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from src.train_multilabel import (encode_targets, select_labels, positive_weights,
                                 multilabel_metrics, train_epoch, load_cache)


class MultilabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_two_findings_are_encoded_simultaneously_and_empty_selected_set_is_valid(self):
        rows = [{"labels": ["effusion", "cardiomegaly", "effusion"]}, {"labels": ["normal"]}]
        result = encode_targets(rows, ["cardiomegaly", "effusion"])
        np.testing.assert_array_equal(result, [[1, 1], [0, 0]])
        self.assertEqual(result.dtype, np.float32)

    def test_label_selection_ignores_holdouts_and_counts_patients_not_images(self):
        config = {"excluded_targets": ["normal"], "min_positive_train_patients": 2,
                  "min_negative_train_patients": 1}
        rows = [{"patient": "a", "split": "train", "labels": ["x", "y", "rare", "normal"]},
                {"patient": "a", "split": "train", "labels": ["rare"]},
                {"patient": "b", "split": "train", "labels": ["x", "y"]},
                {"patient": "c", "split": "train", "labels": ["normal"]}]
        labels, audit = select_labels(rows, config)
        altered = rows + [{"patient": str(i), "split": "test", "labels": ["rare", "heldout_only"]}
                          for i in range(100)]
        self.assertEqual(labels, ["x", "y"])
        self.assertEqual(select_labels(altered, config), (labels, audit))

    def test_imbalance_weights_use_supplied_training_matrix(self):
        targets = np.array([[1, 1], [0, 1], [0, 0], [0, 0]], dtype=np.float32)
        torch.testing.assert_close(positive_weights(targets), torch.tensor([3.0, 1.0]))
        with self.assertRaises(ValueError):
            positive_weights(np.zeros((3, 2), dtype=np.float32))

    def test_fixed_vocabulary_preserves_order_and_does_not_add_supported_labels(self):
        config = {"excluded_targets": [], "min_positive_train_patients": 1,
                  "min_negative_train_patients": 1, "fixed_labels": ["y", "x"]}
        rows = [{"patient": "a", "split": "train", "labels": ["x", "y", "new"]},
                {"patient": "b", "split": "train", "labels": ["normal"]}]
        self.assertEqual(select_labels(rows, config)[0], ["y", "x"])
        config["fixed_labels"] = ["y", "missing"]
        with self.assertRaisesRegex(ValueError, "fixed label"):
            select_labels(rows, config)

    def test_independent_bce_can_raise_both_positive_outputs(self):
        logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.ones(1, 2))
        loss.backward()
        self.assertTrue((logits.grad < 0).all())
        prob = torch.tensor([[3.0, 4.0]]).sigmoid()
        self.assertTrue((prob > 0.5).all())
        self.assertGreater(float(prob.sum()), 1.0)

    def test_metrics_skip_unidentifiable_ranking_labels_without_nan(self):
        targets = [[0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1]]
        probs = [[0.1, 0.2, 0.8], [0.9, 0.2, 0.8], [0.2, 0.2, 0.8], [0.8, 0.2, 0.8]]
        result = multilabel_metrics(targets, probs, ["mixed", "none", "all"])
        self.assertEqual(result["valid_ranking_labels"], 1)
        self.assertEqual(result["macro_auroc"], 1.0)
        self.assertEqual(result["macro_average_precision"], 1.0)
        self.assertIsNone(result["per_label"]["none"]["auroc"])
        self.assertIsNone(result["per_label"]["all"]["average_precision"])
        self.assertEqual(result["exact_match_accuracy"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_all_negative_predictions_do_not_receive_good_f1(self):
        result = multilabel_metrics([[1, 1], [0, 0]], [[0.1, 0.1], [0.1, 0.1]], ["a", "b"])
        self.assertEqual(result["micro_f1"], 0.0)
        self.assertEqual(result["macro_auroc"], 0.5)
        self.assertEqual(result["exact_match_accuracy"], 0.5)

    def test_real_resnet_multilabel_step_updates_backbone_and_restores_outputs(self):
        torch.manual_seed(29)
        model = resnet18(weights=None, num_classes=3)
        original = model.conv1.weight.detach().clone()
        x = torch.rand(2, 1, 64, 64)
        y = torch.tensor([[1., 1., 0.], [0., 1., 1.]])
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        loss, elapsed = train_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=0.0001),
                                    torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2., 1., 3.])), 1)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(elapsed, 0)
        self.assertFalse(torch.equal(original, model.conv1.weight))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        restored = resnet18(weights=None, num_classes=3)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        model.eval()
        restored.eval()
        rgb = x.repeat(1, 3, 1, 1)
        with torch.inference_mode():
            a, b = model(rgb).sigmoid(), restored(rgb).sigmoid()
        self.assertEqual(tuple(a.shape), (2, 3))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_cache_rejects_inconsistent_patient_split(self):
        config = {"seed": 1, "image_size": 4, "allowed_projections": ["PA"],
                  "excluded_labels": [], "split_percentages": [70, 15, 15], "shards": 1, "slices": 1}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps(config))
            (path / "manifest.json").write_text(json.dumps([
                {"image": "1", "patient": "same", "split": "train", "labels": ["x"]},
                {"image": "2", "patient": "same", "split": "test", "labels": ["x"]}]))
            with self.assertRaisesRegex(ValueError, "patient split"):
                load_cache(path, config)

    def test_cache_checks_fresh_rule_for_added_patients_and_stable_rule_for_others(self):
        from src.pilot_data import assignment, fresh_assignment
        rule = {"seed": 7, "namespace": "fresh-split", "fresh_test_draw_below": 32, "train_draw_below": 88,
                "applies_to": ["new"]}
        config = {"seed": 1, "image_size": 4, "allowed_projections": ["PA"], "excluded_labels": [],
                  "split_percentages": [70, 15, 15], "shards": 1, "slices": 1}
        old = next(f"o{i}" for i in range(100) if assignment(f"o{i}", config)["split"] == "train"
                   and fresh_assignment(f"o{i}", rule) != "train")  # Rules disagree, so a wrong rule is caught.
        new = next(f"n{i}" for i in range(100) if fresh_assignment(f"n{i}", rule) == "fresh_test")
        good = [{"image": "1", "patient": old, "split": "train", "origin": "old", "labels": ["x"]},
                {"image": "2", "patient": new, "split": "fresh_test", "origin": "new", "labels": ["x"]}]
        for rows, error in [(good, None),
                            ([dict(good[0]), dict(good[1], split="train")], "patient split"),
                            ([dict(good[0], origin="new"), good[1]], "patient split")]:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                (path / "config.json").write_text(json.dumps(dict(config, fresh_split=rule)))
                (path / "manifest.json").write_text(json.dumps(rows))
                np.save(path / "images.npy", np.zeros((len(rows), 1, 4, 4), dtype=np.float32))
                if error is None:
                    self.assertEqual([r["split"] for r in load_cache(path, config)[0]], ["train", "fresh_test"])
                else:
                    with self.assertRaisesRegex(ValueError, error):
                        load_cache(path, config)


if __name__ == "__main__":
    unittest.main()
