"""Checks for the conventional training path; no network or patient data needed."""
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from src.train_baseline import imagenet_input, metrics, train_epoch


class NormalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_grayscale_channels_receive_correct_imagenet_normalization(self):
        gray = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 2)
        rgb = imagenet_input(gray)
        self.assertEqual(tuple(rgb.shape), (1, 3, 1, 2))
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(rgb[0, :, 0, 0], -mean / std)
        torch.testing.assert_close(rgb[0, :, 0, 1], (1 - mean) / std)
        with self.assertRaises(ValueError):
            imagenet_input(torch.zeros(2, 3, 10, 10))

    def test_metrics_reveal_majority_only_classifier(self):
        result = metrics([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.1])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["sensitivity_normal_label"], 0.0)
        self.assertEqual(result["confusion_matrix_true_rows_predicted_columns"], [[3, 0], [1, 0]])

    def test_training_updates_backbone_and_head_and_state_restores_predictions(self):
        torch.manual_seed(13)
        model = resnet18(weights=None, num_classes=2)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        conv = model.conv1.weight.detach().clone()
        head = model.fc.weight.detach().clone()
        images = torch.rand(2, 1, 64, 64)
        loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)
        loss, seconds = train_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=0.0001), 1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(seconds, 0)
        self.assertFalse(torch.equal(conv, model.conv1.weight))
        self.assertFalse(torch.equal(head, model.fc.weight))
        self.assertEqual(int(model.bn1.num_batches_tracked), 1)
        restored = resnet18(weights=None, num_classes=2)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(imagenet_input(images)), restored(imagenet_input(images)), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
