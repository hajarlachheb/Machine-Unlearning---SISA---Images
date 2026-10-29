"""Conventional full fine-tuning of ImageNet ResNet-18 on local PadChest images.

One model, no SISA stages. Select using validation AUROC, evaluate test once
after selection. All patient-level outputs and checkpoints stay under outputs/.
"""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from src.pilot_data import prepare


def imagenet_input(gray):
    """Repeat normalized grayscale into RGB, then apply ImageNet channel stats."""
    if gray.ndim != 4 or gray.shape[1] != 1:
        raise ValueError("Expected N,1,H,W grayscale inputs")
    mean = gray.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = gray.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (gray.repeat(1, 3, 1, 1) - mean) / std


class Images(Dataset):
    def __init__(self, images, rows, split):
        self.images = images
        self.rows = rows
        self.indices = [i for i, row in enumerate(rows) if row["split"] == split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return torch.from_numpy(self.images[i].copy()), self.rows[i]["target"]


def build_model(weights, expected_hash, num_classes=2):
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError("Pretrained checkpoint SHA256 mismatch")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"images": len(labels), "auroc": float(roc_auc_score(labels, probabilities)),
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "sensitivity_normal_label": float(tp / (tp + fn)),
            "specificity_normal_label": float(tn / (tn + fp)),
            "confusion_matrix_true_rows_predicted_columns": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": 0.5}


def evaluate(model, loader):
    model.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for gray, target in loader:
            prob = model(imagenet_input(gray)).softmax(dim=1)[:, 1]
            labels.extend(target.tolist())
            probabilities.extend(prob.tolist())
    return metrics(labels, probabilities), probabilities


def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total, count = 0.0, 0
    started = time.perf_counter()
    last_message = started
    for step, (gray, target) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(imagenet_input(gray)), target)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
        count += len(target)
        if time.perf_counter() - last_message >= 25:
            print(f"Epoch {epoch}: batch {step}/{len(loader)}, loss {total / count:.4f}", flush=True)
            last_message = time.perf_counter()
    return total / count, time.perf_counter() - started


def checkpoint(model, optimizer, epoch, generator, config):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(), "loader_rng": generator.get_state(), "config": config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/normal_resnet18.json"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("outputs/model-cache/resnet18-f37072fd.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, help="Reuse only a cache with this exact configuration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["architecture"] != "resnet18" or config["initialization"] != "ImageNet1K_V1":
        raise ValueError("Only ImageNet ResNet-18 is implemented")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    preparation_started = time.perf_counter()
    if args.prepared_data:
        data_dir = args.prepared_data
        if json.loads((data_dir / "config.json").read_text()) != config:
            raise ValueError("Prepared-data configuration differs")
        rows = json.loads((data_dir / "manifest.json").read_text())
        summary = json.loads((data_dir / "summary.json").read_text())
        if hashlib.sha256(args.metadata.read_bytes()).hexdigest() != summary["metadata_sha256"]:
            raise ValueError("Metadata differs from prepared cache")
        if args.archive.stat().st_size != summary["archive_bytes"]:
            raise ValueError("Archive size differs from prepared cache")
    else:
        data_dir = args.output / "data"
        rows, _, summary = prepare(args.metadata, args.archive, data_dir, config)
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    if images.shape != (len(rows), 1, config["image_size"], config["image_size"]):
        raise ValueError("Prepared image shape differs")
    if any(summary["patient_overlap"].values()):
        raise ValueError("Patient leakage across splits")
    preparation_seconds = time.perf_counter() - preparation_started
    generator = torch.Generator().manual_seed(config["seed"])
    train = DataLoader(Images(images, rows, "train"), batch_size=config["batch_size"],
                       shuffle=True, generator=generator, num_workers=0)
    loaders = {s: DataLoader(Images(images, rows, s), batch_size=config["batch_size"], num_workers=0)
               for s in ("train", "validation", "test")}
    model = build_model(args.weights, config["weights_sha256"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    # Save the exact common initialization for future retained-data retraining.
    torch.save(checkpoint(model, optimizer, 0, generator, config), args.output / "initial.pt")
    history, best_epoch, best_score = [], None, float("-inf")
    initial_conv = model.conv1.weight.detach().clone()
    fit_started = time.perf_counter()
    print(f"Fine-tuning all {sum(p.numel() for p in model.parameters()):,} parameters on CPU; "
          f"{len(train.dataset)} training images, {config['epochs']} epochs", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        loss, seconds = train_epoch(model, train, optimizer, epoch)
        validation, _ = evaluate(model, loaders["validation"])
        history.append({"epoch": epoch, "train_loss": loss, "training_seconds": seconds,
                        "validation": validation})
        if validation["auroc"] > best_score:  # Ties retain the earliest epoch.
            best_score, best_epoch = validation["auroc"], epoch
            torch.save(checkpoint(model, optimizer, epoch, generator, config), args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}/{config['epochs']}: loss={loss:.4f}, val AUROC={validation['auroc']:.4f}, "
              f"val balanced accuracy={validation['balanced_accuracy']:.4f}, train={seconds:.1f}s", flush=True)
    torch.save(checkpoint(model, optimizer, config["epochs"], generator, config), args.output / "last.pt")
    fit_seconds = time.perf_counter() - fit_started
    # Own trusted checkpoint includes Python/NumPy RNG state. Never load untrusted files this way.
    chosen = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(chosen["model"])
    final = {}
    for split, loader in loaders.items():
        final[split], _ = evaluate(model, loader)
    result = {"experiment": "conventional_resnet18_imagenet_full_finetune", "config": config,
              "data": summary, "best_epoch": best_epoch, "selection": "highest validation AUROC; earliest tie",
              "preparation_seconds": preparation_seconds, "fit_wall_seconds": fit_seconds,
              "training_seconds": sum(h["training_seconds"] for h in history),
              "backbone_updated": not torch.equal(initial_conv, model.conv1.weight),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "history": history, "metrics": final,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "torchvision": torchvision.__version__, "numpy": np.__version__,
                              "sklearn": sklearn.__version__, "device": "cpu", "platform": platform.platform(),
                              "threads": torch.get_num_threads(), "mkldnn": torch.backends.mkldnn.enabled},
              "limitations": ["Single-seed convenience sample, exploratory reused test split",
                              "Predicts normal annotation presence, not confirmed disease absence",
                              "No SISA or unlearning in this run; no exactness claim",
                              "Different architecture/resolution/initialization from the TinyCNN pilot"]}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "metrics": final, "training_seconds": result["training_seconds"],
                      "backbone_updated": result["backbone_updated"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
