"""Patient-disjoint, conventional ResNet-18 multilabel training on PadChest.

Reuse the audited 224-pixel cache but ignore its binary target entirely. Derive
vocabulary and imbalance weights from TRAIN ONLY; hold all patients fixed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.pilot_data import assignment
from src.train_baseline import build_model, checkpoint, imagenet_input


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_labels(rows, config):
    """Deterministic vocabulary using patient-level TRAIN support only."""
    train = [r for r in rows if r["split"] == "train"]
    patients = {r["patient"] for r in train}
    candidates = sorted({label for r in train for label in r["labels"]})
    fixed = config.get("fixed_labels")
    if fixed is not None and (len(fixed) != len(set(fixed)) or len(fixed) < 2):
        raise ValueError("Fixed vocabulary must have at least two unique labels")
    audit, labels = [], []
    for label in candidates:
        positive = len({r["patient"] for r in train if label in r["labels"]})
        supported = (label not in config["excluded_targets"] and
                    positive >= config["min_positive_train_patients"] and
                    len(patients) - positive >= config["min_negative_train_patients"])
        selected = supported and (fixed is None or label in fixed)
        audit.append({"label": label, "train_positive_patients": positive,
                      "train_positive_images": sum(label in r["labels"] for r in train),
                      "selected": selected})
        if selected:
            labels.append(label)
    if fixed is not None:
        if set(labels) != set(fixed):
            raise ValueError("A fixed label is absent, excluded or below the training support threshold")
        labels = list(fixed)  # Preserve the original head's semantic order.
    if len(labels) < 2:
        raise ValueError("Fewer than two supported training labels; expand the cohort")
    return labels, audit


def encode_targets(rows, labels):
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Label vocabulary must be nonempty and unique")
    # Exact supplied terms, no invented hierarchy or label synonym mapping.
    return np.asarray([[float(label in row["labels"]) for label in labels] for row in rows], dtype=np.float32)


def positive_weights(train_targets):
    positives = train_targets.sum(axis=0)
    if np.any(positives <= 0) or np.any(positives >= len(train_targets)):
        raise ValueError("Each training label needs positive and negative images")
    return torch.as_tensor((len(train_targets) - positives) / positives, dtype=torch.float32)


def summarize_support(rows, targets, labels):
    result = {}
    for split in ("train", "validation", "test"):
        ids = [i for i, r in enumerate(rows) if r["split"] == split]
        part = targets[ids]
        result[split] = {"images": len(ids), "patients": len({rows[i]["patient"] for i in ids}),
                         "images_with_multiple_selected_labels": int((part.sum(axis=1) > 1).sum()),
                         "images_with_no_selected_labels": int((part.sum(axis=1) == 0).sum()),
                         "annotation_method_images": dict(Counter(rows[i]["method_label"] for i in ids)),
                         "labels": {label: {"positive_images": int(part[:, j].sum()),
                            "positive_patients": len({rows[i]["patient"] for i in ids if targets[i, j] == 1})}
                            for j, label in enumerate(labels)}}
    return result


def multilabel_metrics(targets, probabilities, labels, threshold=0.5):
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError("Targets/probabilities must have matching N x labels shapes")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid targets or probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between zero and one")
    predicted = probabilities >= threshold
    per_label, aucs, aps = {}, [], []
    for j, label in enumerate(labels):
        truth, pred = targets[:, j].astype(bool), predicted[:, j]
        tp, tn = int((truth & pred).sum()), int((~truth & ~pred).sum())
        fp, fn = int((~truth & pred).sum()), int((truth & ~pred).sum())
        valid = truth.any() and (~truth).any()
        auc = float(roc_auc_score(truth, probabilities[:, j])) if valid else None
        ap = float(average_precision_score(truth, probabilities[:, j])) if valid else None
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        per_label[label] = {"positive_images": int(truth.sum()), "negative_images": int((~truth).sum()),
            "prevalence": float(truth.mean()), "auroc": auc, "average_precision": ap,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if valid else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "confusion_matrix_true_rows_predicted_columns": [[tn, fp], [fn, tp]]}
        if valid:
            aucs.append(auc)
            aps.append(ap)
    flat_valid = len(np.unique(targets)) == 2
    return {"images": len(targets), "threshold": threshold, "per_label": per_label,
            "valid_ranking_labels": len(aucs), "total_labels": len(labels),
            "macro_auroc": float(np.mean(aucs)) if aucs else None,
            "macro_average_precision": float(np.mean(aps)) if aps else None,
            "micro_auroc": float(roc_auc_score(targets.ravel(), probabilities.ravel())) if flat_valid else None,
            "micro_average_precision": float(average_precision_score(targets.ravel(), probabilities.ravel())) if flat_valid else None,
            "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(targets, predicted, average="micro", zero_division=0)),
            "hamming_loss": float(np.mean(predicted != targets)),
            "exact_match_accuracy": float(np.mean(np.all(predicted == targets, axis=1)))}


class MultilabelImages(Dataset):
    def __init__(self, images, targets, rows, split):
        self.images, self.targets = images, targets
        self.indices = [i for i, row in enumerate(rows) if row["split"] == split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return torch.from_numpy(self.images[i].copy()), torch.from_numpy(self.targets[i].copy())


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    loss_sum, count = 0.0, 0
    started = last_message = time.perf_counter()
    for step, (gray, target) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        # Feed logits directly to BCEWithLogitsLoss (no sigmoid before the loss).
        loss = criterion(model(imagenet_input(gray)), target)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(target)
        count += len(target)
        if time.perf_counter() - last_message >= 25:
            print(f"Epoch {epoch}: batch {step}/{len(loader)}, weighted BCE {loss_sum/count:.4f}", flush=True)
            last_message = time.perf_counter()
    return loss_sum / count, time.perf_counter() - started


def evaluate(model, loader, labels, threshold):
    model.eval()
    truth, probs = [], []
    with torch.inference_mode():
        for gray, target in loader:
            # Independent probabilities: several findings may be positive together.
            probs.append(model(imagenet_input(gray)).sigmoid().numpy())
            truth.append(target.numpy())
    truth, probs = np.concatenate(truth), np.concatenate(probs)
    return multilabel_metrics(truth, probs, labels, threshold), probs


def load_cache(data_dir, config):
    source_config = json.loads((data_dir / "config.json").read_text())
    for key in ("seed", "image_size", "allowed_projections", "excluded_labels", "split_percentages"):
        if source_config[key] != config[key]:
            raise ValueError(f"Cache preparation setting differs: {key}")
    rows = json.loads((data_dir / "manifest.json").read_text())
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Duplicate cached images")
    splits = {}
    for r in rows:
        if r["split"] != assignment(r["patient"], source_config)["split"]:
            raise ValueError("Cached patient split does not match stable assignment")
        if not r["labels"] or not isinstance(r["labels"], list):
            raise ValueError("Missing label annotations")
        splits.setdefault(r["patient"], set()).add(r["split"])
    if any(len(s) != 1 for s in splits.values()):
        raise ValueError("Patient overlap")
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    if images.shape != (len(rows), 1, config["image_size"], config["image_size"]):
        raise ValueError("Image shape does not match cache metadata")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Invalid cached pixel values")
    return rows, images, {name: sha256(data_dir / name) for name in ("manifest.json", "images.npy", "config.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multilabel_resnet18.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/normal-resnet18-20261021/data"))
    parser.add_argument("--weights", type=Path, default=Path("outputs/model-cache/resnet18-f37072fd.pth"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config["architecture"] != "resnet18" or config["initialization"] != "ImageNet1K_V1" or
            config["positive_weight"] != "training_negative_images_divided_by_positive_images" or
            config["selection"] != "validation_macro_auroc"):
        raise ValueError("Unsupported model, initialization, weighting or selection policy")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    rows, images, provenance = load_cache(args.data_dir, config)
    labels, audit = select_labels(rows, config)
    targets = encode_targets(rows, labels)
    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    weights = positive_weights(targets[train_indices])
    config.update({"labels": labels, "pos_weight": weights.tolist()})
    save_json(args.output / "config.json", config)
    save_json(args.output / "label_selection.json", audit)
    # Vocabulary/weights have now been fixed, before examining held-out support.
    support = summarize_support(rows, targets, labels)
    save_json(args.output / "support.json", support)
    loaders = {s: DataLoader(MultilabelImages(images, targets, rows, s), batch_size=config["batch_size"],
                            num_workers=0) for s in ("train", "validation", "test")}
    generator = torch.Generator().manual_seed(config["seed"])
    train = DataLoader(loaders["train"].dataset, batch_size=config["batch_size"], shuffle=True,
                       generator=generator, num_workers=0)
    model = build_model(args.weights, config["weights_sha256"], num_classes=len(labels))
    initial_conv = model.conv1.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    torch.save(checkpoint(model, optimizer, 0, generator, config), args.output / "initial.pt")
    best_score, best_epoch, history = float("-inf"), None, []
    print(f"Labels ({len(labels)}): {', '.join(labels)}", flush=True)
    print(f"Full fine-tuning: {len(train.dataset)} images; {config['epochs']} epochs on CPU", flush=True)
    fit_started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        loss, seconds = train_epoch(model, train, optimizer, criterion, epoch)
        validation, _ = evaluate(model, loaders["validation"], labels, config["threshold"])
        score = validation["macro_auroc"]
        if score is None:
            raise ValueError("No validation label has both classes; cannot select a checkpoint")
        history.append({"epoch": epoch, "training_loss": loss, "training_seconds": seconds, "validation": validation})
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(checkpoint(model, optimizer, epoch, generator, config), args.output / "best.pt")
        save_json(args.output / "history.json", history)
        print(f"Epoch {epoch}: weighted BCE={loss:.4f}; val macro AUROC={score:.4f}; "
              f"val mAP={validation['macro_average_precision']:.4f}; train={seconds:.1f}s", flush=True)
    torch.save(checkpoint(model, optimizer, config["epochs"], generator, config), args.output / "last.pt")
    fit_seconds = time.perf_counter() - fit_started
    chosen = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)  # own trusted file
    model.load_state_dict(chosen["model"])
    final, baseline = {}, {}
    train_prevalence = targets[train_indices].mean(axis=0)
    for split, loader in loaders.items():
        final[split], prob = evaluate(model, loader, labels, config["threshold"])
        np.save(args.output / f"{split}_probabilities.npy", prob)  # Local only, manifest order within split.
        truth = targets[loader.dataset.indices]
        baseline[split] = multilabel_metrics(truth, np.broadcast_to(train_prevalence, truth.shape), labels, config["threshold"])
    result = {"experiment": "multilabel_resnet18_imagenet_full_finetune", "config": config,
              "label_selection": audit, "support": support, "cache_sha256": provenance,
              "best_epoch": best_epoch, "history": history, "metrics": final,
              "training_prevalence_baseline": baseline, "training_seconds": sum(h["training_seconds"] for h in history),
              "fit_wall_seconds": fit_seconds, "backbone_updated": not torch.equal(initial_conv, model.conv1.weight),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
                              "numpy": np.__version__, "device": "cpu", "threads": config["threads"]},
              "limitations": ["Single seed, small convenience sample and previously inspected exploratory test split",
                              "Absent annotation treated as zero, not confirmed clinical absence",
                              "Raw label terms without hierarchy propagation; findings subset, not all PadChest labels",
                              "Patient-based uncertainty intervals not yet estimated",
                              "Ranking metrics omit labels with only one observed class; counts reported",
                              "No unlearning performed in this experiment"]}
    save_json(args.output / "results.json", result)
    print(json.dumps({"best_epoch": best_epoch, "test_macro_auroc": final["test"]["macro_auroc"],
                     "test_mAP": final["test"]["macro_average_precision"], "training_seconds": result["training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
