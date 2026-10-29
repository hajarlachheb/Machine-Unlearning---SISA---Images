"""Compare saved multilabel predictions on identical original test images."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.train_multilabel import encode_targets, multilabel_metrics, sha256


def align_test_rows(old_rows, new_rows):
    old = [r for r in old_rows if r["split"] == "test"]
    new = [r for r in new_rows if r["split"] == "test"]
    if len({r["image"] for r in old}) != len(old) or len({r["image"] for r in new}) != len(new):
        raise ValueError("Duplicate test image IDs")
    lookup = {r["image"]: i for i, r in enumerate(new)}
    matched = []
    for r in old:
        if r["image"] not in lookup:
            raise ValueError("Original test image is missing from expanded test split")
        i = lookup[r["image"]]
        if any(r[key] != new[i][key] for key in ("patient", "labels", "study")):
            raise ValueError("Original test image identity or annotation changed")
        matched.append(i)
    matched_set = set(matched)
    additional = [i for i in range(len(new)) if i not in matched_set]
    old_patients = {r["patient"] for r in old}
    if any(new[i]["patient"] in old_patients for i in additional):
        raise ValueError("Additional test images include an original test patient")
    if not old or not additional:
        raise ValueError("Both original and additional test groups are required")
    return old, new, matched, additional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--old-data", type=Path, required=True)
    parser.add_argument("--new-data", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    old_result = json.loads((args.old_run / "results.json").read_text())
    new_result = json.loads((args.new_run / "results.json").read_text())
    labels = old_result["config"]["labels"]
    if labels != new_result["config"]["labels"]:
        raise ValueError("Ordered label vocabularies differ")
    for data, result in ((args.old_data, old_result), (args.new_data, new_result)):
        if sha256(data / "manifest.json") != result["cache_sha256"]["manifest.json"]:
            raise ValueError("Manifest changed since training")
    old_rows = json.loads((args.old_data / "manifest.json").read_text())
    new_rows = json.loads((args.new_data / "manifest.json").read_text())
    old, new, matched, additional = align_test_rows(old_rows, new_rows)
    old_probs = np.load(args.old_run / "test_probabilities.npy")
    new_probs = np.load(args.new_run / "test_probabilities.npy")
    if old_probs.shape != (len(old), len(labels)) or new_probs.shape != (len(new), len(labels)):
        raise ValueError("Saved probability array shape differs from manifest")
    old_initial = torch.load(args.old_run / "initial.pt", map_location="cpu", weights_only=False)["model"]
    new_initial = torch.load(args.new_run / "initial.pt", map_location="cpu", weights_only=False)["model"]
    initial_equal = old_initial.keys() == new_initial.keys() and all(torch.equal(v, new_initial[k]) for k, v in old_initial.items())
    if not initial_equal:
        raise ValueError("Initial model parameters/buffers differ")
    truth = encode_targets(new, labels)
    groups = {"original_model_original_test": (old, encode_targets(old, labels), old_probs),
              "expanded_model_original_test": (old, truth[matched], new_probs[matched]),
              "expanded_model_additional_test": ([new[i] for i in additional], truth[additional], new_probs[additional]),
              "expanded_model_full_test": (new, truth, new_probs)}
    result = {"initial_model_state_identical": initial_equal, "labels": labels, "groups": {},
              "old_training_seconds": old_result["training_seconds"], "new_training_seconds": new_result["training_seconds"]}
    lines = ["# Expanded training: comparison on the same test images", "",
             "Both runs use identical initial model parameters/buffers and the same ordered 19 findings. "
             "Predictions are aligned by image ID after checking patient, study and annotation identity. "
             "These metrics are computed after validation-only checkpoint selection.", "",
             "| Evaluation | Patients | Images | Macro AUROC | Macro AP | Micro F1 at 0.5 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, (records, targets, probs) in groups.items():
        m = multilabel_metrics(targets, probs, labels, threshold=0.5)
        result["groups"][name] = {"patients": len({r["patient"] for r in records}), "metrics": m}
        lines.append(f"| {name.replace('_', ' ')} | {result['groups'][name]['patients']} | {len(records)} | "
                     f"{m['macro_auroc']:.3f} | {m['macro_average_precision']:.3f} | {m['micro_f1']:.3f} |")
    lines += ["", "The first two rows compare the same original test images. The other rows describe different test populations "
              "and should not be interpreted as a controlled before/after improvement. Training and validation cohorts expanded together; "
              "six epochs on more training images also mean more optimizer updates, and training-derived class weights changed. "
              "This is an exploratory, single-seed comparison without uncertainty intervals, not a causal estimate of the effect of data quantity alone. "
              "Original test patients were inspected in earlier experiments. No test metric was used to select the checkpoint or tune thresholds.", ""]
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False))
    args.output_stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
