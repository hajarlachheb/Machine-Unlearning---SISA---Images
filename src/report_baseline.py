"""Render aggregate-only baseline results; never reads images or patient manifests."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    config = result["config"]
    if config["target_label"] != "normal":
        raise ValueError("This report is specific to the normal-annotation experiment")
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    history = result["history"]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", color="#16324f")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [h["validation"]["auroc"] for h in history], "o-", color="#007f82", label="AUROC")
    axes[1].plot(epochs, [h["validation"]["balanced_accuracy"] for h in history], "s--", color="#a86917", label="Balanced accuracy")
    axes[1].axvline(result["best_epoch"], color="#888888", linestyle=":", label="Selected epoch")
    axes[1].set(title="Validation only: checkpoint selection", xlabel="Epoch", ylim=(0.4, 1.0))
    axes[1].legend(fontsize=8)
    cm = result["metrics"]["test"]["confusion_matrix_true_rows_predicted_columns"]
    axes[2].imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i][j]), ha="center", va="center",
                         color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black", fontsize=15)
    labels = ["No normal\nlabel", "Normal\nlabel"]
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
                xlabel="Predicted annotation", ylabel="Recorded annotation", title="Test: selected checkpoint, threshold 0.5")
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    # The training result schema contains configuration, aggregates and runtime facts only.
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Conventional ResNet-18: first PadChest run", "",
             "ImageNet-initialized ResNet-18, all layers fine-tuned on the downloaded archive. "
             "This is an exploratory conventional training baseline, with no unlearning.", "",
             "| Setting | Measured/configured value |", "|---|---|",
             f"| Patients (train / validation / test) | " + " / ".join(str(result["data"]["splits"][s]["patients"]) for s in ("train", "validation", "test")) + " |",
             f"| Images (train / validation / test) | " + " / ".join(str(result["data"]["splits"][s]["images"]) for s in ("train", "validation", "test")) + " |",
             f"| Input and training | {config['image_size']}×{config['image_size']}; batch {config['batch_size']}; Adam {config['learning_rate']}; {len(history)} epochs |",
             f"| Selected checkpoint | Epoch {result['best_epoch']}, highest validation AUROC |",
             f"| Training only | {result['training_seconds']:.1f} seconds ({result['training_seconds']/60:.2f} minutes) |",
             f"| Fit incl. validation/checkpoint writes | {result['fit_wall_seconds']:.1f} seconds |",
             f"| Image preparation | {result['preparation_seconds']:.1f} seconds |",
             f"| Backbone updated | {result['backbone_updated']} |", "",
             "| Split | AUROC | Accuracy | Balanced accuracy | Normal-label sensitivity | Normal-label specificity |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, metric in result["metrics"].items():
        lines.append(f"| {split} | " + " | ".join(f"{metric[k]:.4f}" for k in
                     ("auroc", "accuracy", "balanced_accuracy", "sensitivity_normal_label", "specificity_normal_label")) + " |")
    test_counts = result["data"]["splits"]["test"]
    negative = test_counts["images"] - test_counts["positive_images"]
    lines += ["", f"![Learning curves and test confusion matrix]({stem.name}.png)", "",
              "The decision threshold is fixed at 0.5. The always-negative test baseline has accuracy "
              f"{negative}/{test_counts['images']} = {negative/test_counts['images']:.2%} and balanced accuracy 50%. "
              "The target is **normal annotation presence**; "
              "its absence does not confirm disease.", "",
              "**Limits:** one seed, small archive-based convenience sample, and an exploratory test split "
              "already inspected in the earlier pilot. No confidence interval or clinical generalization claim. "
              "The earlier 64×64 TinyCNN SISA run is not a matched model comparison.", "",
              "[Training protocol and sources](../docs/normal-training-protocol.md). "
              "The initial, best and last checkpoints remain local under the run's ignored output directory. "
              "Next, train SISA with the same ResNet-18 initialization and preprocessing.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Created aggregate report, JSON and figure: {stem}")


if __name__ == "__main__":
    main()
