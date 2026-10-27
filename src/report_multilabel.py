"""Export aggregate multilabel metrics and learning curves without patient data."""
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def value(number):
    return "N/A" if number is None else f"{number:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    config, support, metrics = result["config"], result["support"], result["metrics"]
    labels = config["labels"]
    history = result["history"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), gridspec_kw={"height_ratios": [1, 3]}, layout="constrained")
    epochs = [h["epoch"] for h in history]
    axes[0, 0].plot(epochs, [h["training_loss"] for h in history], "o-", color="#16324f")
    axes[0, 0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted binary cross-entropy")
    for metric, label, color in [("macro_auroc", "Macro AUROC", "#007f82"),
                                 ("macro_average_precision", "Macro AP", "#b06518")]:
        axes[0, 1].plot(epochs, [h["validation"][metric] for h in history], "o-", color=color, label=label)
    axes[0, 1].axvline(result["best_epoch"], color="#888888", linestyle=":", label="Selected epoch")
    axes[0, 1].set(title="Validation: checkpoint selection", xlabel="Epoch", ylim=(0, 1))
    axes[0, 1].legend(fontsize=9)
    tick_labels = [textwrap.fill(label, 26) + f" (n={metrics['test']['per_label'][label]['positive_images']})" for label in labels]
    for ax, key, title, color in [(axes[1, 0], "auroc", "Test AUROC", "#007f82"),
                                   (axes[1, 1], "average_precision", "Test average precision", "#16324f")]:
        values = [metrics["test"]["per_label"][label][key] for label in labels]
        ax.barh(np.arange(len(labels)), [v if v is not None else 0 for v in values], color=color)
        ax.set(yticks=np.arange(len(labels)), yticklabels=tick_labels, xlim=(0, 1), title=title)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
        for i, v in enumerate(values):
            ax.text(min((v or 0) + 0.015, 0.93), i, value(v), va="center", fontsize=8)
        if key == "auroc":
            ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        else:
            ax.scatter([metrics["test"]["per_label"][label]["prevalence"] for label in labels],
                       np.arange(len(labels)), color="#d57821", s=25, label="Positive prevalence", zorder=3)
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("PadChest multilabel ResNet-18: exploratory pilot\nn = positive test images; no uncertainty intervals", fontsize=13)
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Multilabel ResNet-18 on PadChest", "",
             f"Full ImageNet-initialized ResNet-18 fine-tuning with **{len(labels)} independent finding outputs**. "
             "Multiple findings can be positive in the same image. The old binary target is not used.", "",
             "| Setting | Value |", "|---|---|",
             f"| Input / budget | {config['image_size']}×{config['image_size']}, batch {config['batch_size']}, {config['epochs']} epochs |",
             f"| Loss / optimizer | Weighted BCEWithLogitsLoss / Adam {config['learning_rate']} |",
             f"| Selected checkpoint | Epoch {result['best_epoch']}, highest validation macro AUROC |",
             f"| Training time | {result['training_seconds']:.1f} seconds ({result['training_seconds']/60:.2f} minutes) |",
             f"| Fit including validation/checkpoints | {result['fit_wall_seconds']:.1f} seconds; cached preparation excluded |",
             f"| Backbone updated | {result['backbone_updated']} |", "",
             "| Split | Patients | Images | Images with ≥2 selected findings | Macro AUROC | Macro AP | Micro F1 at 0.5 | Valid ranking labels |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "validation", "test"):
        s, m = support[split], metrics[split]
        lines.append(f"| {split} | {s['patients']} | {s['images']} | {s['images_with_multiple_selected_labels']} | "
                     f"{value(m['macro_auroc'])} | {value(m['macro_average_precision'])} | {value(m['micro_f1'])} | {m['valid_ranking_labels']}/{len(labels)} |")
    lines += ["", "## Per-finding support and test performance", "",
              ("Vocabulary fixed to the earlier pilot's ordered labels; no new findings added as the cohort expands. "
               if config.get("fixed_labels") else
               "Vocabulary chosen using training patients only: ≥10 positive and ≥10 negative patients, excluding `normal` and `unchanged`. ") +
              "These are raw recorded finding terms; this is not the entire PadChest label catalogue.", "",
              "| Finding | Train + patients | Train + images | Val + images | Test + images | Test AUROC | Test AP | Test recall at 0.5 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        s = support["train"]["labels"][label]
        m = metrics["test"]["per_label"][label]
        lines.append(f"| {label} | {s['positive_patients']} | {s['positive_images']} | "
                     f"{support['validation']['labels'][label]['positive_images']} | {m['positive_images']} | "
                     f"{value(m['auroc'])} | {value(m['average_precision'])} | {value(m['recall'])} |")
    ref = result["training_prevalence_baseline"]["test"]
    lines += ["", f"![Learning curves and per-finding test metrics]({stem.name}.png)", "",
              f"The constant training-prevalence predictor gives test macro AUROC **{value(ref['macro_auroc'])}** "
              f"and macro AP **{value(ref['macro_average_precision'])}**. AP is compared with positive prevalence, not 0.5. "
              "Per-label confusion matrices, macro/micro F1, micro ranking metrics, Hamming loss and exact-match accuracy are in the JSON.", "",
              "Training ranking scores are almost perfect while held-out scores are much lower, indicating substantial overfitting. "
              "The fixed-threshold F1 and per-finding recalls also show that useful ranking does not guarantee reliable positive predictions. "
              "The current result establishes a working multilabel baseline, not a clinically validated classifier.", "",
              "**Interpretation limits:** single seed; small archive-based convenience sample; sparse positive support; "
              "previously inspected exploratory test cohort; no patient-bootstrap intervals. Missing annotations are treated as zeros, "
              "not confirmed clinical absence. Null ranking scores indicate only one observed class and are excluded from macro ranking averages. "
              "Scores are image-level. No unlearning has been performed in this run.", "",
              "[Protocol, loss formula and reproduction commands](../docs/multilabel-training-protocol.md). "
              "Checkpoints and individual predictions remain local. The binary experiment is retained as an earlier pipeline check.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Created aggregate report: {stem}")


if __name__ == "__main__":
    main()
