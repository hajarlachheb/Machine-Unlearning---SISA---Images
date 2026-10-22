# Multilabel ResNet-18 baseline

This replaces the **task definition** of the first binary experiment with prediction of multiple recorded findings per image. It is a new run from official ImageNet initialization, not a continuation of the binary-trained model. One ResNet-18 is fine-tuned on all training images; no SISA or deletion is performed here.

## Cohort and target definition

Reuse the audited PadChest archive-6 cache: 686 eligible frontal PA/AP images from 653 patients. The previous patient-disjoint split is unchanged: 459/128/99 train/validation/test images from 435/123/95 patients. Images were prepared directly from the original PNGs at 224×224 using per-image percentile normalization, aspect-preserving resizing and zero padding. The single grayscale channel is repeated three times and normalized with ImageNet channel statistics.

Vocabulary selection uses **training patients only**: keep exact recorded terms with at least 10 positive training patients and at least 10 training patients without that term. Exclude `normal` (absence-of-findings annotation) and `unchanged` (comparison term). This gives **19 finding outputs**. The minimum is a pragmatic pilot threshold, not a guarantee of statistical reliability. Labels are sorted alphabetically and saved with every checkpoint.

No validation/test frequencies or scores determine inclusion. The complete training-label audit, including excluded rare terms, is saved. All 686 images remain in the cohort, including images with no selected labels. An all-zero target means none of these **selected annotations** was recorded; it does not establish a normal image.

For each image i and selected label c, `y[i,c]=1` if the exact label occurs in its annotation list, otherwise `0`. Multiple entries can be one simultaneously. There is no conversion to one mutually exclusive disease and no parent/child label propagation. Although PadChest overall mixes physician and automated report annotations, all 686 images in this cache have `MethodLabel=Physician` in the supplied metadata. Absent labels are treated as zeros for this supervised baseline, not as confirmed clinical negatives.

## Model and training

| Setting | Value |
|---|---|
| Architecture | Torchvision ResNet-18, final layer 512 → 19 logits |
| Initialization | Official ImageNet1K_V1 weights; new randomly initialized head |
| Trainable layers | All layers, including the backbone |
| Output at evaluation | Independent sigmoid probabilities; no softmax |
| Loss | `BCEWithLogitsLoss`, with one positive-class weight per finding |
| Positive weight | Number of negative **training images** / number of positive training images for that label |
| Optimizer | Adam, learning rate 0.0001, weight decay 0.0001 |
| Budget | Six epochs, batch size 16, no augmentation or scheduler |
| Seed/hardware | 20261021, deterministic CPU execution, four threads |
| Checkpoint selection | Highest validation macro AUROC; earliest epoch on ties |
| Decision threshold | Fixed 0.5 for every label; not tuned on test |

For batch size B, C labels, logits z, and training-derived positive weights w:

\[
p_{ic}=\sigma(z_{ic}),\qquad
w_c=\frac{N_{\mathrm{train},c}^{-}}{N_{\mathrm{train},c}^{+}},\qquad
\mathcal L=-\frac{1}{BC}\sum_{i=1}^{B}\sum_{c=1}^{C}
\left[w_c y_{ic}\log p_{ic}+(1-y_{ic})\log(1-p_{ic})\right].
\]

The implementation passes logits directly to the numerically stable loss. Sigmoid is applied separately only for prediction. Weighting increases the contribution of rare positive annotations; its scores are not assumed to be calibrated clinical probabilities.

## Evaluation

| Metric | Purpose and convention |
|---|---|
| Per-label AUROC | Ranking performance for each finding |
| Per-label average precision (AP) | Precision–recall summary; compare with that label's positive prevalence |
| Macro AUROC / macro AP | Unweighted mean over labels with both observed classes in the evaluated split; valid-label count always reported |
| Micro AUROC / micro AP | Flatten all image-label pairs; common labels have more influence |
| Macro / micro F1 | Fixed-threshold positive-label performance; undefined F1 set to zero |
| Per-label recall, specificity and confusion matrix | Which findings are missed or over-predicted |
| Hamming loss | Fraction of individual label decisions that are wrong |
| Exact-match accuracy | Fraction of images whose complete selected-label vector is correct |
| Training / fit time | Training batches separately from the loop including validation and checkpoint writes; cached image preparation excluded |

For a label with no positive or no negative held-out image, AUROC and AP are recorded as `null` and excluded from ranking macro averages, rather than silently assigning an arbitrary score. Consequently macro metrics can cover different label subsets in different splits; inspect their counts. AP here is **average precision**, not trapezoidal PR-curve area.

In this fixed cohort, validation has no positives for `callus rib fracture` or `kyphosis`, so checkpoint selection averages **17 of 19 labels**. The test split has both classes for all 19, but several labels have only one positive image. The label list and split are not changed in response to these held-out counts.

The reference predictor assigns each label its training prevalence for every image. It provides per-label AUROC 0.5 where defined and per-label AP equal to held-out prevalence. This contextualizes rare-label results. Test evaluation occurs only after checkpoint selection in this run; the overall test cohort remains exploratory because it was used in earlier experiments.

## Run

```powershell
python -m pip install -r requirements-training.txt
python -m src.train_multilabel --data-dir outputs/normal-resnet18-20261021/data --output outputs/multilabel-resnet18-run
python -m src.report_multilabel outputs/multilabel-resnet18-run/results.json --output-stem results/multilabel-resnet18-run
```

The audited 224-pixel cache and official checkpoint are prerequisites; the [conventional training protocol](normal-training-protocol.md) documents their preparation. The multilabel loader verifies relevant preparation settings, patient-hash assignments, unique images, array dimensions and pixel ranges. It records SHA256 hashes of the reused manifest, pixels and configuration. These hashes identify the local input artifacts, not an independent re-audit of source annotations. The previous binary `target` field is ignored entirely.

New runs require a fresh output directory. Local outputs include initial/best/last model, optimizer and RNG checkpoints; the ordered label vocabulary; training-derived loss weights; per-epoch validation results; aggregate support/metrics; and image-level prediction arrays in manifest order within each split. Only code, aggregate reports and aggregate plots belong in Git.

## Limits and next SISA experiment

This is a single-seed convenience sample and a **19-label subset**, not the full PadChest catalogue or a reproduction of the CheXpert protocol. Sparse labels may have only a handful of held-out positive patients. Patient-bootstrap confidence intervals, multiple seeds and a larger independent evaluation set remain necessary before making performance claims. Resizing to 224 pixels may discard details needed for some findings.

For SISA, match this task, initialization and preprocessing. First establish a fixed label vocabulary as part of the experimental task, independently of patients whose deletion will be tested. This pilot's frequency-based vocabulary and global loss weights are data-dependent: freezing them after a withdrawal does not automatically match training from the retained data. Recomputing a global weight can affect every shard. Define the learning algorithm explicitly, including whether weights are fixed externally or computed within each shard, then verify replay against its clean retained-data reference. No exact-unlearning claim is made by this conventional run.

## Implementation checks

All 21 selected regression tests passed: the 13 existing metadata, SISA and binary-training checks plus eight multilabel checks. New checks cover simultaneous positives, training-only patient-based vocabulary selection, positive weights, independent BCE gradients, missing held-out classes, majority-only predictions, cache split consistency, and a real ResNet training step followed by output restoration.

## Sources

- [PadChest: multilabel findings, annotation provenance and example label lists](https://bimcv.cipf.es/bimcv-projects/padchest/)
- [PyTorch 2.7: multilabel BCEWithLogitsLoss and positive weights](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html)
- [Torchvision ResNet-18 and ImageNet initialization](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)
