# Expand the PadChest cohort with archive 54

The original cohort used `6.zip`. The next cohort reads that ZIP together with the extracted `54` directory in place, avoiding another large copy. The archive-54 inventory matches all 4,297 expected PNG filenames; archive 6 contributes 1,645. Inventory completeness does not by itself verify image contents.

The completed [audit](../results/cohort-6-54-20261027.md) verified all 2,441 eligible images from 2,317 patients and preserved all 686 earlier images with bitwise-identical prepared pixels. Splits contain 1,665/413/363 images from 1,586/386/345 patients. All 25 selected regression tests passed. Only 112 patients have more than one eligible PA/AP image, so the variation in image count per patient remains limited for patient-priority studies. All eligible annotations are marked `Physician` in the supplied metadata.

## Preparation and verification

```powershell
python -m src.prepare_cohort --metadata data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz --source "6=$env:USERPROFILE/Downloads/6.zip" --source "54=$env:USERPROFILE/Downloads/54" --config configs/multilabel_resnet18_expanded.json --previous-cache outputs/normal-resnet18-20261021/data --output outputs/cohort-6-54-20261027
```

The source IDs before `=` refer to the metadata `ImageDir`, independently of the directory or ZIP's filename. Each source must contain exactly its expected PNG basenames, without duplicates or missing entries. All metadata-listed images of a patient must be present across the combined sources before applying the existing PA/AP and quality-label filters.

Every eligible image is read and decoded, PNG structure/chunks and dimensions are checked, and the original PNG payload is hashed in the local manifest. ZIP reads additionally check their entry CRC. Images excluded by eligibility are inventoried but not decoded. Preprocessing uses the same shared percentile window, polarity handling, aspect-preserving resize, zero padding and float32 224×224 representation as before.

The seed and patient-hash split rule are unchanged. Preparation checks that every previously eligible image is retained with identical patient, study, date, labels and split, and that its prepared pixels are exactly equal to the earlier cache. Raw source images are not copied, moved or deleted. The output is a memory-mapped resized array and local metadata/configuration/aggregate checks. A successful run writes `summary.json`; an interrupted directory must not be treated as a complete cache.

Export aggregate counts without patient identifiers after preparation succeeds:

```powershell
python -m src.report_cohort outputs/cohort-6-54-20261027/summary.json --previous-results results/multilabel-resnet18-20261022.json --output-stem results/cohort-6-54-20261027
```

## Training on the expanded cohort

The expanded configuration explicitly fixes the **same ordered 19-label vocabulary** from the initial multilabel experiment. New common labels are not added automatically. Training-image positive weights will be recalculated from the expanded training split. This isolates dataset expansion while keeping model, targets, preprocessing, epochs and split rule unchanged.

```powershell
python -m src.train_multilabel --config configs/multilabel_resnet18_expanded.json --data-dir outputs/cohort-6-54-20261027 --output outputs/multilabel-resnet18-expanded-run
python -m src.report_multilabel outputs/multilabel-resnet18-expanded-run/results.json --output-stem results/multilabel-resnet18-expanded-run
```

These commands reproduce the expanded training experiment; preparing the data alone does not train a model. Use fresh output directories. Model initialization remains the original ImageNet checkpoint, rather than the binary or prior multilabel fitted checkpoint.

The run launched on October 27 completed all six epochs and was checked on October 29. Validation macro AUROC selected epoch 3. Training took 787.7 seconds; the full fitting loop including validation/checkpoint writes took 857.4 seconds. Full-test macro AUROC/AP/micro F1 are 0.733/0.180/0.201. See the [results and curves](../results/multilabel-resnet18-expanded-20261027.md).

The [comparison on identical original test images](../results/multilabel-expanded-comparison-20261029.md) checks unchanged patient/study/label identity, ordered label vocabulary, input-manifest hashes and exact equality of initial model parameters and buffers. Original-test macro AUROC changed from 0.705 to 0.814, AP from 0.193 to 0.314, and micro F1 remained about 0.215 at threshold 0.5. Both models were selected using their own validation cohorts. Three added synthetic alignment checks passed; no model training code or hyperparameters were changed for this evaluation.

```powershell
python -m src.compare_cohorts --old-run outputs/multilabel-resnet18-20261022 --new-run outputs/multilabel-resnet18-expanded-20261027 --old-data outputs/normal-resnet18-20261021/data --new-data outputs/cohort-6-54-20261027 --output-stem results/multilabel-expanded-comparison-20261029
```

For comparisons, report the original held-out cohort and the additional held-out patients separately as well as the expanded aggregate: aggregate scores across different test populations are not a controlled estimate of improvement. Previous test patients remain previously inspected, even when new patients are added. Sparse findings and the archive-based convenience sample still limit generalization. The frequency-selected label vocabulary remains an exploratory task definition; exact SISA requires the treatment described in the [multilabel protocol](multilabel-training-protocol.md).
