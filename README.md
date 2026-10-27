# Machine Unlearning - SISA - Images

Research project: reduce whole-patient deletion time in SISA for medical-image classification, while preserving diagnostic utility and establishing the conditions for exact unlearning.

## Multilabel ResNet-18: the current conventional baseline

Train one ResNet-18 to predict **19 findings independently**, allowing several positives per image. Labels are selected from training-patient support only; a new head is fine-tuned with weighted `BCEWithLogitsLoss` from ImageNet initialization. The 686-image patient-disjoint cohort and 224×224 preprocessing are unchanged. This is a supported-label subset of PadChest, not its full catalogue.

```powershell
python -m src.train_multilabel --data-dir outputs/normal-resnet18-20261021/data --output outputs/my-multilabel-run
python -m src.report_multilabel outputs/my-multilabel-run/results.json --output-stem results/my-multilabel-run
```

The audited cache and official ImageNet checkpoint are required; their preparation is documented in the earlier conventional protocol below. See the [multilabel protocol](docs/multilabel-training-protocol.md) and [first measured multilabel results](results/multilabel-resnet18-20261022.md). The six-epoch run selected epoch 4: test macro AUROC 0.705, macro AP 0.193 and micro F1 0.215 at threshold 0.5. Sparse held-out support and overfitting limit interpretation. The next SISA experiment should use this multilabel task with a matched model and preprocessing.

The new `54` image folder is now verified and combined with `6.zip`: **2,441 eligible images from 2,317 patients**, with zero patient overlap. See the [expanded-cohort audit](results/cohort-6-54-20261027.md) and [preparation/next-training commands](docs/expanded-cohort.md). All 686 original images retain identical pixels and splits, and the 19-label task is fixed. **Training on the expanded cohort has not been run yet**; the metrics above are from the original small cohort.

## Earlier binary pipeline check

Use **ResNet-18 with ImageNet weights**, fine-tuning all layers on the downloaded PadChest subset. This conventional baseline uses one model, no shards or slices, and 224×224 inputs prepared from the original PNGs. See the [model choice, training settings and commands](docs/normal-training-protocol.md). Checkpoint selection uses validation AUROC. The current target is presence of the `normal` annotation, not a confirmed normal-versus-abnormal diagnosis.

The next fair SISA comparison must use the same backbone, initialization, target and preprocessing. The earlier TinyCNN run below checks SISA mechanics and is not a matched comparator.

## Run the first SISA pilot

The first patient-random SISA implementation uses the downloaded PadChest `6.zip`: **4 shards x 3 cumulative slices**, full-state checkpoints and whole-patient replay. This is a single-seed CPU mechanics pilot with a small CNN, not a medical benchmark reproduction. Read the [pilot protocol](docs/sisa-pilot-protocol.md) and [first measured results](results/sisa-pilot-20261021.md). All three real-data replay checks match clean retained-data SISA exactly; predictive quality still needs improvement.

```powershell
python -m pip install -r requirements-training.txt
python -m src.run_pilot --metadata data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz --archive "$env:USERPROFILE/Downloads/6.zip" --output outputs/my-first-sisa-run
python -m unittest discover -s tests -v
```

Use a new output directory for each run; existing runs are never overwritten. The command prepares patient-disjoint splits, decodes retained images, trains SISA, tests early/late/cross-shard withdrawals against clean retained-data SISA, and runs a diagnostic monolithic retained-data comparator. It records timings, image presentations, checkpoint-state comparisons and utility in `results.json` under the local output directory. Raw identifiers, tensors, models and traces remain ignored by Git.

The pilot predicts the presence of the `normal` annotation; its absence is not a confirmed abnormal diagnosis. Eligibility, image preprocessing, fixed seed and training settings are recorded in `configs/sisa_pilot.json`. Hash-based patient assignments survive deletion but do not enforce equal workloads. Model/state equality is checked on this deterministic implementation, not asserted as a general privacy certificate.

**Withdrawal simulations retain historical experimental checkpoints and source data locally.** A lineage ledger invalidates affected old checkpoints and records the active version. This prototype does not physically erase records or backups.

## First milestone: discover the data

PadChest is the provisional primary dataset. Audit the actual release metadata before using acquisition dates to rank patients. CheXpert remains the reproduction and confirmation dataset.

1. Obtain authorized PadChest access and record the release version.
2. Inspect metadata columns, missing values, labels and patient/image identifiers.
3. Confirm whether acquisition dates are comparable between patients.
4. Summarize images and studies per patient and diagnostic coverage.
5. Build patient-disjoint development splits before training.

See [the first local audit](results/initial-data-audit.md), [archive 6 inspection](results/archive6-audit.md) and [the discovery checklist](docs/data-discovery.md). Raw data and image previews stay in ignored local folders; this repository contains code and reviewed aggregate findings.

## Run the metadata audit

Requires Python 3.10 or newer; no third-party packages are needed. Place your authorized metadata file under the ignored `data/` folder, then run from this repository:

```powershell
python src/audit_metadata.py data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz --output outputs/padchest-audit.json
python -m unittest discover -s tests -p test_audit_metadata.py -v
```

CSV.gz input is also supported. The report covers schema, missingness, duplicate/image-patient conflicts, images and studies per patient, date parsing and label counts. It does not establish cross-patient date comparability automatically. Outputs stay local by default.

## Preview the local sample

Install the optional inspection dependencies, then read one PNG directly from the downloaded ZIP:

```powershell
python -m pip install -r requirements-inspection.txt
python src/inspect_dataset.py data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz data/raw/50.zip
```

Open `outputs/inspection/sample-preview.png` for the radiograph, `metadata-overview.png` for aggregate charts, and `discovery-summary.json` for image checks and patient completeness. The preview uses two display windows; it does not change the original pixels or define training preprocessing. The script reads one image, not every image in a larger archive.

The supplied `50.zip` contains only one image. It is sufficient to test loading, but not to train or evaluate unlearning. Select patients from the metadata and obtain their eligible images across archives before creating patient-disjoint splits. Keep dataset copies and individual previews local in accordance with the supplied dataset terms.

## Inspect the downloaded pilot archive

Archive `6.zip` has been matched to the metadata: 1,645 PNGs and 699 patients whose complete listed image sets are present. See [archive findings](results/archive6-audit.md). Read directly from Downloads to avoid copying or extracting the entire archive:

```powershell
python src/inspect_archive.py data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz "$env:USERPROFILE/Downloads/6.zip" --output-dir outputs/archive6
python -m unittest discover -s tests -v
```

Install `requirements-inspection.txt` first for these commands. Outputs include a six-image contact sheet, aggregate JSON, and a **local-only** candidate CSV for complete patients with `Projection` exactly `PA` or `AP`. `AP_horizontal` and lateral views are excluded from this provisional candidate list. No label/quality filters or train/test assignments have been applied. Full-archive CRC/decode validation is not performed; six preview images are decoded and CRC-checked. Patients confined to one archive are a convenience sample for debugging, not a representative final study cohort.

## Layout

- `configs/`: dataset and experiment configuration.
- `src/`: data preparation, training and unlearning code.
- `tests/`: patient-split and deletion-correctness checks.
- `docs/`: protocol and dataset audit.
- `results/`: reviewed aggregate summaries and figures only.

## Planned comparisons

- E1: patient-priority SISA, high-priority patients entering later slices.
- E2: patient-random SISA, using matched patient-to-shard assignments.
- E3: image-random SISA, deleting every image of the withdrawing patient.
- E4: full retained-data retraining.
- E5: random relabeling and saliency unlearning (approximate baselines).

Each SISA variant needs a clean retained-data SISA correctness control. Data-dependent partitioning requires an explicit exactness argument; numerical prediction similarity alone does not prove exact unlearning.

## Sources

- [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/1000/)
- [PadChest paper](https://arxiv.org/abs/1901.07441)
- [Medical unlearning baseline](https://arxiv.org/abs/2407.07539)
- [SISA](https://arxiv.org/abs/1912.03817)
