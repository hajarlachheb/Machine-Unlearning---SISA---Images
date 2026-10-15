# Patient-priority unlearning

Research project: reduce whole-patient deletion time in SISA for medical-image classification, while preserving diagnostic utility and establishing the conditions for exact unlearning.

## First milestone: discover the data

PadChest is the provisional primary dataset. Audit the actual release metadata before using acquisition dates to rank patients. CheXpert remains the reproduction and confirmation dataset.

1. Obtain authorized PadChest access and record the release version.
2. Inspect metadata columns, missing values, labels and patient/image identifiers.
3. Confirm whether acquisition dates are comparable between patients.
4. Summarize images and studies per patient and diagnostic coverage.
5. Build patient-disjoint development splits before training.

See [the first local audit](results/initial-data-audit.md) and [the discovery checklist](docs/data-discovery.md). The downloaded metadata and one sample image have been inspected. Raw data and image previews stay in ignored local folders; this repository contains code and reviewed aggregate findings.

## Run the metadata audit

Requires Python 3.10 or newer; no third-party packages are needed. Place your authorized metadata file under the ignored `data/` folder, then run from this repository:

```powershell
python src/audit_metadata.py data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz --output outputs/padchest-audit.json
python -m unittest discover -s tests -v
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
