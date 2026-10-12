# Patient-priority unlearning

Research project: reduce whole-patient deletion time in SISA for medical-image classification, while preserving diagnostic utility and establishing the conditions for exact unlearning.

## First milestone: discover the data

PadChest is the provisional primary dataset. Audit the actual release metadata before using acquisition dates to rank patients. CheXpert remains the reproduction and confirmation dataset.

1. Obtain authorized PadChest access and record the release version.
2. Inspect metadata columns, missing values, labels and patient/image identifiers.
3. Confirm whether acquisition dates are comparable between patients.
4. Summarize images and studies per patient and diagnostic coverage.
5. Build patient-disjoint development splits before training.

See `docs/data-discovery.md` for the audit checklist. This scaffold contains no downloaded patient data or experimental results.

## Run the metadata audit

Requires Python 3.10 or newer; no third-party packages are needed. Place your authorized metadata file under the ignored `data/` folder, then run from this repository:

```powershell
python src/audit_metadata.py data/PADCHEST_chest_x_ray_images_labels_160K.csv --output outputs/padchest-audit.json
python -m unittest discover -s tests -v
```

CSV.gz input is also supported. The report covers schema, missingness, duplicate/image-patient conflicts, images and studies per patient, date parsing and label counts. It does not establish cross-patient date comparability automatically. Outputs stay local by default.

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
