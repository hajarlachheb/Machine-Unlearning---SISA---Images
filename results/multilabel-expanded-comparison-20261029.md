# Expanded training: comparison on the same test images

Both runs use identical initial model parameters/buffers and the same ordered 19 findings. Predictions are aligned by image ID after checking patient, study and annotation identity. These metrics are computed after validation-only checkpoint selection.

| Evaluation | Patients | Images | Macro AUROC | Macro AP | Micro F1 at 0.5 |
|---|---:|---:|---:|---:|---:|
| original model original test | 95 | 99 | 0.705 | 0.193 | 0.215 |
| expanded model original test | 95 | 99 | 0.814 | 0.314 | 0.215 |
| expanded model additional test | 250 | 264 | 0.710 | 0.190 | 0.195 |
| expanded model full test | 345 | 363 | 0.733 | 0.180 | 0.201 |

The first two rows compare the same original test images. The other rows describe different test populations and should not be interpreted as a controlled before/after improvement. Training and validation cohorts expanded together; six epochs on more training images also mean more optimizer updates, and training-derived class weights changed. This is an exploratory, single-seed comparison without uncertainty intervals, not a causal estimate of the effect of data quantity alone. Original test patients were inspected in earlier experiments. No test metric was used to select the checkpoint or tune thresholds.
