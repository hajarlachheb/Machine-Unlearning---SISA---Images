# First SISA pilot on downloaded PadChest images

Executed 2026-10-21 with Python 3.13.5 / PyTorch 2.7.1+cpu, one CPU thread, seed 20261021. This is a mechanics pilot, not a clinical result or reproduction of a paper's benchmark. [Protocol](../docs/sisa-pilot-protocol.md) and [configuration](../configs/sisa_pilot.json).

## Data and model

All 686 retained image payloads were decoded, CRC-checked and dimension-checked from the local `6.zip`. The complete-patient set initially contained 1,071 images; filtering removed 379 non-PA/AP images and 6 images with quality/exclusion annotations. The retained cohort has 653 patients. Preparation took 171.38 seconds; training and replay below use the prepared tensors and exclude that cost.

| Split | Patients | Images | `normal` annotation present |
|---|---:|---:|---:|
| Training | 435 | 459 | 187 |
| Validation | 123 | 128 | 51 |
| Test | 95 | 99 | 40 |

Zero patient overlap between any pair of splits. Each training patient belongs to one shard and entry slice. The model is a two-convolution CNN trained from scratch on 64 x 64 grayscale images, with three epochs at each cumulative stage. No priority score, pretrained model, augmentation, balancing or early stopping was used.

| Shard (1-based) | Slice 1 images | Slice 2 images | Slice 3 images | Total |
|---|---:|---:|---:|---:|
| 1 | 30 | 40 | 31 | 101 |
| 2 | 39 | 46 | 34 | 119 |
| 3 | 26 | 39 | 40 | 105 |
| 4 | 45 | 41 | 48 | 134 |

These are entry-slice sizes. Stage 2 trains on slices 1+2, and stage 3 on 1+2+3. Hash-based assignment is stable under deletion, but cell sizes are unequal and no hard workload cap is enforced. All cells contain both target classes in this run.

## Real-data replay checks

Each scenario starts from the same original SISA run. Its reference retrains all four shards from initialization on retained data, with identical assignments, stage schedule and randomness. The early and late requests involve different patients and image counts; this is not a controlled experiment isolating slice position.

| Request | Patients / images removed | Stages replayed (out of 12) | Replay time | Clean SISA time | Observed ratio |
|---|---:|---:|---:|---:|---:|
| Early slice (1) | 1 / 3 | 3 | 1.027 s | 3.455 s | 3.36x |
| Late slice (3) | 1 / 2 | 1 | 0.524 s | 3.447 s | 6.58x |
| Cross-shard batch (slices 1, 2) | 2 / 4 | 5 | 1.624 s | 3.400 s | 2.09x |

**All three checks passed:** every final parameter and buffer, optimizer/scheduler/RNG state, and every ensemble probability matched the clean SISA reference exactly. Maximum parameter/buffer difference = 0; maximum probability difference = 0. Actual replay traces contained zero withdrawn images.

These are single sequential observations with warm, in-memory tensors. Times include restore/initialization, training and checkpoint writes, but exclude preprocessing, reference comparisons and evaluation. Ratios are not statistically established speedups. Original SISA training took 26.886 seconds including first-use initialization overhead; do not compare that cold first run directly to warmed replay. Original training performed 105 updates / 2,715 image presentations, and its 16 checkpoints occupy 548,952 bytes.

## Predictive utility is not ready

| Model | Validation AUROC | Test AUROC | Test accuracy | Test balanced accuracy |
|---|---:|---:|---:|---:|
| Original SISA | 0.553 | 0.706 | 0.596 | 0.500 |
| SISA after late-slice withdrawal | 0.555 | 0.690 | 0.596 | 0.500 |
| Monolithic retained-data comparator, same late request | 0.594 | 0.699 | 0.606 | 0.657 |

At threshold 0.5, the original SISA ensemble predicts the negative class for every validation/test image. Its 59.6% test accuracy equals the test-set majority baseline, and balanced accuracy is 0.5. Correct replay does **not** establish a useful classifier. The target is annotation presence, not a verified healthy/abnormal diagnosis.

The monolithic comparator ran six epochs (90 updates / 2,742 presentations) in 3.322 seconds. Its presentation budget is approximately matched to sliced training, but its learning-rate schedule differs. It is a diagnostic comparator, not a correctness oracle or the medical-paper E4 reproduction. All utility metrics are image-level, one seed, without confidence intervals.

## Validation and next work

Ten synthetic tests passed. They cover metadata and cohort checks plus cumulative slices, stable grouping, early/late and cross-shard deletion, removing an entire shard, sequential requests, repeated requests, checkpoint invalidation and unknown-patient rejection. Real-data checks additionally verify full state and all-image predictions in the three scenarios above.

Tasks 02/03/05/08 have an initial working implementation. Keep them in progress while extending the protocol: improve the task/model with validation-only decisions, evaluate multiple seeds and timings, enforce or justify workload constraints, expand the cohort, and reproduce the medical-paper baselines separately. Priority placement comes after the E2 baseline is stable. Dates remain unused pending comparability verification.

Raw images, cached tensors, identifiers and model checkpoints stay local. This is a withdrawal simulation: old models are invalidated in the lineage ledger but retained as research artifacts, alongside source images. It does not provide physical data erasure or an unconditional privacy certificate.
