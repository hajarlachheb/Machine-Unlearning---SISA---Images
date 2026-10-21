# First SISA mechanics pilot

Protocol started 2026-10-21. This implements the cumulative-slice mechanics in [Bourtoule et al., Machine Unlearning, Section IV](https://arxiv.org/pdf/1912.03817), adapted to whole patients in the available PadChest archive. It is not a reproduction of the original datasets/results or Nasirigerdeh et al.'s medical benchmarks.

## Cohort and target

- Use only patients whose entire metadata-listed image set is present in the downloaded `6.zip`.
- Retain exact `PA`/`AP` projections. Strip label whitespace, deduplicate labels and discard empty lists. Exclude images with `exclude` or `suboptimal study` annotations. Other eligible images from those patients remain included.
- Predict the presence of the `normal` annotation (two-class cross-entropy). Its absence is not a clinically verified abnormal diagnosis. This convenient pilot target does not freeze the final paper's clinical task.
- Decode and CRC-check every retained image, verify dimensions, normalize using its own 1st/99th percentiles, invert MONOCHROME1, bilinearly resize preserving aspect ratio and zero-pad to 64 x 64. No fitted normalization, augmentation, pretrained weights or data-dependent class weights.
- Assign whole patients to train/validation/test with stable seeded hash buckets at nominal 70/15/15 proportions. Actual sizes vary. Seed is fixed before training; do not search seeds against test performance.
- Dates and demographics are retained locally for audit, but do not enter the model, partition or priority score. Date comparability remains unresolved.

## Baseline E2

Use 4 shards and 3 slices, matching the initial setting in the existing Notion task. Independent seeded hash namespaces assign each training patient to one shard and one entry slice. All eligible images of that patient share the assignment. Hash assignment is an engineering choice for deletion stability; it is not perfectly equal-size random partitioning or a diagnostic/workload balancing algorithm. Report actual cell sizes and class counts, without rebalancing after withdrawal. Hard workload caps and their exactness implications remain future work.

For shard s at stage r, train on the cumulative union D[s,1] through D[s,r]. Each stage runs three epochs. Start each shard's two-convolution CNN independently from a seeded initialization. Use SGD (learning rate 0.02, momentum 0.9), batch size 32 and a learning-rate multiplier of 0.9 after each non-empty stage. Average the four softmax probability vectors for prediction. CPU, one thread, deterministic PyTorch operations, no mixed precision.

Save a checkpoint before stage 1 and after every stage: model weights, BatchNorm buffers, optimizer momentum, scheduler state, and Python/NumPy/PyTorch RNG states. Dedicated generators fix epoch permutations by seed/shard/stage/epoch. There is no validation-driven early stopping or model selection.

For withdrawal set U, find each affected shard's earliest entry slice r*. Restore its checkpoint immediately before r*, then replay r* through R on cumulative retained data. Untouched shards reuse their checkpoints. Each replay's actual optimizer inputs are audited against all withdrawn patient IDs. Repeated requests perform no updates; sequential requests retain the previous deletion set.

## Correctness reference and limits

For each request, train **every shard from initialization** on the retained data, with the same patient assignment function, stage schedule, initialization seeds, preprocessing and sampling rules. Compare:

1. Every final model parameter and buffer, including BatchNorm counters.
2. Optimizer, scheduler and RNG checkpoint state.
3. Ensemble probabilities on every eligible image.
4. The actual replay input trace: zero withdrawn images are allowed.

Acceptance for this deterministic local pilot: exact tensor equality and zero maximum probability difference. Synthetic tests cover early/late entry, all-patient images, cross-shard batches, empty stages/shards, sequential and repeated requests. Real-data checks cover early, late and cross-shard batch requests; pick request patients by entry slice and image count, never by evaluation performance.

The replay argument is conditional on fixed hyperparameters, per-image preprocessing, stable identity-based assignment, and identical deterministic computations: unaffected prefixes are identical to clean retained-data training, and the replayed suffix uses exactly the retained inputs. Equality observed in a few tests is implementation evidence, not an unconditional proof, privacy audit or guarantee across hardware/software versions. Priority-based assignment, fitted preprocessing and tuning on subsequently withdrawn data need separate analysis.

This is a **withdrawal simulation**. Original images, cached tensors and superseded experimental checkpoints remain on disk for research. A lineage ledger marks superseded affected checkpoints invalid and points to the active version. This is not an operational erasure service; production deletion would also purge source/cached records, superseded models and relevant backups. Inference must use the returned active run, not an arbitrary old checkpoint.

## Timing and utility

Time training/replay with `perf_counter`, including initialization or restore and checkpoint writes. Exclude initial imports, image preprocessing/cache creation, correctness-reference training and evaluation from replay time. Report data preparation separately. Timing uses already loaded tensors and sequential CPU training, with one observation per scenario; ordering, warm caches and system load can affect ratios.

Report affected shards/stages, optimizer updates, image presentations, checkpoint size, image-level AUROC, accuracy and balanced accuracy on fixed validation/test patients. No confidence intervals or multi-seed claims in this first run. Image-level metrics do not account for within-patient correlation.

A diagnostic monolithic retained-data CNN is also trained for the late-slice request. With R=3 and three epochs per cumulative stage, six full-data epochs approximately match image presentations for equal-size slices: E_full = E_stage (R+1)/2. Actual slices are unequal; report the actual presentations/updates. Its learning-rate schedule differs, and it is neither the exactness reference nor the planned medical-paper E4 reproduction.

## Scope relative to the Notion plan

This advances tasks 02 (pilot manifest/splits), 03 (pilot protocol), 05 (patient-random SISA) and 08 (replay checks). The requested start uses downloaded data before the larger-cohort and CheXpert reproduction milestones. It does not complete the five-seed protocol, shard/slice selection, priority-score method, E3/E5, CheXpert confirmation, or the main timing/utility study. Keep those planned comparisons separate.

Relevant references: [SISA paper](https://arxiv.org/pdf/1912.03817), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [saving optimizer checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
