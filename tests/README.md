# Correctness tests

Tests use synthetic fixtures rather than patient data. Install `requirements-training.txt` and run `python -m unittest discover -s tests -v`.

The SISA tests compare full state and predictions after early/late, cross-shard, sequential, repeated and empty-shard withdrawals against clean retained-data training. They also verify cumulative prefixes, whole-patient exclusion, stable assignments, unchanged checkpoint paths for untouched shards, checkpoint invalidation and rejection of unknown patients. Metadata tests cover identifier preservation, missing values, duplicate/invalid records and ZIP patient completeness.

The conventional-training checks cover grayscale-to-ImageNet normalization, detection of a majority-only classifier, actual backbone/head updates and restoration of ResNet predictions including batch-normalization buffers. They use randomly initialized weights and synthetic images, with no downloads.

On restricted Windows hosts where the default temporary directory is inaccessible, point Python's `tempfile.tempdir` to a writable local directory before invoking the unittest runner. Do not change scientific behavior to work around host permissions.
