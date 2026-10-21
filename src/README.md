# Source code

Run modules from the repository root, as documented in the main README.

- `audit_metadata.py`: aggregate release audit; identifiers remain strings.
- `inspect_dataset.py` / `inspect_archive.py`: local image previews and patient completeness checks.
- `pilot_data.py`: deterministic patient grouping, fixed hash assignments, per-image preprocessing and local manifests/tensors.
- `sisa.py`: cumulative-stage training, full checkpoints, probability aggregation and patient withdrawal replay.
- `run_pilot.py`: real-data pilot, clean SISA correctness controls, utility and timing results.
