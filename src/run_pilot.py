"""Run the local PadChest SISA mechanics pilot, never a clinical benchmark."""
import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .pilot_data import prepare
from .sisa import configure_cpu, train_all, unlearn, predict, compare_runs


def metrics(run, x, rows, config, split):
    ids = [i for i,r in enumerate(rows) if r["split"] == split]
    truth = np.array([rows[i]["target"] for i in ids])
    probabilities = predict(run, x[ids], config)[:, 1].numpy()
    return {"images": len(ids), "positive_images": int(truth.sum()),
            "auroc": float(roc_auc_score(truth, probabilities)) if len(set(truth)) == 2 else None,
            "accuracy": float(accuracy_score(truth, probabilities >= .5)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, probabilities >= .5))}


def run(args):
    config = json.loads(args.config.read_text())
    if min(config[k] for k in ("shards", "slices", "epochs_per_stage", "batch_size", "image_size", "threads")) < 1:
        raise ValueError("Positive training sizes required")
    configure_cpu(config["threads"])
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if args.prepared_data:
        source = args.prepared_data
        if json.loads((source / "config.json").read_text()) != config:
            raise ValueError("Prepared data configuration differs; prepare again in a new output directory")
        rows = json.loads((source / "manifest.json").read_text())
        array = np.load(source / "images.npy", allow_pickle=False)
        summary = json.loads((source / "summary.json").read_text())
    else:
        source = args.output / "data"
        rows, array, summary = prepare(args.metadata, args.archive, source, config)
    preparation_seconds = time.perf_counter() - start
    x = torch.from_numpy(array)
    y = torch.tensor([r["target"] for r in rows], dtype=torch.long)
    if len(x) != len(rows):
        raise ValueError("Tensor count differs from manifest")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Training {config['shards']} shards x {config['slices']} cumulative slices", flush=True)
    initial = train_all(x, y, rows, config, args.output / "original")
    result = {"scope": "Single-seed CPU mechanics pilot; not a clinical or paper-reproduction result",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "threads": config["threads"]},
              "config": config, "data": summary, "data_load_or_prepare_seconds": preparation_seconds,
              "used_prepared_cache": bool(args.prepared_data),
              "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
              "original_training_seconds": initial.stats["seconds"],
              "original_updates": initial.stats["updates"],
              "original_image_presentations": initial.stats["image_presentations"],
              "initial_utility": {s: metrics(initial,x,rows,config,s) for s in ("validation","test")},
              "deletion_cases": []}
    patients = {}
    for row in rows:
        if row["split"] == "train":
            patients[row["patient"]] = (row["shard"],row["slice"])
    counts = Counter(r["patient"] for r in rows if r["split"] == "train")
    early = min((p for p in patients if patients[p][1] == 0), key=lambda p:(-counts[p],p))
    late = min((p for p in patients if patients[p][1] == config["slices"]-1), key=lambda p:(-counts[p],p))
    other = next(p for p in sorted(patients) if patients[p][0] != patients[early][0])
    cases = [("early_slice", {early}), ("late_slice", {late}), ("cross_shard_batch", {early,other})]
    late_removed = None
    for name, forget in cases:
        print(f"Deletion check: {name}", flush=True)
        updated = unlearn(initial,forget,x,y,rows,config,args.output/name)
        clean = train_all(x,y,rows,config,args.output/(name+"-clean-reference"),forget)
        check = compare_runs(updated,clean)
        # Check predictions on every eligible image as well as exact state.
        max_prediction_delta = float((predict(updated,x,config)-predict(clean,x,config)).abs().max())
        record = {"case": name, "patients_removed": len(forget),
                  "images_removed": sum(r["patient"] in forget and r["split"] == "train" for r in rows),
                  "entry_slices_one_based": sorted({patients[p][1]+1 for p in forget}),
                  "affected_shards": len(updated.stats["affected_shards"]),
                  "replayed_stages": len(updated.stats["stages"]),
                  "unlearning_seconds": updated.stats["seconds"],
                  "clean_sisa_seconds": clean.stats["seconds"],
                  "speedup_vs_clean_sisa": clean.stats["seconds"]/max(updated.stats["seconds"],1e-12),
                  "replay_updates": updated.stats["updates"], "clean_updates": clean.stats["updates"],
                  "replay_image_presentations": updated.stats["image_presentations"],
                  "clean_image_presentations": clean.stats["image_presentations"],
                  "withdrawn_images_seen": updated.stats["withdrawn_images_seen"],
                  "max_prediction_delta": max_prediction_delta, **check,
                  "utility": {s:metrics(updated,x,rows,config,s) for s in ("validation","test")}}
        if not check["full_checkpoint_state_exact"] or max_prediction_delta != 0:
            raise AssertionError(f"Replay diverged from matched clean SISA reference: {name}")
        result["deletion_cases"].append(record)
        if name == "late_slice":
            late_removed = forget
    # Diagnostic monolithic comparator, NOT the clean SISA correctness oracle.
    mono_config = {**config,"shards":1,"slices":1,
                   "epochs_per_stage":config["epochs_per_stage"]*(config["slices"]+1)//2}
    mono_rows = [{**r,"shard":0,"slice":0} if r["split"] == "train" else r for r in rows]
    print("Training monolithic retained-data timing comparator", flush=True)
    mono = train_all(x,y,mono_rows,mono_config,args.output/"monolithic-retained",late_removed)
    result["monolithic_retained_comparator"] = {
        "epochs":mono_config["epochs_per_stage"], "seconds":mono.stats["seconds"],
        "updates":mono.stats["updates"], "image_presentations":mono.stats["image_presentations"],
        "utility": {s:metrics(mono,x,rows,mono_config,s) for s in ("validation","test")},
        "scope":"Late-slice withdrawal only; approximate presentation-budget match, different LR schedule; not Nasirigerdeh reproduction"}
    result["checkpoint_bytes_original"] = sum(p.stat().st_size for paths in initial.checkpoints.values() for p in paths)
    (args.output / "results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/sisa_pilot.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepared-data",type=Path)
    run(parser.parse_args())
