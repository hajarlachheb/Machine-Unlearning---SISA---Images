"""CPU SISA pilot: cumulative slices, full-state replay, stable patient assignments.

Research simulation only. Old versions are invalidated in a lineage ledger but
retained on disk for audits; this is not a physical data-erasure service.
"""
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .pilot_data import keyed_int


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(16 * 4 * 4, 2))

    def forward(self, x):
        return self.layers(x)


def configure_cpu(threads=1):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def initialize(config, shard):
    seed = keyed_int(config["seed"], "initialization", str(shard)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["stage_lr_gamma"])
    return model, optimizer, scheduler


def snapshot(model, optimizer, scheduler):
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()), "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": copy.deepcopy(np.random.get_state()), "python_rng": random.getstate()}


def restore(path, config, shard):
    # Only load checkpoints created by this local experiment. They contain Python
    # and NumPy RNG states, so weights_only=False is required; never load strangers' files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model, optimizer, scheduler = initialize(config, shard)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return model, optimizer, scheduler


def train_stage(model, optimizer, scheduler, x, y, indices, config, shard, stage):
    model.train()
    updates = presentations = 0
    seen = set()
    loss_sum = 0.0
    # Dedicated epoch generator makes sampling independent of other shards and
    # prediction calls. Prefix contents and their order are fixed before shuffling.
    for epoch in range(config["epochs_per_stage"]):
        seed = keyed_int(config["seed"], f"shuffle-{shard}-{stage}", str(epoch)) % (2**63)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(indices), generator=generator).tolist()
        for offset in range(0, len(order), config["batch_size"]):
            batch = [indices[i] for i in order[offset:offset + config["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            updates += 1
            presentations += len(batch)
            loss_sum += loss.item() * len(batch)
            seen.update(batch)
    if updates:
        scheduler.step()
    return {"updates": updates, "image_presentations": presentations,
            "loss": loss_sum / presentations if presentations else None, "seen_indices": sorted(seen)}


@dataclass
class Run:
    directory: Path
    checkpoints: dict
    deleted: set
    stats: dict


def prefix_indices(rows, shard, stage, removed):
    return [i for i,r in enumerate(rows) if r["split"] == "train" and r["shard"] == shard
            and r["slice"] <= stage and r["patient"] not in removed]


def train_all(x, y, rows, config, directory, removed=frozenset()):
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    checkpoints, logs = {}, []
    for shard in range(config["shards"]):
        model, optimizer, scheduler = initialize(config, shard)
        initial = directory / f"shard-{shard}-stage-0.pt"
        torch.save(snapshot(model, optimizer, scheduler), initial)
        checkpoints[shard] = [initial]
        for stage in range(config["slices"]):
            stats = train_stage(model, optimizer, scheduler, x, y,
                                prefix_indices(rows, shard, stage, removed), config, shard, stage)
            path = directory / f"shard-{shard}-stage-{stage+1}.pt"
            torch.save(snapshot(model, optimizer, scheduler), path)
            checkpoints[shard].append(path)
            logs.append({"shard": shard, "stage": stage, **stats})
    stats = {"seconds": time.perf_counter() - start, "stages": logs,
             "updates": sum(l["updates"] for l in logs),
             "image_presentations": sum(l["image_presentations"] for l in logs)}
    (directory / "training-log.json").write_text(json.dumps(stats, indent=2))
    return Run(directory, checkpoints, set(removed), stats)


def unlearn(base, patients, x, y, rows, config, directory):
    requested = set(patients)
    known = {r["patient"] for r in rows if r["split"] == "train"}
    if not requested or not requested.issubset(known):
        raise ValueError("Request must contain known training patients only")
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    newly_removed = requested - base.deleted
    removed = base.deleted | requested
    earliest = {}
    for row in rows:
        if row["split"] == "train" and row["patient"] in newly_removed:
            shard = row["shard"]
            earliest[shard] = min(earliest.get(shard, config["slices"]), row["slice"])
    checkpoints = {s: list(paths) for s,paths in base.checkpoints.items()}
    logs, invalidated = [], []
    for shard, stage0 in sorted(earliest.items()):
        # stage0 is zero-based; checkpoint stage0 is BEFORE that slice entered.
        model, optimizer, scheduler = restore(base.checkpoints[shard][stage0], config, shard)
        invalidated.extend(str(p) for p in base.checkpoints[shard][stage0+1:])
        for stage in range(stage0, config["slices"]):
            stats = train_stage(model, optimizer, scheduler, x, y,
                                prefix_indices(rows, shard, stage, removed), config, shard, stage)
            path = directory / f"shard-{shard}-stage-{stage+1}.pt"
            torch.save(snapshot(model, optimizer, scheduler), path)
            checkpoints[shard][stage+1] = path
            logs.append({"shard": shard, "stage": stage, **stats})
    leaked = {i for l in logs for i in l["seen_indices"] if rows[i]["patient"] in removed}
    assert not leaked, "Withdrawn image reached optimizer"
    stats = {"seconds": time.perf_counter() - start, "stages": logs, "affected_shards": sorted(earliest),
             "newly_removed_patients": len(newly_removed), "total_removed_patients": len(removed),
             "withdrawn_images_seen": len(leaked), "updates": sum(l["updates"] for l in logs),
             "image_presentations": sum(l["image_presentations"] for l in logs)}
    ledger = {"deleted_patients": sorted(removed), "invalidated_checkpoints": invalidated,
              "active_checkpoints": {s: list(map(str,p)) for s,p in checkpoints.items()},
              "scope": "Simulation: superseded files retained for research; do not serve them."}
    (directory / "lineage.json").write_text(json.dumps(ledger, indent=2))
    (directory / "replay-log.json").write_text(json.dumps(stats, indent=2))
    return Run(directory, checkpoints, removed, stats)


def predict(run, x, config):
    if len(x) == 0:
        return torch.empty(0, 2)
    total = torch.zeros(len(x), 2)
    for shard, paths in sorted(run.checkpoints.items()):
        model, _, _ = restore(paths[-1], config, shard)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x), config["batch_size"]):
                total[start:start+config["batch_size"]] += model(x[start:start+config["batch_size"]]).softmax(1)
    return total / len(run.checkpoints)


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a == b


def compare_runs(a, b):
    model_equal = full_equal = True
    max_delta = 0.0
    for shard in a.checkpoints:
        sa = torch.load(a.checkpoints[shard][-1], map_location="cpu", weights_only=False)
        sb = torch.load(b.checkpoints[shard][-1], map_location="cpu", weights_only=False)
        model_equal &= nested_equal(sa["model"], sb["model"])
        full_equal &= nested_equal(sa, sb)
        for key in sa["model"]:
            delta = (sa["model"][key].double() - sb["model"][key].double()).abs().max().item()
            max_delta = max(max_delta, delta)
    return {"model_parameters_and_buffers_exact": bool(model_equal),
            "full_checkpoint_state_exact": bool(full_equal), "max_parameter_or_buffer_delta": max_delta}
