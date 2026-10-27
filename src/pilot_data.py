"""Patient-disjoint, deletion-stable preparation from one local PadChest ZIP."""
import ast
import csv
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import numpy as np
from PIL import Image


def keyed_int(seed, namespace, identifier):
    value = json.dumps([seed, namespace, identifier], ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def assignment(patient, config):
    seed = config["seed"]
    a, b, c = config["split_percentages"]
    if min(a, b, c) <= 0 or a + b + c != 100:
        raise ValueError("Three positive split percentages must sum to 100")
    draw = keyed_int(seed, "split", patient) % 100
    split = "train" if draw < a else "validation" if draw < a + b else "test"
    return {"split": split,
            "shard": keyed_int(seed, "shard", patient) % config["shards"] if split == "train" else -1,
            "slice": keyed_int(seed, "slice", patient) % config["slices"] if split == "train" else -1}


def normalized_labels(raw):
    if raw.strip().lower() in {"", "nan", "none", "null"}:
        return None
    value = ast.literal_eval(raw)
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("Expected a list of label strings")
    return sorted({x.strip() for x in value if x.strip()})


def preprocess_pixels(pixels, photo, size):
    """Shared, per-image transform for the original and expanded cohorts."""
    if pixels.ndim != 2 or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("Expected integer grayscale PNG")
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Unknown photometric interpretation")
    lo, hi = np.percentile(pixels, [1, 99])
    if hi <= lo:
        raise ValueError("Constant/degenerate image; revise eligibility explicitly")
    values = np.clip((pixels.astype(np.float32) - lo) / (hi - lo), 0, 1).astype(np.float32)
    if photo == "MONOCHROME1":
        values = 1 - values
    height, width = values.shape
    scale = min(size / width, size / height)
    resized = np.asarray(Image.fromarray(values).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR))
    canvas = np.zeros((size, size), dtype=np.float32)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas[None]


def prepare(metadata, archive, output, config):
    """Decode all eligible images; fail on corruption instead of silently dropping."""
    output.mkdir(parents=True, exist_ok=False)
    opener = gzip.open if metadata.suffix == ".gz" else open
    with opener(metadata, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed metadata row")
    if len({r["ImageID"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    grouped = defaultdict(list)
    for row in rows:
        patient = row["PatientID"].strip()
        if not patient or patient.lower() in {"nan", "none"}:
            raise ValueError("Missing patient ID")
        grouped[patient].append(row)
    filtered = Counter()
    manifest, arrays = [], []
    with zipfile.ZipFile(archive) as bundle:
        entries = [e for e in bundle.infolist() if not e.is_dir() and e.filename.lower().endswith(".png")]
        members = {PurePosixPath(e.filename).name: e for e in entries}
        if len(members) != len(entries):
            raise ValueError("Duplicate image filenames in archive")
        complete = {p: records for p, records in grouped.items()
                    if {r["ImageID"] for r in records}.issubset(members)}
        for patient in sorted(complete):
            for row in sorted(complete[patient], key=lambda r: r["ImageID"]):
                if row["Projection"] not in config["allowed_projections"]:
                    filtered["projection"] += 1
                    continue
                labels = normalized_labels(row["Labels"])
                if not labels:
                    filtered["missing_or_empty_labels"] += 1
                    continue
                if set(labels) & set(config["excluded_labels"]):
                    filtered["quality_label"] += 1
                    continue
                entry = members[row["ImageID"]]
                if entry.file_size > 100 * 1024 * 1024:
                    raise ValueError("PNG exceeds 100 MiB limit")
                with Image.open(io.BytesIO(bundle.read(entry))) as image:
                    image.load()  # ZIP read checks this member's CRC.
                    pixels = np.array(image)
                if pixels.ndim != 2 or not np.issubdtype(pixels.dtype, np.integer):
                    raise ValueError("Expected integer grayscale PNG")
                if pixels.shape != (int(row["Rows_DICOM"]), int(row["Columns_DICOM"])):
                    raise ValueError("Decoded dimensions do not match metadata")
                photo = row["PhotometricInterpretation_DICOM"]
                arrays.append(preprocess_pixels(pixels, photo, config["image_size"]))
                manifest.append({"patient": patient, "image": row["ImageID"], "member": entry.filename,
                                 "study": row["StudyID"], "date": row["StudyDate_DICOM"],
                                 "sex": row["PatientSex_DICOM"], "projection": row["Projection"],
                                 "labels": labels, "method_label": row["MethodLabel"],
                                 "target": int(config["target_label"] in labels), **assignment(patient, config)})
                if len(arrays) % 100 == 0:
                    print(f"Decoded {len(arrays)} eligible images", flush=True)
    if not arrays:
        raise ValueError("No eligible images")
    for split in ("train", "validation", "test"):
        if {r["target"] for r in manifest if r["split"] == split} != {0, 1}:
            raise ValueError(f"Both target classes are required in {split}; do not silently change the seed")
    np.save(output / "images.npy", np.stack(arrays))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"complete_patients_before_filters": len(complete), "excluded_images": dict(filtered),
               "eligible_images": len(manifest), "eligible_patients": len({r["patient"] for r in manifest}),
               "splits": {}, "shard_slice_counts": []}
    for split in ("train", "validation", "test"):
        records = [r for r in manifest if r["split"] == split]
        patients = {r["patient"] for r in records}
        sex = Counter(next(r["sex"] for r in records if r["patient"] == p) for p in patients)
        summary["splits"][split] = {"patients": len(patients), "images": len(records),
                                    "positive_images": sum(r["target"] for r in records),
                                    "patient_sex_counts": dict(sex)}
    for shard in range(config["shards"]):
        for stage in range(config["slices"]):
            cell = [r for r in manifest if r["shard"] == shard and r["slice"] == stage]
            summary["shard_slice_counts"].append({"shard": shard, "slice": stage,
                "patients": len({r["patient"] for r in cell}), "images": len(cell),
                "positive_images": sum(r["target"] for r in cell)})
    summary["patient_overlap"] = {f"{a}/{b}": len({r["patient"] for r in manifest if r["split"] == a}
        & {r["patient"] for r in manifest if r["split"] == b}) for a,b in [("train","validation"),("train","test"),("validation","test")]}
    summary["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    summary["archive_bytes"] = archive.stat().st_size
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest, np.stack(arrays), summary
