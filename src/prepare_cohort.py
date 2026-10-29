"""Prepare a patient-complete cohort from a mixture of PadChest ZIPs and folders.

Source specifications are ImageDir=path, e.g. --source 6=.../6.zip --source 54=.../54.
Source files are read in place; only the small resized cache is written locally.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
from PIL import Image

from src.pilot_data import assignment, normalized_labels, preprocess_pixels


class Source:
    def __init__(self, name, path, stack):
        self.name, self.path = name, path
        if path.is_dir():
            self.kind = "folder"
            self.bundle = None
            entries = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
            self.entries = {p.name: p for p in entries}
        else:
            self.kind = "zip"
            self.bundle = stack.enter_context(zipfile.ZipFile(path))
            entries = [e for e in self.bundle.infolist() if not e.is_dir() and e.filename.lower().endswith(".png")]
            self.entries = {PurePosixPath(e.filename).name: e for e in entries}
        if len(self.entries) != len(entries):
            raise ValueError(f"Duplicate PNG basenames in source {name}")

    def read(self, image):
        entry = self.entries[image]
        size = entry.stat().st_size if self.kind == "folder" else entry.file_size
        if size > 100 * 1024 * 1024:
            raise ValueError("PNG exceeds 100 MiB limit")
        return entry.read_bytes() if self.kind == "folder" else self.bundle.read(entry)

    def member(self, image):
        entry = self.entries[image]
        return entry.relative_to(self.path).as_posix() if self.kind == "folder" else entry.filename


def select_records(rows, available, config):
    groups = defaultdict(list)
    for row in rows:
        if not row["PatientID"].strip() or row["PatientID"].strip().lower() in {"nan", "none", "null"}:
            raise ValueError("Missing patient ID")
        groups[row["PatientID"]].append(row)
    complete = {p: rs for p, rs in groups.items() if {r["ImageID"] for r in rs} <= available}
    selected, filtered = [], Counter()
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
            selected.append((row, labels))
    return selected, {"complete_patients_before_filters": len(complete), "excluded_images": dict(filtered)}


def prepare(metadata, specifications, output, config, previous_cache=None):
    output.mkdir(parents=True, exist_ok=False)
    opener = gzip.open if metadata.suffix == ".gz" else open
    with opener(metadata, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Empty/malformed metadata")
    required = {"ImageID", "PatientID", "ImageDir", "Projection", "Labels", "Rows_DICOM", "Columns_DICOM",
                "PhotometricInterpretation_DICOM", "StudyID", "StudyDate_DICOM", "PatientSex_DICOM", "MethodLabel"}
    if not required <= rows[0].keys() or len({r["ImageID"] for r in rows}) != len(rows):
        raise ValueError("Missing metadata fields or duplicate image IDs")
    source_audit, manifest, old_lookup = [], [], {}
    if previous_cache:
        old_config = json.loads((previous_cache / "config.json").read_text())
        for key in ("seed", "image_size", "allowed_projections", "excluded_labels", "split_percentages"):
            if old_config[key] != config[key]:
                raise ValueError("Previous-cache preparation settings differ")
        old_rows = json.loads((previous_cache / "manifest.json").read_text())
        old_lookup = {r["image"]: (i, r) for i, r in enumerate(old_rows)}
        old_pixels = np.load(previous_cache / "images.npy", mmap_mode="r")
    started = time.perf_counter()
    with ExitStack() as stack:
        sources, available = {}, set()
        for name, path in specifications:
            if name in sources:
                raise ValueError("Duplicate source ImageDir")
            source = Source(name, path, stack)
            expected = {r["ImageID"] for r in rows if r["ImageDir"] == name}
            actual = set(source.entries)
            if not expected or actual != expected:
                raise ValueError(f"Source {name}: {len(expected-actual)} missing and {len(actual-expected)} unexpected images")
            if actual & available:
                raise ValueError("Image occurs in multiple sources")
            sources[name] = source
            available.update(actual)
            source_audit.append({"image_dir": name, "kind": source.kind, "png_files": len(actual),
                                 "missing": 0, "unexpected": 0})
        selected, summary = select_records(rows, available, config)
        if not selected:
            raise ValueError("No eligible images")
        print(f"Inventory matched: {len(available)} PNGs; preparing {len(selected)} eligible images", flush=True)
        arrays = np.lib.format.open_memmap(output / "images.npy", mode="w+", dtype=np.float32,
                                           shape=(len(selected), 1, config["image_size"], config["image_size"]))
        previous_checked = degenerate = 0
        for n, (row, labels) in enumerate(selected):
            i = n - degenerate
            source = sources[row["ImageDir"]]
            payload = source.read(row["ImageID"])
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()  # PNG structure/chunk integrity, including extracted files.
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                pixels = np.asarray(image)
            if pixels.shape != (int(row["Rows_DICOM"]), int(row["Columns_DICOM"])):
                raise ValueError("Decoded shape differs from metadata")
            lo, hi = np.percentile(pixels, [1, 99])
            if hi <= lo:  # Explicit eligibility rule: no usable intensity window (e.g. blank image).
                if row["ImageID"] in old_lookup:
                    raise ValueError("A previously eligible image is now degenerate")
                degenerate += 1
                continue
            arrays[i] = preprocess_pixels(pixels, row["PhotometricInterpretation_DICOM"], config["image_size"])
            record = {"patient": row["PatientID"], "image": row["ImageID"], "member": source.member(row["ImageID"]),
                      "source": row["ImageDir"], "source_png_sha256": hashlib.sha256(payload).hexdigest(),
                      "study": row["StudyID"], "date": row["StudyDate_DICOM"], "sex": row["PatientSex_DICOM"],
                      "projection": row["Projection"], "labels": labels, "method_label": row["MethodLabel"],
                      **assignment(row["PatientID"], config)}
            if record["image"] in old_lookup:
                j, old = old_lookup[record["image"]]
                if any(record[key] != old[key] for key in ("patient", "split", "labels", "projection", "study", "date")):
                    raise ValueError("An existing image's patient/split/annotation changed")
                if not np.array_equal(arrays[i], old_pixels[j]):
                    raise ValueError("Preprocessing differs from the previous cache")
                previous_checked += 1
            manifest.append(record)
            if (n + 1) % 100 == 0:
                print(f"Verified and prepared {n+1}/{len(selected)} images", flush=True)
        arrays.flush()
        if degenerate:
            trimmed = np.lib.format.open_memmap(output / "images.trimmed.npy", mode="w+", dtype=np.float32,
                                                shape=(len(manifest),) + arrays.shape[1:])
            trimmed[:] = arrays[:len(manifest)]
            trimmed.flush()
            del trimmed
            (output / "images.trimmed.npy").replace(output / "images.npy")
        del arrays
        summary["excluded_images"]["degenerate_pixels"] = degenerate
    if previous_cache and previous_checked != len(old_lookup):
        raise ValueError("Expanded cohort lost previously eligible images")
    summary.update({"sources": source_audit, "eligible_images": len(manifest),
                    "eligible_patients": len({r["patient"] for r in manifest}),
                    "previous_images_preserved_and_pixels_identical": previous_checked,
                    "verified_eligible_pngs": len(manifest), "splits": {},
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "preparation_seconds": time.perf_counter() - started})
    patient_sets = {split: {r["patient"] for r in manifest if r["split"] == split}
                    for split in ("train", "validation", "test")}
    summary["patient_overlap"] = {f"{a}/{b}": len(patient_sets[a] & patient_sets[b])
        for a,b in (("train","validation"),("train","test"),("validation","test"))}
    if any(summary["patient_overlap"].values()):
        raise ValueError("Patient overlap")
    for split, patients in patient_sets.items():
        records = [r for r in manifest if r["split"] == split]
        counts = Counter(r["patient"] for r in records)
        summary["splits"][split] = {"patients": len(patients), "images": len(records),
            "patients_with_multiple_eligible_images": sum(n > 1 for n in counts.values()),
            "annotation_method_images": dict(Counter(r["method_label"] for r in records)),
            "labels": {label: {"positive_images": sum(label in r["labels"] for r in records),
                                "positive_patients": len({r["patient"] for r in records if label in r["labels"]})}
                       for label in config.get("fixed_labels", [])}}
    for name, data in (("config.json", config), ("manifest.json", manifest), ("summary.json", summary)):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-cache", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(shards=1, slices=1)
    specs = []
    for spec in args.source:
        name, path = spec.split("=", 1)
        specs.append((name, Path(path)))
    prepare(args.metadata, specs, args.output, config, args.previous_cache)


if __name__ == "__main__":
    main()
