"""Inspect PadChest locally: aggregate charts and an image decoded from a ZIP.

No archive extraction, remote requests or Git operations. Outputs stay local.
"""
import argparse
import ast
import csv
import gzip
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .audit_metadata import is_missing, parse_date
except ImportError:
    from audit_metadata import is_missing, parse_date


def inspect(metadata, archive, output):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [entry for entry in bundle.infolist()
                      if not entry.is_dir() and entry.filename.lower().endswith(".png")]
        if not candidates:
            raise ValueError("No PNG images in the supplied archive")
        # Decode one small member; do not extract untrusted paths or the full ZIP.
        selected = min(candidates, key=lambda entry: (entry.file_size, entry.filename))
        if selected.file_size > 100 * 1024 * 1024:
            raise ValueError("Selected PNG exceeds the 100 MiB preview limit")
        with Image.open(io.BytesIO(bundle.read(selected))) as image:
            image.load()
            mode = image.mode
            pixels = np.asarray(image)
    if pixels.ndim != 2:
        raise ValueError("Expected a grayscale radiograph")
    sample_name = PurePosixPath(selected.filename).name
    patient_images, patient_archives = defaultdict(set), defaultdict(set)
    years, counts = Counter(), Counter()
    matches = []
    opener = gzip.open if str(metadata).endswith(".gz") else open
    with opener(metadata, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"PatientID", "ImageID", "ImageDir", "StudyDate_DICOM"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required PadChest metadata columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed CSV record")
            patient, name = row["PatientID"].strip(), row["ImageID"].strip()
            if not is_missing(patient) and not is_missing(name):
                patient_images[patient].add(name)
                if not is_missing(row["ImageDir"]):
                    patient_archives[patient].add(row["ImageDir"].strip())
            date = parse_date(row["StudyDate_DICOM"])
            if date:
                years[date.year] += 1
            if name == sample_name:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one metadata match; found {len(matches)}")
    row = matches[0]
    sample_patient = row["PatientID"].strip()
    labels = [] if is_missing(row.get("Labels", "")) else ast.literal_eval(row["Labels"])
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("Invalid sample label list")
    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max()) + 1
    cmap = "gray_r" if row.get("PhotometricInterpretation_DICOM") == "MONOCHROME1" else "gray"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), facecolor="white")
    axes[0].imshow(pixels, cmap=cmap, vmin=0, vmax=np.iinfo(pixels.dtype).max)
    axes[0].set_title("Full stored pixel range")
    axes[1].imshow(pixels, cmap=cmap, vmin=low, vmax=high)
    axes[1].set_title("1st–99th percentile display window")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"PadChest sample | {pixels.shape[1]} × {pixels.shape[0]} | {mode}", fontsize=15)
    fig.text(.5, .025, "Dataset label: " + ", ".join(labels) + " | Source: " + row.get("MethodLabel", "unknown")
             + "\nDisplay preview only; label has not been clinically validated here.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.11, wspace=.04)
    fig.savefig(output / "sample-preview.png", dpi=150)
    plt.close(fig)

    for images in patient_images.values():
        n = len(images)
        bucket = "1" if n == 1 else "2" if n == 2 else "3–5" if n <= 5 else "6–10" if n <= 10 else "11+"
        counts[bucket] += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    buckets = ["1", "2", "3–5", "6–10", "11+"]
    axes[0].bar(buckets, [counts[b] for b in buckets], color="#167d9a")
    axes[0].set(xlabel="Images per patient", ylabel="Patients", title="Patient workload varies")
    axes[1].bar([str(y) for y in sorted(years)], [years[y] for y in sorted(years)], color="#167d9a")
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].set(xlabel="Released study year", ylabel="Metadata rows", title="Dates parse; comparability is unverified")
    fig.savefig(output / "metadata-overview.png", dpi=150)
    plt.close(fig)
    summary = {
        "archive_png_count": len(candidates), "metadata_matches_for_preview": len(matches),
        "image": {"width": int(pixels.shape[1]), "height": int(pixels.shape[0]), "mode": mode,
                  "dtype": str(pixels.dtype), "pixel_min": int(pixels.min()), "pixel_max": int(pixels.max()),
                  "declared_original_bits_stored": row.get("BitsStored_DICOM"),
                  "metadata_dimensions_match": (int(row["Rows_DICOM"]) == pixels.shape[0]
                                                and int(row["Columns_DICOM"]) == pixels.shape[1])},
        "sample_patient": {"total_metadata_images": len(patient_images[sample_patient]),
                           "archive_count": len(patient_archives[sample_patient])},
        "patients_spanning_multiple_archives": sum(len(a) > 1 for a in patient_archives.values()),
        "patients_by_image_count": {b: counts[b] for b in buckets},
        "rows_outside_documented_2009_2017": sum(n for year, n in years.items() if year < 2009 or year > 2017),
        "cautions": ["Only one local image was decoded; this is not an image-quality audit of the entire dataset.",
                     "Original DICOM BitsStored may differ from PNG storage; do not rescale solely from that tag.",
                     "A downloaded archive is not a patient-complete cohort."]
    }
    (output / "discovery-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inspection"))
    args = parser.parse_args()
    inspect(args.metadata, args.archive, args.output_dir)
