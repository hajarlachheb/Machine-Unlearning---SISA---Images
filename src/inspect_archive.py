"""Match a local ZIP to PadChest metadata and preview six images without extraction.

Patient-level manifests and image previews are written only to the chosen local
output directory (default: ignored outputs/). No cohort sampling or splits yet.
"""
import argparse
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
    from .audit_metadata import is_missing
except ImportError:
    from audit_metadata import is_missing


def complete_patients(rows, available):
    """Require every metadata image of a patient to be present in this ZIP."""
    groups = defaultdict(list)
    for row in rows:
        if is_missing(row["PatientID"]) or is_missing(row["ImageID"]):
            raise ValueError("Missing patient or image ID; resolve before checking completeness")
        groups[row["PatientID"]].append(row)
    return {patient: records for patient, records in groups.items()
            if {r["ImageID"] for r in records}.issubset(available)}


def inspect(metadata, archive, output):
    output.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if metadata.suffix == ".gz" else open
    with opener(metadata, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"PatientID", "ImageID", "ImageDir", "Projection", "StudyDate_DICOM",
                    "Rows_DICOM", "Columns_DICOM", "PhotometricInterpretation_DICOM"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Required metadata columns missing")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed metadata row")
    lookup = {r["ImageID"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate image IDs; resolve before matching ZIP")
    with zipfile.ZipFile(archive) as bundle:
        entries = [e for e in bundle.infolist() if not e.is_dir() and e.filename.lower().endswith(".png")]
        members = {PurePosixPath(e.filename).name: e for e in entries}
        if len(members) != len(entries):
            raise ValueError("Duplicate PNG basenames in ZIP")
        available = set(members)
        unknown = available - lookup.keys()
        if unknown:
            raise ValueError(f"{len(unknown)} ZIP images have no metadata match")
        groups = complete_patients(rows, available)
        matched = [lookup[name] for name in sorted(available)]
        expected = {r["ImageID"] for r in rows if r["ImageDir"] == archive.stem}
        complete_rows = [r for patient in sorted(groups) for r in groups[patient]]
        frontal = [r for r in complete_rows if r["Projection"] in {"PA", "AP"}]
        frontal_counts = Counter(r["PatientID"] for r in frontal)
        # This is a candidate manifest, not an eligible training cohort: no label
        # quality filters, train/test split or priority-score selection is applied.
        fields = ["PatientID", "ImageID", "ArchiveMember", "StudyID", "StudyDate_DICOM",
                  "Projection", "Labels", "MethodLabel"]
        with (output / "complete-patient-frontal-candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in frontal:
                item = {f: row.get(f, "") for f in fields}
                item["ArchiveMember"] = members[row["ImageID"]].filename
                writer.writerow(item)

        # Evenly spaced examples from sorted filenames; these are not a random
        # quality sample, and their labels are deliberately omitted from the image.
        names = sorted(available)
        if not names:
            raise ValueError("No PNG images in archive")
        selected = [names[i] for i in np.linspace(0, len(names) - 1, min(6, len(names)), dtype=int)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 9), facecolor="white")
        checks = []
        for ax in axes.flat:
            ax.axis("off")
        for index, (ax, name) in enumerate(zip(axes.flat, selected), 1):
            entry = members[name]
            if entry.file_size > 100 * 1024 * 1024:
                raise ValueError("Preview member exceeds 100 MiB limit")
            with Image.open(io.BytesIO(bundle.read(entry))) as image:
                image.load()
                mode = image.mode
                pixels = np.asarray(image)
            if pixels.ndim != 2:
                raise ValueError("Expected grayscale PNG")
            row = lookup[name]
            height, width = pixels.shape
            low, high = np.percentile(pixels, [1, 99])
            if high <= low:
                low, high = float(pixels.min()), float(pixels.max()) + 1
            cmap = "gray_r" if row["PhotometricInterpretation_DICOM"] == "MONOCHROME1" else "gray"
            ax.imshow(pixels, cmap=cmap, vmin=low, vmax=high)
            ax.set_title(f"Example {index} | {row['Projection']}\n{width} x {height} | {mode}", fontsize=10)
            checks.append({"example": index, "width": width, "height": height, "mode": mode,
                           "dimensions_match_metadata": height == int(row["Rows_DICOM"]) and width == int(row["Columns_DICOM"])})
        fig.suptitle(f"PadChest {archive.name}: six loading checks", fontsize=16)
        fig.text(.5, .018, "1st–99th percentile display window; original pixels unchanged. No diagnostic assessment.", ha="center", fontsize=10)
        fig.subplots_adjust(left=.03, right=.97, bottom=.07, top=.89, wspace=.10, hspace=.30)
        fig.savefig(output / "image-contact-sheet.png", dpi=140)
        plt.close(fig)
    summary = {
        "archive": archive.name, "archive_bytes": archive.stat().st_size,
        "png_entries": len(entries), "uncompressed_bytes": sum(e.file_size for e in entries),
        "metadata_matches": len(matched), "expected_images_missing": len(expected - available),
        "images_not_assigned_to_this_archive": len(available - expected),
        "all_patients_represented": len({r["PatientID"] for r in matched}),
        "complete_patients_all_views": len(groups), "complete_patient_images_all_views": len(complete_rows),
        "complete_patients_with_multiple_images_all_views": sum(len(v) > 1 for v in groups.values()),
        "frontal_candidate_patients": len(frontal_counts), "frontal_candidate_images": len(frontal),
        "frontal_candidate_patients_with_multiple_frontal_images": sum(n > 1 for n in frontal_counts.values()),
        "archive_projection_counts": dict(Counter(r["Projection"] for r in matched)),
        "decoded_preview_checks": checks,
        "limitations": ["ZIP directory checked; CRC and PNG decoding checked only for previewed members.",
                        "Completeness is relative to the supplied metadata, not a patient's entire clinical history.",
                        "Frontal candidate manifest has no label or quality exclusions and is not a training split.",
                        "Selecting patients confined to one archive can bias the cohort; use for a technical pilot only."]
    }
    (output / "archive-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/archive-inspection"))
    args = parser.parse_args()
    inspect(args.metadata, args.archive, args.output_dir)
