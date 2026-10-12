"""Local, standard-library-only PadChest CSV audit. No patient rows are exported."""
import argparse
import ast
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


def parse_date(value):
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    return None


def audit(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    patient_images = defaultdict(set)
    patient_studies = defaultdict(set)
    image_patients = defaultdict(set)
    image_counts = Counter()
    missing = Counter()
    label_rows = Counter()
    label_methods = Counter()
    years = Counter()
    rows = invalid_dates = missing_dates = invalid_labels = 0
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {"PatientID", "ImageID"}
        if not required.issubset(columns):
            raise ValueError("Required columns missing: " + ", ".join(sorted(required - set(columns))))
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate CSV column names; fix schema before auditing.")
        for row in reader:
            rows += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record at data row {rows}.")
            for column in columns:
                if not row[column].strip():
                    missing[column] += 1
            patient, image = row["PatientID"].strip(), row["ImageID"].strip()
            study = row.get("StudyID", "").strip()
            if image:
                image_counts[image] += 1
                if patient:
                    image_patients[image].add(patient)
            if patient:
                patient_images[patient]  # Include patients with no usable image ID.
                patient_studies[patient]
                if image:
                    patient_images[patient].add(image)
                if study:
                    patient_studies[patient].add(study)
            if "StudyDate_DICOM" in columns:
                raw_date = row["StudyDate_DICOM"].strip()
                date = parse_date(raw_date)
                if not raw_date:
                    missing_dates += 1
                elif date is None:
                    invalid_dates += 1
                else:
                    years[str(date.year)] += 1
            if "Labels" in columns and row["Labels"].strip():
                try:
                    labels = ast.literal_eval(row["Labels"])
                    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                        raise ValueError("Expected list of strings")
                    label_rows.update(set(labels))
                except (SyntaxError, ValueError, TypeError):
                    invalid_labels += 1
            if "MethodLabel" in columns:
                label_methods[row["MethodLabel"].strip() or "missing"] += 1

    def summary(groups):
        sizes = [len(values) for values in groups.values()]
        return {"min": min(sizes), "median": median(sizes), "max": max(sizes)} if sizes else None

    return {
        "scope": "Local aggregate audit of supplied metadata; no images opened or downloaded.",
        "columns": columns,
        "rows": rows,
        "unique_patients": len(patient_images),
        "unique_images": len(image_counts),
        "duplicate_image_rows": sum(n - 1 for n in image_counts.values()),
        "images_assigned_to_multiple_patients": sum(len(patients) > 1 for patients in image_patients.values()),
        "missing_cells": {c: missing[c] for c in columns},
        "unique_images_per_patient": summary(patient_images),
        "unique_studies_per_patient": summary(patient_studies) if "StudyID" in columns else None,
        "dates": {"column_present": "StudyDate_DICOM" in columns, "missing": missing_dates,
                  "unparseable": invalid_dates, "rows_by_year": dict(sorted(years.items())),
                  "cross_patient_comparability": "UNVERIFIED: requires release documentation or provider confirmation"},
        "labels": {"column_present": "Labels" in columns, "unparseable_rows": invalid_labels,
                   "row_counts_not_patient_prevalence": dict(sorted(label_rows.items())),
                   "method_row_counts": dict(sorted(label_methods.items()))},
        "limitations": ["Duplicate rows are flagged, not automatically cleaned.",
                        "A parseable date does not establish date comparability or withdrawal likelihood.",
                        "Review aggregate outputs before sharing, especially rare labels and small cohorts."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="Local metadata CSV or CSV.gz")
    parser.add_argument("--output", type=Path, default=Path("outputs/padchest-audit.json"))
    args = parser.parse_args()
    if args.metadata.resolve() == args.output.resolve():
        parser.error("Output must differ from input.")
    report = audit(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audited {report['rows']} rows; aggregate report saved to {args.output}")


if __name__ == "__main__":
    main()
