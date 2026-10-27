"""Write an aggregate-only expansion report from a completed preparation summary."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    old = json.loads(args.previous_results.read_text())["support"]
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Expanded PadChest cohort: archives 6 and 54", "",
             "Verified and prepared data for the next multilabel ResNet-18 run. **No model has been retrained on this expanded cache yet.**", "",
             "| Check | Result |", "|---|---|",
             f"| Available PNG filenames | {sum(s['png_files'] for s in result['sources'])} across both sources; no missing/unexpected names |",
             f"| Eligible images decoded and checked | {result['verified_eligible_pngs']} |",
             f"| Eligible patients | {result['eligible_patients']} |",
             f"| Previous images retained with identical pixels/splits | {result['previous_images_preserved_and_pixels_identical']} |",
             f"| Patient overlap | {sum(result['patient_overlap'].values())} |",
             f"| Preparation wall time | {result['preparation_seconds']/60:.2f} minutes |", "",
             "| Split | Previous patients | Expanded patients | Previous images | Expanded images | Expanded patients with multiple eligible images |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, data in result["splits"].items():
        lines.append(f"| {split} | {old[split]['patients']} | {data['patients']} | {old[split]['images']} | "
                     f"{data['images']} | {data['patients_with_multiple_eligible_images']} |")
    lines += ["", "## Finding coverage", "",
              "The same 19 ordered findings are retained. Positive-patient counts below refer to recorded annotations; absent annotations do not establish clinical negatives.", "",
              "| Finding | Previous positive training patients | Expanded positive training patients | Expanded positive validation patients | Expanded positive test patients |",
              "|---|---:|---:|---:|---:|"]
    for label, data in result["splits"]["train"]["labels"].items():
        lines.append(f"| {label} | {old['train']['labels'][label]['positive_patients']} | {data['positive_patients']} | "
                     f"{result['splits']['validation']['labels'][label]['positive_patients']} | "
                     f"{result['splits']['test']['labels'][label]['positive_patients']} |")
    lines += ["", "All eligible PNGs passed structure/chunk verification, decoding, metadata-dimension checks and the shared preprocessing. "
              "ZIP reads also check entry CRC. Excluded PNGs were inventoried but not decoded. "
              "Patient completeness is assessed using all metadata-listed images across the available sources, before PA/AP and quality filtering.", "",
              "Raw data and identifiers stay local. Source PNG hashes are recorded only in the local manifest. The cache uses the same per-image preprocessing and patient split seed as the original experiment. "
              "More data does not by itself establish improved model performance; the next training run must measure it. "
              "The original test cohort remains exploratory, and aggregate scores on an expanded test population are not directly comparable to the old aggregate.", "",
              "[Preparation and next-training commands](../docs/expanded-cohort.md).", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Created aggregate cohort report: {stem}")


if __name__ == "__main__":
    main()
