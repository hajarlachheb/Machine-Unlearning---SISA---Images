# PadChest archive 6: pilot readiness

Inspected on 2026-10-15 using `src/inspect_archive.py`, the previously audited metadata release, and the downloaded `6.zip`. The archive stays in Downloads; no full extraction or second copy was created.

| Check | Result |
|---|---:|
| ZIP size | 12,173,530,987 bytes (11.34 GiB) |
| PNG entries / metadata matches | 1,645 / 1,645 |
| Expected archive-6 images missing | 0 |
| Images assigned to another archive | 0 |
| Patients represented | 1,047 |
| Patients with every metadata-listed image present | 699 |
| Images belonging to those complete patients, all views | 1,071 |
| Complete patients with multiple images, all views | 331 |
| Complete patients with at least one PA/AP image | 659 |
| PA/AP candidate images from those patients | 692 |
| Candidate patients with multiple PA/AP images | 30 |
| Preview images decoded and CRC-checked | 6 / 6 successful |
| Preview dimensions agreeing with metadata | 6 / 6 |

All six preview images are 16-bit grayscale. The contact sheet was visually checked for rendering. This is a technical loading check, not a clinical quality assessment; the remaining image payloads have not been decoded or CRC-checked. ZIP membership alone does not prove their integrity.

## What this supports

The archive is suitable for an initial loading, grouping and deletion pipeline test. A local candidate CSV contains 692 PA/AP images from 659 complete patients, with image/member names, patient/study IDs, dates, labels and label provenance. It remains in ignored `outputs/archive6/`, together with the image contact sheet.

The candidate filter includes exact `PA` and `AP` projection values. The archive also contains lateral (`L`) and `AP_horizontal` images; these are not included in this provisional frontal cohort. Label handling, exclusions, quality checks and patient-disjoint splits still need to be defined. Completeness is relative to the supplied metadata, not a person's entire clinical history.

Only 30 candidate patients have multiple retained PA/AP images, compared with 331 when counting all views. Thus this download is useful for debugging, but alone offers limited variation for evaluating an image-count priority score. Selecting patients confined to one archive can introduce sampling bias. Expand the cohort for the paper's main experiments and report the selection procedure.

Seven synthetic tests passed, including checks that a missing image excludes the entire patient from the completeness list and that string identifiers remain distinct. No training or unlearning measurements have been performed.
