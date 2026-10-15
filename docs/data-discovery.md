# PadChest discovery checklist

Status: initial metadata audit and one-image inspection completed on 2026-10-15. See [findings](../results/initial-data-audit.md). Patient-complete cohort construction and date-policy verification remain open.

## Access and provenance

- [ ] Record official access approval, usage terms and release/version identifier.
- [ ] Obtain the metadata dictionary and CSV before the full image collection.
- [x] Record input checksums; keep raw metadata, identifiers and image paths out of Git.

## Metadata audit

- [x] Confirm actual PatientID, StudyID, ImageID and StudyDate_DICOM columns.
- [x] Read patient and study identifiers as strings, preserving leading zeros.
- [x] Report missing IDs, duplicate images and images mapped to multiple patients.
- [x] Validate date parsing and missing dates.
- [ ] Verify the date de-identification policy and explain the 68 rows before 2009.
- [ ] Confirm explicitly whether dates are comparable between patients; within-patient ordering is insufficient for a global recency ranking.
- [x] Summarize image/study counts per patient (minimum, median and maximum).
- [x] Count raw label strings and physician versus automatic label provenance.
- [ ] Normalize label whitespace, handle empty strings and report eligible-cohort prevalence.
- [ ] Obtain complete eligible image sets for selected patients across archives.
- [ ] Define eligible views, image quality filters and label handling before counting training images.
- [ ] Split by patient and check that all images and studies of each patient stay within one split.

## Score decision

Proposed score: a weighted combination of acquisition recency and normalized log image count. Tune weights and recency timescale on validation only. Image volume represents workload, not proven withdrawal probability. Acquisition date is not the date of research contribution.

If date comparability is unverified, omit real recency and evaluate volume-only placement. Any synthetic timing or withdrawal scenarios must be labelled as simulated. Consent-expiry and real withdrawal-request labels have not been established.

## Audit deliverable

A reviewed aggregate Markdown report containing dataset version, inclusion rules, usable columns, missingness, patient/image counts, date-comparability evidence, class coverage and a go/no-go decision for recency. Do not commit patient-level rows or screenshots.
