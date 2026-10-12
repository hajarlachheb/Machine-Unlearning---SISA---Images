# PadChest discovery checklist

Status: planned; the actual release metadata has not yet been audited.

## Access and provenance

- [ ] Record official access approval, usage terms and release/version identifier.
- [ ] Obtain the metadata dictionary and CSV before the full image collection.
- [ ] Record checksums locally; keep raw metadata, identifiers and image paths out of Git.

## Metadata audit

- [ ] Confirm actual columns against documented PatientID, StudyID, ImageID and StudyDate_DICOM.
- [ ] Read patient and study identifiers as strings, preserving leading zeros.
- [ ] Report missing IDs, duplicate images and images mapped to multiple patients.
- [ ] Validate date parsing, missing dates and the date de-identification policy.
- [ ] Confirm explicitly whether dates are comparable between patients; within-patient ordering is insufficient for a global recency ranking.
- [ ] Summarize image/study counts per patient using aggregate quantiles.
- [ ] Summarize label prevalence and physician versus automatic label provenance.
- [ ] Define eligible views, image quality filters and label handling before counting training images.
- [ ] Split by patient and check that all images and studies of each patient stay within one split.

## Score decision

Proposed score: a weighted combination of acquisition recency and normalized log image count. Tune weights and recency timescale on validation only. Image volume represents workload, not proven withdrawal probability. Acquisition date is not the date of research contribution.

If date comparability is unverified, omit real recency and evaluate volume-only placement. Any synthetic timing or withdrawal scenarios must be labelled as simulated. Consent-expiry and real withdrawal-request labels have not been established.

## Audit deliverable

A reviewed aggregate Markdown report containing dataset version, inclusion rules, usable columns, missingness, patient/image counts, date-comparability evidence, class coverage and a go/no-go decision for recency. Do not commit patient-level rows or screenshots.
