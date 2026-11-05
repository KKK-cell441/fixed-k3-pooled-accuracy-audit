# PHM2012 Locked External Qualification Protocol

Registration: `PHM2012-EXTERNAL-QUALIFICATION-20261105`

Status: frozen before data extraction.

## Source

The source is the complete IEEE PHM 2012 / PRONOSTIA bearing lives already
stored at:

`D:/EvidenceAware_20261011/data/kaggle_phm2012/ieee-phm-2012-data-challenge-dataset-master`

The 17 complete bearing lives are taken from `Learning_set` and
`Full_Test_Set`. The truncated challenge `Test_set` is not used.

## Endpoint

This is a separate run-to-failure endpoint, not a reproduction of the
five-class HUSTbearing fault-type endpoint.

- healthy: first 20 percent of each bearing life;
- degraded: last 20 percent of each bearing life;
- middle 60 percent: excluded;
- 12 acquisitions per class per bearing, evenly spaced inside each interval.

The target unit is the physical bearing. Validation is leave-one-bearing-out.
Repeated acquisitions and windows never count as independent units.

## Signal and noise

The horizontal acceleration column is used. The first 2,048 samples of each
acquisition are normalized by their own mean and standard deviation.

Two protocols are evaluated:

- clean-trained: clean training windows, 0 dB noisy test;
- matched-augmentation: clean plus a fixed noisy training realization, 0 dB
  noisy test.

Raw and K3 share the same test and training noise realizations. The primary
candidate is fixed K3; no candidate is selected from the result.

## Qualification

The same decision rule is used with:

- `delta = 1 pp`;
- `eta = 5 pp`;
- Bonferroni-adjusted simultaneous class intervals over two classes;
- 20,000 unit-bootstrap draws;
- minimum non-fragile support of six independent bearings per class.

Qualified requires a positive unit lower bound above `delta`, both class
intervals available and non-fragile, and every class lower bound at least
`-eta`.

## Claim boundary

Even a Qualified result would establish only that fixed K3 is authorized for
this run-to-failure endpoint and this set of bearing units. It would not
establish a five-class fault-type result, a prospective field deployment, or a
universal VMD rule.
