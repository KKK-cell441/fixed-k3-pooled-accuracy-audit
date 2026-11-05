# Pooled Accuracy Is Not Enough to Authorize Fixed-K3 VMD Preprocessing

Reproducibility artifact for same-source and external bearing audits.

## Main findings

- On HUSTbearing, a clean-trained speed-level gain of +9.72 percentage points
  reverses to -7.94 points after matched 0 dB noise augmentation. The
  unit-only interval is [-12.61, -3.44] points.
- The matched HUSTbearing bearing-level effect is +0.49 points [-0.62, 1.53]
  and is Insufficient, not a neutral or authorized result.
- On the complete PHM2012/PRONOSTIA data, 17 independent bearings are used for
  a healthy/degraded run-to-failure audit. Clean-trained K3 has a mean effect
  of +24.51 points, but the healthy class changes by +70.75 points and the
  degraded class changes by -21.73 points [-43.14, -5.39]. The block is Not
  qualified because of class harm.
- No empirical block in this study receives a full Qualified status.

The three-state audit is a decision aid, not a validated certification
framework. The contribution is the empirical demonstration that pooled
accuracy can reverse under matched measurement conditions and conceal
class-specific harm.

## Contents

- `code/`: candidate audit, PHM2012 external audit, figure generation, and
  utility code.
- `results/`: candidate and PHM2012 results, run records, and the consolidated
  reachability audit.
- `protocols/`: frozen protocols.
- `tables/`: manuscript LaTeX tables.
- `figures/`: generated PDF figures.
- `manuscript/`: manuscript source, bibliography, compiled PDF, and manifest.

## Reproduction

Install the Python requirements:

```text
pip install -r requirements.txt
```

For the candidate audit, the HUSTbearing cache and external result files used by
the manuscript must be placed in the working directory expected by the scripts.
For PHM2012, the complete learning and full-test bearing lives must be placed at
the source path configured in `code/phm2012_external_qualification.py`.

Run:

```text
python code/hust_candidate_qualification_audit.py
python code/phm2012_external_qualification.py
python code/make_final_submission_figures.py
```

The source datasets retain their original licences and are not redistributed
in this repository. This repository contains derived results, frozen protocols,
and analysis code.

## Claim boundary

The PHM2012 result is a retrospective locked external audit with a binary
run-to-failure endpoint and one fixed noise realization. It is not a prospective
field study and not a reproduction of the five-class HUSTbearing fault-type
endpoint.
