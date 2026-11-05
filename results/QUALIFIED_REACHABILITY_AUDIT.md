# Real Qualified Reachability Audit

Date: 2026-11-05

Decision margins fixed before the candidate audit:

- `delta = 1 pp`;
- `eta = 5 pp`;
- simultaneous class confidence: 99% family-wise
  (Bonferroni over five classes);
- independent inference unit: held-out motor speed or physical bearing,
  never a window, seed, or noise realization;
- minimum non-fragile class support: six independent validation units.

## 1. HUSTbearing candidate audit

Candidate set fixed before inspecting results: K2, K3, K4, K5.

Protocols:

- clean-trained raw versus clean-trained K candidate, tested at 0 dB;
- matched noise augmentation on both raw and K candidate, tested at 0 dB;
- 10 held-out speeds, 3 training seeds, 20 fixed epochs;
- unit interval from 20,000 bootstrap draws over speeds;
- simultaneous class intervals from 20,000 bootstrap draws with Bonferroni
  adjustment over five classes.

| Protocol | K | Mean effect (pp) | Unit 95% CI | Most negative class interval (pp) | Status |
|---|---:|---:|---|---:|---|
| Clean-trained | 2 | +17.67 | [11.89, 23.61] | outer: [-70.00, -45.56] | Not qualified (class harm) |
| Clean-trained | 3 | +9.72 | [6.78, 13.06] | outer: [-56.39, -13.33] | Not qualified (class harm) |
| Clean-trained | 4 | +4.50 | [0.89, 8.17] | outer: [-41.67, -6.11] | Not qualified (class harm) |
| Clean-trained | 5 | +4.72 | [2.56, 6.78] | outer: [-11.94, 6.39] | Insufficient |
| Matched augmentation | 2 | -10.50 | [-14.11, -6.67] | normal: [-25.56, -3.33] | Not qualified (mean) |
| Matched augmentation | 3 | -7.94 | [-12.56, -3.56] | rolling element: [-38.89, -8.06] | Not qualified (mean) |
| Matched augmentation | 4 | -5.61 | [-8.33, -3.11] | normal: [-33.89, 2.22] | Not qualified (mean) |
| Matched augmentation | 5 | -3.78 | [-6.11, -1.39] | normal: [-18.89, 4.44] | Not qualified (mean) |

All five classes have positive support in all ten speed units under both
protocols. The absence of Qualified is therefore not caused by fragile or
missing class evidence.

The closest candidate is clean-trained K5. Its mean effect is positive and its
outer-class interval does not establish harm below `-5 pp`, but the outer-class
lower bound is `-11.94 pp`. The class-safety condition therefore fails, and the
rule correctly returns Insufficient rather than Qualified.

## 2. HSE-SimilarSystem feasibility

HSE has 11 independent physical bearings, but the class support is:

- inner: 3 bearings;
- normal: 3 bearings;
- outer: 2 bearings;
- rolling element: 3 bearings.

The minimum non-fragile threshold is six independent bearings per class.
Additional seeds or additional noise realizations cannot increase this support
because they are repeated measurements inside the same bearing units.

Consequently, HSE cannot produce a full Qualified decision under the current
class-support rule without adding independent bearings. A multi-noise rerun
would quantify mean-effect stability, but it cannot remove the structural
fragility of the class audit. No such rerun is justified as a route to
Qualified.

HUST-Hanoi has 33 bearings, but its maximum class support is five bearings.
SCA has sufficient support for two binary classes but a negative mean effect.
Mehran-UET has six bearings per class but established inner-class harm.

## 3. Locked PHM2012 external audit

The user's local PHM2012/PRONOSTIA data contain 17 complete bearing lives
(`Learning_set` plus `Full_Test_Set`). This block was not used in the earlier
HUSTbearing or HSE analyses.

A protocol was frozen before extraction:

- healthy: first 20 percent of each bearing life;
- degraded: last 20 percent of each bearing life;
- middle 60 percent excluded;
- 12 acquisitions per class per bearing, evenly spaced;
- one horizontal acceleration channel, 2,048-sample windows;
- leave-one-bearing-out;
- three seeds and 20 epochs;
- fixed K3 versus raw;
- `delta = 1 pp`, `eta = 5 pp`, 99% simultaneous class intervals;
- 17 independent bearings per class, so the class evidence is non-fragile.

Results:

| Protocol | Healthy class change | Degraded class change | Mean effect | Unit 95% CI | Status |
|---|---:|---:|---:|---|---|
| Clean-trained | +70.75 pp [57.52, 83.66] | -21.73 pp [-43.14, -5.39] | +24.51 pp | [19.20, 29.74] | Not qualified (class harm) |
| Matched augmentation | -8.17 pp [-18.63, -0.82] | -4.41 pp [-17.81, 5.07] | -6.29 pp | [-12.17, -0.98] | Not qualified (mean) |

The clean-trained result is a particularly clear example of why a pooled
gain cannot authorize preprocessing: the mean improves by 24.51 pp, but the
degraded class interval is entirely below zero and below the `-5 pp` harm
margin. The matched protocol removes the apparent gain and returns a negative
mean effect.

At the individual-unit level, clean-trained K3 improved balanced accuracy in
all 17 bearings, with unit effects ranging from +4.17 to +41.67 pp. This is
not evidence of a safe preprocessing decision: the improvement is concentrated
in the healthy class while the degraded class is systematically sacrificed.
The example therefore separates "the pooled metric improves" from "the
qualification contract authorizes the intervention."

This is a locked external audit, not a prospective acquisition. It also uses a
different endpoint from the five-class HUSTbearing fault-type task. It can
strengthen the evidence that class safety and training-condition matching
matter, but it does not create a real Qualified case.

## 4. Prospective target-unit study

A genuine prospective study still requires an untouched labelled target unit,
a pre-specified candidate family, fixed margins, and at least six independent
units per required class. The PHM2012 run is retrospective because the data
already existed before the protocol was written. It must not be called
prospective.

## Conclusion

The current evidence does not contain a real Qualified case.

The reason is now explicit:

- HUSTbearing has non-fragile class support, but every candidate has either
  established class harm or unresolved class risk;
- HSE cannot qualify because its class support is structurally fragile;
- HUST-Hanoi, SCA, and Mehran-UET either have insufficient mean evidence or
  established class harm;
- PHM2012 has non-fragile support and was not used previously, but its
  clean-trained K3 audit shows clear degraded-class harm, and matched
  augmentation returns a negative mean effect.

The rule's Qualified branch remains logically reachable and is demonstrated in
the synthetic operating-characteristic study, but no empirical positive case
has yet been established.

## Artifacts

- `hust_candidate_qualification_audit.py`
- `hust_candidate_qualification_runs.csv`
- `hust_candidate_qualification_results.json`
- `phm2012_external_qualification.py`
- `phm2012_external_runs.csv`
- `phm2012_external_results.json`
