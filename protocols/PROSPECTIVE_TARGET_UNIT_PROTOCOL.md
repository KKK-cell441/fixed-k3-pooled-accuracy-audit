# Prospective Target-Unit Qualification Protocol

Status: locked external audit completed on PHM2012; prospective acquisition
still not available.

## Why this is blocked

The local PHM2012/PRONOSTIA data provide 17 complete bearing lives and were
used as a locked external run-to-failure audit after the protocol was frozen.
That audit is still retrospective because the data existed before the protocol
was written. It is also a different endpoint from the five-class HUSTbearing
fault-type task.

A true prospective acquisition is still unavailable. Reusing HUSTbearing,
HSE, HUST-Hanoi, SCA, or Mehran-UET would be retrospective relabelling, not
prospective validation.

## Locked protocol for the future run

- Candidate family: pre-specified before data inspection.
- Validation unit: physical bearing or operating-speed unit, declared before
  model training.
- Materiality margin: `delta = 1 pp`, unless an independent engineering cost
  model replaces it.
- Harm margin: `eta = 5 pp`, unless an independent maintenance or safety policy
  replaces it.
- Class intervals: simultaneous at the declared family-wise confidence level.
- Minimum class support: six independent validation units per required class.
- Candidate selection: no candidate may be added after the target-unit outcomes
  are inspected.
- Evaluation: one locked final run; all candidate outcomes and all class
  outcomes are reported, including failures.

## Acceptance rule

Qualified is allowed only when:

1. the unit-level lower confidence bound exceeds `delta`;
2. every required class interval is available and non-fragile;
3. every class lower bound is at least `-eta`;
4. the candidate, margins, and support rule were registered before evaluation.

Otherwise the result is Not qualified or Insufficient according to the
published three-state rule.

## Required data

The target block must provide:

- independent bearing or target-unit identifiers;
- at least six independent units for each required class;
- enough recordings per unit to estimate class-level recall;
- a declared operating condition and noise protocol;
- raw waveforms and licence/access terms.

Until such a block is available, the paper must state that the positive
Qualified branch has not been validated on a real prospective target unit.
