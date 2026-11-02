# Calibration age and verification

The **Calibration age and verification** box in the Calibration window answers
two questions that span coverage cannot: when were these standards measured,
and does one of them still read correctly now? Coverage, age and verification
are separate statements and stay on separate lines in the window and in
reports.

None of them is a measurement uncertainty. They never produce an error bar.

## What is recorded

Measuring a standard stamps the calibration with the current time in UTC.
Applying the calibration also records the connected device, and applying a
kit records the kit name and connector. The operator field is optional.

All of this is stored as one comment line in the `.cal` file:

```text
! cal-metadata: {"created": "2026-11-02T09:30:00Z", "device": "NanoVNA-H"}
```

Older versions of the application read that line as a note and otherwise
ignore it. The metadata does not affect the calibration's identity digest,
which covers only the measured data.

A loaded file keeps what it recorded. The device connected today is not
evidence of which device measured those standards, so it is never filled in.

## Age

| Shown | Meaning |
| --- | --- |
| `Age 3.2 h` | Time since the standards were measured. |
| `Age 3.2 h, past your 2 h reminder` | Older than the interval you chose. |
| `Age unknown` | No usable creation time: an older file, an unreadable metadata line, or a time with no timezone. |
| `Age unavailable: created after the current time` | A clock is wrong, so no age is stated. |

The reminder is off until you set **Remind after**. There is no fixed number
of hours after which a calibration expires: drift depends on temperature,
connectors and how much accuracy the measurement needs. A reminder is a prompt
you set up, and it does not change any data.

Each sweep's snapshot keeps its calibration's creation time. A report states
the age when that trace was swept, so reopening an old session does not make
its data look older than it was.

## Verifying a standard

1. Apply a calibration and set a single sweep (not continuous).
2. Connect a short, open or load, and choose it under **Connected standard**.
3. Set **Tolerance** and press **Sweep and verify**.

The application sweeps, corrects the *raw* S11 with the calibration under test,
and compares it with the model that calibration assumes for that standard:
ideal values, the kit's coefficients, or its `.s1p` file. The result is the
largest `|ΔΓ|` in linear reflection-coefficient units, where it occurred, and
whether it is within your tolerance.

- **Tolerance is your threshold.** The default of 0.05 is a starting point, not
  a property of any kit or instrument.
- **A pass is narrow.** It means this one standard was within the tolerance at
  this moment. It does not bound the error of other measurements. Re-measuring
  a standard used in the calibration mainly reveals drift, a poor connection or
  the wrong standard being connected.
- **Only the calibrated span is judged.** Points outside it are counted and
  reported, because their correction is held at an endpoint.
- **Results last only as long as the corrections.** Measuring a standard,
  recalculating or loading a calibration clears them, and they are not saved to
  files.
- **Simulated results are labelled** `[simulated, not hardware evidence]`.

## Hardware validation still required

The feature is tested against an analytic three-term error box and the
simulator only. On a real instrument, still to check:

1. A calibration followed immediately by verifying each standard reports small
   residuals on the bench's actual kit, as a basis for choosing a tolerance.
2. Connecting the wrong standard is reported as outside tolerance.
3. A loosened connector or a warmed-up instrument moves the residual visibly.
4. The recorded device name identifies the instrument usefully.
