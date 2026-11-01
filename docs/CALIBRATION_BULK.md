# Bulk calibration correction

Status: implemented by `src/NanoVNASaver/Calibration.py` and
`src/NanoVNASaver/SweepWorker.py`.

## Why

`SweepWorker.applyCalibration` used to call `Calibration.correct11` and
`Calibration.correct21` once per datapoint. Each of those calls performs three
or five separate `scipy.interpolate.interp1d` evaluations, so a 1 000-point
sweep paid roughly 5 000 scalar SciPy calls — twice that for a two-port
calibration and again for every averaging pass. The correction arithmetic is
trivial next to the per-call overhead.

## What changed

`Calibration` gained an additive bulk API. No RF equation changed, no file
format changed, and the scalar methods remain in place as the reference
implementation:

| Bulk entry point | Scalar reference |
| --- | --- |
| `Calibration.correct11_bulk(freq, s11)` | `Calibration.correct11(dp)` |
| `Calibration.correct21_bulk(freq, s21, s11)` | `Calibration.correct21(dp, dp11)` |
| `correct_delay_bulk(freq, values, delay, reflect)` | `correct_delay(dp, delay, reflect)` |
| `datapoints_to_arrays(points)` / `arrays_to_datapoints(freq, values)` | — (worker glue) |

Each bulk method evaluates every `interp1d` **once** with the whole frequency
array. SciPy then does the segment search in C and applies the same endpoint
`fill_value`, so frequencies outside the calibrated span are still held at the
first/last calibration sample. The corrected terms are then combined with the
identical expressions used by the scalar code:

```
S11 = (s11 - e00) / (s11 * e11 - delta_e)
S21 = (s21 - e30) / e10e32 * e10e01 / (e11 * s11_raw - delta_e)
```

`correct21_bulk` takes the **raw** port-1 reflection, exactly like the scalar
`correct21`; the worker passes the raw array, never its own corrected output.

`SweepWorker.applyCalibration` now converts the segment once
(`datapoints_to_arrays`), corrects both channels, applies the offset delay in
bulk, and converts back (`arrays_to_datapoints`). The integer hertz grid is
carried through unchanged, so published frequencies are exactly the measured
ones.

## Preserved behaviour

* **Equations and calibration-kit models** — unchanged.
* **Interpolation and edge holding** — the same `interp1d` objects with the
  same `slinear` kind and endpoint `fill_value`; results outside the span
  match the scalar oracle to rounding.
* **Raw S11 in the transmission correction** — preserved (see above).
* **Ordering** — the output grid follows the input order, including
  unsorted or interleaved frequencies.
* **One-port / two-port availability** — one-port calibrations correct S11
  only and pass S21 through unchanged; without a calculated calibration both
  channels are returned as raw copies. When only S11 was measured, S11 is
  corrected and S21 stays empty; S21 without raw S11 is rejected for a
  two-port calibration and passed through otherwise.
* **Offset delay** — the captured `SweepCapture.offset_delay` still wins over
  the live `worker.offsetDelay`, and S11 keeps the doubled phase factor
  (`reflect=True`). The delay is a correction in its own right: it is applied
  with or without a calculated SOL calibration, and the snapshot reports
  `correction.applied = True`, the applied `offset_delay`, and an empty
  `correction.identity` (there is no calibration to identify). Raw samples
  are never relabelled as corrected.
* **Captured calibration isolation** — `SweepCapture.calibration` is a frozen
  copy (`Calibration.frozen_copy()`) taken at the run boundary, not a
  reference to the live object. Reloading *or* recalculating the live
  calibration in place during a sweep therefore cannot change that run's
  coefficients, validity or published identity, and a later run picks up the
  newly selected calibration normally.
* **Raw/corrected provenance** — unchanged; `rawData11`/`rawData21` and the
  snapshot channels keep the measured values. Report labels and the HTML
  correction line distinguish a calibration correction from an
  offset-delay-only one.

## Numerical equivalence

The bulk path is numerically equal to the scalar path, not bit-identical:
NumPy's complex division may differ from CPython's in the last bit, and
`numpy.exp` may differ from `cmath.exp` in the last bit. Measured maximum
absolute differences on the benchmark below: <= 7e-17 for S11, <= 3e-14 for
S21 (values are O(1)). `tests/test_calibration_bulk.py` pins a 1e-12
tolerance against the scalar oracle.

## Invalid input

Previously an invalid calibration could produce `inf`/`NaN` and be published
as if corrected. The bulk path refuses instead of hiding it:

* mismatched channel lengths, non-1-D input, an empty grid, or an
  uncalculated calibration raise `ValueError` (never silent broadcasting);
* S11/S21 frequency grids that differ raise `ValueError` in the worker
  (equal lengths are not enough: the S21 correction consumes the matching
  raw S11 sample);
* a non-finite corrected sample raises `ValueError` naming the first bad
  frequency;
* a non-finite offset delay (`inf`, `NaN`) raises at capture — the run fails
  with an actionable message instead of publishing a COMPLETE measurement
  full of `NaN` — and a finite delay whose phase term overflows is rejected
  the same way;
* frequency conversion rejects non-finite, fractional or out-of-`int64`
  values instead of casting them to an arbitrary integer (`inf` used to
  become `INT64_MAX`).

Large-but-finite corrections (near-singular data) are passed through exactly
as the scalar path produces them.

Invalid-input behaviour is deliberately stricter here than in the unscaled
scalar helpers (`correct_delay(dp, inf)` still returns `nan` samples): the
bulk path refuses to publish them. Valid finite input keeps exact scalar
parity.

## Benchmark

`tools/benchmarks/benchmark_calibration_bulk.py` (standalone, deterministic
calibration and sweep, 2 warmup + 7 repeats, medians reported). Run:

```
.venv/bin/python tools/benchmarks/benchmark_calibration_bulk.py
.venv/bin/python tools/benchmarks/benchmark_calibration_bulk.py --points 1000 10000 100000 --repeats 9
```

The benchmark measures four paths. Only the last one is production:

* `scalar` — the original per-datapoint loop; the reference oracle.
* `bulk arrays` — the bulk helpers on pre-built NumPy arrays, arithmetic only.
* `fused prototype` — an arrays-only prototype that converts once in and
  once out because it keeps the calibration and the offset delay on the same
  arrays. The shipping worker does **not** do that, so this column is a
  lower bound on conversion cost, not a shipping figure.
* `worker` — `SweepWorker.applyCalibration` itself, the code that runs per
  sweep segment. It converts to arrays, corrects both channels, converts back
  to `Datapoint`, then converts again per channel to apply the offset delay
  and back. This is the column to quote.

Measured on this machine (macOS arm64, Python 3.13.9, NumPy 2.2.2, two-port
calibration of 900 points over 1–900 MHz, 33 ps offset delay), `--repeats 3`.
Two consecutive runs gave 34.0–34.6x at 1 000 points and 37.5–38.9x at
10 000; one run is tabulated:

| points | scalar | bulk (arrays) | fused prototype | **actual worker** | speedup (worker) | to-array | to-points | max abs dS11 | max abs dS21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 000 | 45.4 ms | 0.1 ms | 0.7 ms | **1.3 ms** | 34.6x | 0.11 ms | 0.13 ms | 1.4e-17 | 0 |
| 10 000 | 443.4 ms | 0.9 ms | 6.2 ms | **11.8 ms** | 37.5x | 1.06 ms | 1.46 ms | 2.8e-17 | 2.2e-16 |

The maximum absolute difference columns refer to the fused prototype, which
is numerically identical to the worker path (a `repeats 3` run at both sizes
showed the same deltas for each path against the scalar oracle).

The `Datapoint` <-> array conversion dominates the worker path. With an offset
delay active the worker converts each channel to arrays twice and back twice:
once around `correct11_bulk`/`correct21_bulk`, and once more per channel
around `correct_delay_bulk`. That is why it costs roughly five times the
fused prototype instead of near it. The realistic speedup over the scalar
loop is therefore ~35–40x at the sizes a NanoVNA produces, not the ~75x the
prototype shows. Real sweeps are 101–100 000 points per segment.

Each conversion microbenchmark measures **one channel**: one `Datapoint` list
to arrays, one arrays-to-`Datapoint` list. A sweep with an offset delay pays
both, for both channels, twice — four times each benchmark's figure. Without
a calibration (raw samples) the worker skips the calibration conversion and
pays only the delay pass; with no offset delay it pays only the calibration
pass.

Capture overhead is measured separately and is **not** included in any entry
above: freezing a `MeasurementSnapshot` with `Calibration.frozen_copy()` costs
about 0.45 ms per sweep (identity digest about 0.40 ms), which is negligible
next to even the 1 000-point worker figure but is paid by every capture.

These numbers are a measurement on this host, not a guarantee. No unit test
depends on them.

## Limitations

* The bulk path needs NumPy arrays; callers holding `Datapoint` lists pay the
  conversion cost shown above.
* `correct11_bulk`/`correct21_bulk` require a calculated calibration; they
  raise rather than lazily interpolating like `Touchstone` does.
* No multi-threading, no float32 fast path, no changes to uncertainty,
  measurement capability, or calibration-kit models.