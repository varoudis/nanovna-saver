# PERF-04 — Fused calibration/delay processing: measurement and decision

**Decision: keep the bulk path that is already in production. Do not pursue a
further "fused" prototype.** The gain it would chase has already been
collected, and the remaining cost is dominated by something fusion cannot
remove.

## What was executed

| Item | Value |
| --- | --- |
| Path | `Calibration.correct11`/`correct21` (scalar, reference) vs `Calibration.correct11_bulk`/`correct21_bulk` (NumPy), reached through `SweepWorker._calibrate_bulk` |
| Dataset | `tests/data/test_2port_long.cal`, span 200 000 000 – 300 000 000 Hz, frequency-dependent load (resonant + delayed) and a lossy transmission path |
| Harness | `tools/benchmark_calibration.py`, 7 repeats per measurement, median reported, spread printed separately |
| Environment | macOS arm64, CPython 3.12.9, NumPy 2.2.2 |
| Error tolerance | max abs difference vs the scalar reference, asserted before timing |
| Scope | correction maths only; conversion timed apart; rendering excluded (see PERF-03) |

## Result

| points | scalar S11 | bulk S11 | speedup | scalar S21 | bulk S21 | speedup | max │Δ│ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 1.75 ms | 0.02 ms | 71× | 2.88 ms | 0.04 ms | 82× | 0 |
| 401 | 7.00 ms | 0.03 ms | 203× | 11.79 ms | 0.05 ms | 234× | 0 |
| 1001 | 18.09 ms | 0.06 ms | 316× | 30.04 ms | 0.09 ms | 343× | 0 |
| 4001 | 70.72 ms | 0.14 ms | 508× | 118.57 ms | 0.25 ms | 482× | 0 |
| 16001 | 284.55 ms | 0.51 ms | 555× | 471.09 ms | 0.83 ms | 564× | 0 |

The difference is exactly zero at every size: on this dataset the two paths
agree bit-for-bit, not merely within tolerance. The ratio grows with point
count because the scalar path pays per-point CPython overhead while the bulk
path pays it once.

Two runs of the harness agree within a few percent (71× vs 74× at 101
points, 508× vs 503× at 4001), so the conclusion does not depend on one
lucky timings table.

## Conversion overhead, measured separately

| points | to arrays | from arrays | spread |
| --- | --- | --- | --- |
| 401 | 0.06 ms | 0.07 ms | 0.07–0.08 ms |
| 4001 | 0.52 ms | 0.79 ms | 0.77–0.92 ms |

At 4001 points the round trip costs ~1.3 ms against 0.39 ms of bulk kernel
work, so **conversion is now ~77 % of the correction cost**. Any further
fusion of the maths is chasing the remaining 23 %; the honest next step, if
the cost ever matters, is to remove the conversion (keep whole sweeps as
arrays end to end), not to fuse more arithmetic. That is a wider refactor of
`SweepWorker`'s publishing contract and is deliberately not undertaken here.

Rendering dominates everything: PERF-03 measured ~259 ms to repaint the 27
charts at 4001 points against ~10 ms of model work per `dataUpdated`. The
whole correction step is under 2 ms at that size.

## Why keeping the current implementation satisfies the task

The task says adopting fusion is conditional and "keeping the current
implementation is a valid outcome if the gain is immaterial". Two separate
facts settle it:

1. **The fusion is already adopted.** `SweepWorker` corrects through
   `correct11_bulk`/`correct21_bulk` over whole sweeps; the scalar methods
   remain only as the reference implementation the tests compare against.
   There is no unfused production path left to replace.
2. **The remaining headroom is in conversion and rendering, not in the
   maths.** A further fused kernel would move a 0.39 ms term at 4001 points.

## Invariants the bulk path keeps (and which a further fusion would risk)

Each is covered by an existing test in `tests/test_calibration_bulk.py`
(65 tests) — these are the checks PERF-04 asks for, already passing:

- **Frozen calibration state.** `test_in_place_reload_and_recalculate_cannot_change_a_run`,
  `test_in_place_reload_mid_sweep_keeps_published_identity`,
  `test_capture_does_not_alias_the_live_calibration`,
  `test_freezing_a_calibration_preserves_its_numbers`,
  `test_a_later_run_uses_the_newly_selected_calibration`. A run captures its
  calibration; reloading or recalculating mid-sweep cannot alter what a
  queued completion publishes.
- **Channel independence.** `test_apply_calibration_uses_raw_s11_for_s21` —
  the S21 correction consumes the *raw* S11, never the corrected one, so the
  order of the two corrections cannot matter.
- **Cancellation and failure.** `test_non_finite_delay_fails_without_publishing`,
  `test_overflowing_delay_does_not_publish_a_measurement`,
  `test_delay_that_overflows_the_phase_term_is_rejected` — bad values fail
  loudly and publish nothing rather than writing a half-corrected sweep.
- **Delay-only provenance.** `test_delay_only_sweep_publishes_the_applied_delay`,
  `test_no_calibration_and_no_delay_reports_no_correction`,
  `test_calibration_and_delay_report_both`.
- **Singular and invalid values.** Mismatched channels, shifted grids,
  non-finite or fractional frequencies, missing raw S11 for a two-port
  calibration — all refused by name rather than broadcast.
- **Small and large datasets.** 0, 1, 101 … 16001 points.

## Reproduce

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python tools/benchmark_calibration.py
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_calibration_bulk.py -q
```
