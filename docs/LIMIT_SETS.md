# Measurement limit sets

NanoVNA-Saver can check a measurement against explicit pass/fail limits and
freeze the result into a report. A limit set is a small, versioned JSON
document. Change it in any text editor; the GUI only loads, selects and
evaluates it.

An example is checked in as `tests/data/example.nvlimits`:

```json
{
  "format": "nanovna-saver-limits",
  "version": 1,
  "name": "Two metre band amateur radio",
  "description": "144-146 MHz example.",
  "rules": [
    {
      "id": "match",
      "label": "Antenna match",
      "metric": "s11_vswr",
      "segments": [
        { "start_hz": 144000000, "stop_hz": 145000000, "maximum": 2.0 },
        { "start_hz": 145000000, "stop_hz": 146000000, "maximum": 1.5 }
      ]
    },
    {
      "id": "path",
      "metric": "s21_insertion_loss_db",
      "segments": [
        { "start_hz": 144000000, "stop_hz": 146000000, "maximum": 3.0 }
      ]
    }
  ]
}
```

## Document

| key | required | meaning |
| --- | --- | --- |
| `format` | yes | Always `nanovna-saver-limits`. |
| `version` | yes | Integer format version. This build reads version `1` and rejects newer documents with an explicit message. |
| `name` | yes | Non-empty name shown in the GUI and in reports. Loading a set with an existing name replaces it. |
| `description` | no | Free text. |
| `rules` | yes | Non-empty list of rules. |

Unknown keys are rejected: a typo such as `maximum_db` fails loudly instead of
being silently ignored.

## Rules

| key | required | meaning |
| --- | --- | --- |
| `id` | recommended | Short identifier, unique inside the document. Missing ids are numbered `rule-1`, `rule-2`, ... |
| `label` | no | Human-readable name. Defaults to the metric description. |
| `metric` | yes | One of the metrics below. |
| `segments` | yes | Non-empty list of frequency segments. |

### Metrics, units and sign

| metric | channel | unit | limits |
| --- | --- | --- | --- |
| `s11_vswr` | S11 | none (ratio) | `minimum` / `maximum`, normally `maximum` |
| `s11_return_loss_db` | S11 | dB | **positive** for a match: `-20*log10|S11|` |
| `s21_insertion_loss_db` | S21 | dB | **positive** for loss: `-20*log10|S21|` |

Consequences of that sign convention, which are deliberate:

* A gain has a negative insertion loss, so `minimum: 0` rejects a gain while
  `maximum: 0` accepts it. Rejecting gain therefore needs an explicit
  `minimum` (for example `minimum: 0`), not a `maximum`.
* A reflection coefficient above 1 (an active or mis-referenced
  measurement) has a negative return loss, so `minimum: 10` rejects it.
* A total reflection has infinite VSWR, which violates every finite
  `maximum`; it is a valid measurement, not missing data.

There is no unit field. The unit belongs to the metric, so `"unit": "dB"` is
rejected as an unknown key rather than being double-applied.

### Segments and piecewise bounds

Each segment is `{"start_hz": <int>, "stop_hz": <int>, "minimum": <number|null>, "maximum": <number|null>}`.

* Frequencies are whole hertz. Decimal hertz and negative values are
  rejected.
* `stop_hz` must be greater than `start_hz`; a reversed range is an error.
* At least one of `minimum` / `maximum` is required. Both may be given.
* A segment whose `minimum` exceeds its `maximum` is rejected (conflicting
  bounds).
* Values must be finite numbers: `NaN`, `Infinity` and strings are rejected.
* Segments of one rule must tile the band: sorted by start, each segment must
  begin exactly where the previous one ended. Gaps and overlaps are errors.
* Each segment covers `[start_hz, stop_hz)` — both ends for the last
  segment. A sample on a shared boundary therefore belongs to the later
  segment, and every sample is tested exactly once.
* Two rules for the same metric must not have overlapping bands. Use one rule
  with several segments instead. Rules for different metrics are independent.

## Evaluation policy

The check runs on an immutable measurement snapshot; nothing is re-read from
the display or the device afterwards.

* **Bounds are inclusive.** A measured value equal to `maximum` (or
  `minimum`) passes. Only a value strictly beyond the bound fails.
* **Measured samples only.** No value is interpolated or extrapolated between
  samples.
* **Band coverage is required**, and it is checked in two tiers.
  1. **Declared grid (authoritative).** When the snapshot records the
     acquisition settings (`provenance.sweep`), that `SweepInfo` states which
     frequencies were requested. The capture must then contain exactly one
     sample per declared frequency inside the band, in order. A missing,
     clustered-missing, duplicated or off-grid sample makes the rule
     `UNAVAILABLE` with the count and first offending frequency in the reason.
    Device rounding is tolerated (at most 2 ppm, at most half a sample step),
    and a repeated frequency that the declared grid itself contains (a
    multi-segment join) is expected, not an error. Ownership of a sample is
    exclusive: tolerance windows can meet at exactly half a sample step, and a
    sample on that boundary is equally consistent with two different declared
    frequencies, so which declared sample is missing cannot be decided. That
    is reported as `UNAVAILABLE` (ambiguous) rather than credited to whichever
    neighbour it happens to match first, because that credit would let one
    sample stand in for two.
  2. **Data-only fallback.** When no acquisition settings were recorded —
     imported Touchstone, a snapshot built without a `Sweep`, older sessions —
     nothing declares the intended resolution, so coverage is judged from the
     data: every interval between consecutive samples (and between a band edge
     and the outermost sample) is compared with the spacing observed *around
     it*, and an interval more than 1.5 times its local spacing cannot be
     decided. A logarithmic sweep or a segment join is normal data here too.

  The fallback is deliberately conservative about what it can conclude: a
  cluster of missing samples whose surviving gaps each justify their
  neighbour is indistinguishable, from the retained samples alone, from a
  legitimately coarser capture, so it is accepted. No resolution is invented
  to pretend otherwise; such a capture is only detected when its acquisition
  settings were recorded with it.
* **Data must be complete.** A snapshot recorded as `partial` or
  `interrupted` is never reported as a pass.
* **Invalid samples block the check.** A non-finite (`NaN`/`inf`) complex
  value inside the band makes the rule `UNAVAILABLE`; it is not skipped.
* **One outcome per rule and trace:** `PASS`, `FAIL` or `UNAVAILABLE`, with
  the worst value, its frequency and the bound it violated.
* An overall report verdict is `FAIL` if any check failed, otherwise
  `UNAVAILABLE` if any check could not be decided, otherwise `PASS`.
  `UNAVAILABLE` is never presented as a pass.

## GUI workflow

1. **Limits ...** in the main window button grid opens the limit window.
2. **Load ...** reads a `.nvlimits` file. A rejected file is reported
   verbatim and *leaves previously loaded sets untouched*.
3. **Use for reports** elects the active set; **No limits** clears it.
4. **Check captured traces** evaluates the active set against every captured
   trace and lists each per-rule/per-trace verdict.
5. **Report ...** offers a limit selector. **Refresh** freezes the selected
   set together with its results into the report model.

A frozen report keeps its own copy of the limit set and of every result. The
preview, the exported HTML file and the exported CSV file all describe that
same frozen selection; changing or removing the active limit set afterwards,
or capturing new measurements, cannot alter an already built report. Build a
new report (**Refresh**) to apply a different selection.

Reports built without limits are unchanged, and `build_report` /
`write_html` / `write_csv` keep working without the new argument.

## What an export contains

Every limit row carries the frozen verdict, the worst value, its frequency and
the bound it violated. A PDF is not produced (see `Reports.py`).

* **HTML** renders `inf` as `infinite` and `-inf` as `-infinite` in the worst
  value and bound columns.
* **CSV** writes the worst value with the same portable markers the session
  and snapshot encoders use: `inf` and `-inf`. A total reflection therefore
  exports as VSWR `inf`, never as an empty cell. Only a value that does not
  exist — an `UNAVAILABLE` check, which has no worst sample — is left blank,
  and the `reason` column then explains why.
* `report_snapshot()` embeds the same `inf` / `-inf` strings, so the plain
  data view, the HTML and the CSV always agree on the frozen value,
  frequency and verdict.