# NanoVNA-Saver session format (`.nvsession`)

Version: 1  
Status: implemented by `src/NanoVNASaver/Session.py`.

## Purpose

A session is a self-contained, versioned record of an experiment. Reopening
one restores traces, calibration, markers, charts and analysis settings
without the original Touchstone files, calibration files, or a connected
device.

## Container

The file is a ZIP archive with the suffix `.nvsession`. Every member is
either JSON or UTF-8 text. `pickle`, `marshal`, `eval` and any other form of
executable deserialization are never used.

| Member | Required | Content |
| --- | --- | --- |
| `manifest.json` | yes | Session metadata and the list of trace members |
| `traces/<trace-id>.json` | per trace | One trace with its numeric data and provenance |
| `calibration/<name>` | optional | Calibration dataset text, byte-for-byte |

Only these three member kinds are accepted. Absolute paths, `..` segments and
any other member are rejected, so loading never writes files and never
extracts paths.

### Size limits

| Limit | Value |
| --- | --- |
| Single member | 64 MiB uncompressed |
| Whole archive | 512 MiB uncompressed |
| Traces | 256 |

## Manifest

```json
{
  "format": "nanovna-saver-session",
  "version": 1,
  "title": "EFHW tuning",
  "notes": "3 turns removed between runs",
  "created": "2026-11-01T10:00:00+00:00",
  "app_version": "0.6.5",
  "trace_files": ["traces/trace-0.json"],
  "selected_trace": "trace-0",
  "markers": [{"freq": 14150000, "name": "Resonance"}],
  "charts": {"chart_00": "S11 Smith Chart"},
  "sweep": {"start": 13000000, "end": 15000000, "points": 201},
  "analysis": {"vswr_limit": 1.5},
  "preset": {"name": "antenna", "settings": {}},
  "calibration": {"identity": "c675895c...", "one_port": true},
  "calibration_files": ["sol_27_30.cal"],
  "ui": {"splitter": "..."}
}
```

`format` must equal `nanovna-saver-session`. A `version` greater than the
supported version is refused with
`SessionVersionError` ("newer than supported … upgrade NanoVNASaver") rather
than being partially interpreted. Older versions load with documented
defaults for missing optional fields.

## Trace payload

```json
{
  "trace_id": "trace-0",
  "measurement_id": "9f2c...",
  "name": "Antenna 1",
  "timestamp": "2026-11-01T10:00:00+00:00",
  "reference_impedance": 50.0,
  "frequencies": [13000000, 13010000],
  "raw":       {"S11": [[13000000, 0.31, -0.44]]},
  "corrected": {"S11": [[13000000, 0.12, 0.03]]},
  "provenance": {
    "source": "nanovna",
    "simulated": false,
    "device": "NanoVNA-H4",
    "app_version": "0.6.5",
    "created": "2026-11-01T10:00:00+00:00",
    "sweep": {"start": 13000000, "end": 15000000, "points": 201,
              "segments": 1, "mode": "SINGLE", "averages": 1,
              "truncates": 0, "logarithmic": false},
    "calibration": {
      "identity": "c675895c...", "source": "sol_27_30.cal",
      "notes": "SOL Calibration", "one_port": true, "two_port": false,
      "coverage_spans": [[27000000, 30000000]],
      "correction_source": "application"
    },
    "completion": "complete",
    "notes": "",
    "simulator_scenario": null
  },
  "correction": {
    "applied": true, "identity": "c675895c...",
    "source": "application", "offset_delay": 0.0
  },
  "format_version": 1
}
```

### Channel semantics

* `raw` and `corrected` are objects keyed by S-parameter name from the set
  `S11`, `S21`, `S12`, `S22`.
* A channel is present only when it was measured or imported. An absent key
  means *unavailable*; it is never encoded as zeros. A measured zero is a
  real value and is stored as `0.0`.
* Both `raw` and `corrected` are kept so a session can show the difference
  and so provenance survives.
* `correction.applied` and `correction.identity` record whether calibration
  has already been applied. On load the corrected arrays are used as stored;
  calibration is **not** re-applied, which is what makes "apply twice"
  impossible.
* `provenance.completion` is `complete`, `partial` or `interrupted`.
* `provenance.simulated` plus `simulator_scenario` keep simulated data
  identifiable after save and reopen.

### Numbers

Finite values are JSON numbers written with Python's shortest round-trip
representation, so a value saved and reloaded compares exactly. Non-finite
values use strings: `"nan"`, `"inf"`, `"-inf"`. Consumers must accept both
forms.

### Timestamps

ISO 8601 with an explicit offset. A timestamp without an offset is
interpreted as UTC.

## Calibration files

Calibration datasets are embedded verbatim under `calibration/<name>` so a
session never depends on an external path. The original filename is retained
as the member name and in `provenance.calibration.source`.

## Minimal example

```json
{
  "format": "nanovna-saver-session",
  "version": 1,
  "trace_files": ["traces/trace-0.json"]
}
```

with `traces/trace-0.json`:

```json
{
  "trace_id": "trace-0",
  "name": "Trace 1",
  "timestamp": "2026-11-01T10:00:00+00:00",
  "reference_impedance": 50.0,
  "raw": {"S11": [[1000000, 0.1, 0.0]]},
  "corrected": {"S11": [[1000000, 0.1, 0.0]]},
  "provenance": {"completion": "complete"},
  "correction": {"applied": false},
  "format_version": 1
}
```

## Writing rules

1. Build the complete archive in memory first.
2. Write to a temporary file in the destination directory, `fsync` it, then
   `os.replace` onto the destination. Any failure removes the temporary file
   and leaves the previous save untouched.
3. Archive member timestamps are fixed so identical documents produce
   identical bytes.

## Reading rules

1. Read and validate the whole archive before the caller replaces any open
   experiment.
2. Reject: unreadable files, non-ZIP data, missing manifest, unknown format,
   newer version, missing trace or calibration members, malformed JSON,
   invalid trace payloads, and a dangling `selected_trace`.
3. Never extract members to disk.

## Evolution

* Adding an optional manifest or trace field is a backward-compatible change
  and does not bump `version`.
* Changing the meaning of an existing field, or removing one, requires a
  version bump and a documented migration.
* Readers accept unknown fields and ignore them.
* Writers always emit the current version.