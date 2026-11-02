# Automation

Three Qt-free entry points let a script or a production line use
NanoVNA-Saver without a display: a Python API, a command line, and a
versioned test sequence. None of them import Qt widgets, so they work on a
headless machine.

## Python API

```python
from NanoVNASaver.Acquisition import SweepRequest, open_device
from NanoVNASaver.Hardware.Simulated import make_simulated_vna

with open_device(make_simulated_vna("cable", seed=1)) as session:
    result = session.sweep(SweepRequest(
        start_hz=1_000_000, stop_hz=100_000_000, points=101, average=4,
    ))
    snapshot = result.snapshot
```

`open_device` claims the device. A second handle on the same device raises
`DeviceBusyError` rather than sharing a serial port, and `close()` (or the
`with` block) releases it. Every failure is a structured exception:
`DeviceError`, `DeviceBusyError`, `CancelledError`, `UnsupportedError`.

Cancellation is cooperative. Pass a `CancelToken`; the sweep checks it
between reads and publishes nothing when it is cancelled:

```python
from NanoVNASaver.Acquisition import CancelToken, CancelledError

token = CancelToken()
try:
    session.sweep(request, cancel=token)
except CancelledError:
    ...
```

A snapshot is an immutable `MeasurementSnapshot`: `raw` holds the
acquisition, `corrected` holds what was published, and `provenance` records
whether the device was simulated. A channel the device cannot provide is
*absent*, never a zero sample.

## Command line

```
python -m NanoVNASaver.cli --scenario cable --points 101 \
    --limits limits.nvlimits --json result.json
```

| option | meaning |
| --- | --- |
| `--scenario`, `--seed` | simulated device to sweep |
| `--input FILE` | check a Touchstone file instead of sweeping |
| `--start`, `--stop` | band, with `k`/`M`/`G` suffixes (`14.0M`) |
| `--points`, `--average` | sweep size and averaging |
| `--limits FILE` | evaluate a `.nvlimits` set |
| `--touchstone FILE` | write the measurement (`.s1p` or `.s2p` as the data allows) |
| `--json FILE`, `--stdout-format json` | machine-readable record |
| `--log FILE` | append one audit row per run (`.csv`, or `.jsonl`) |
| `--serial-number SN` | unit serial recorded in that row |
| `--overwrite` | replace an existing output (refused otherwise) |
| `--combined-text` | try one masked text scan for every channel (opt-in) |
| `--binary-scan` | validate the binary scan frame instead of trusting it (opt-in) |

Exit codes are the contract:

| code | meaning |
| --- | --- |
| 0 | every requested check passed |
| 1 | a limit check failed |
| 2 | invalid input (arguments, file, limit set, sequence) |
| 3 | a result was **unavailable** — not a pass |
| 4 | a device or transport error |
| 130 | interrupted; no artifact is left behind |

Only the channels the checks read are acquired, so a one-port device is asked
for S11 alone. A run that cannot evaluate a check exits 3, never 0. That includes
a sequence stopped by an unavailable step. An output that cannot be written is
exit 2, never exit 1.

## Read paths

Every driver is read the way it has always been read: one command per
channel. The two optimisation flags above are **off by default** and change
nothing unless you pass them. Each reports the transport it actually used in
the JSON record as `metadata.read_path`:

| `read_path` | meaning |
| --- | --- |
| `trusted` | the ordinary per-channel read |
| `combined` | one masked text `scan` reply carried both channels |
| `scan_bin` | one validated binary frame carried both channels |

A reply is accepted whole or not at all: a short, malformed, non-finite or
implausible optimised read is discarded and the whole sweep is re-read over
the trusted path, so a corrupted reply can never become a completed
measurement. Errors from the trusted re-read propagate (exit 4) rather than
being reported as a sweep.

### Capability matrix

The audited per-driver facts live in `Hardware/ScanCapability.py` and are
exposed as `SweepCapability.scan`, so the planner and the CLI read one model:

| driver | read path | one command per sweep | binary | opt-in available |
| --- | --- | --- | --- | --- |
| `VNA` | `sweep` | no | no | — |
| `NanoVNA`, `NanoVNA_H`, `NanoVNA_H4`, `NanoVNA_F`, `NanoVNA_F_V2`, `NanoVNA_F_V3` | `scan` | firmware ≥ 0.7.1 only | no | combined text |
| `JNCRadio_VNA_3G`, `SV4401A`, `SV6301A` | `scan` | no | no | combined text |
| `NanoVNA_V2`, `LiteVNA64` | `fifo` | yes (always) | yes | `scan_bin` |
| `AVNA` | `sweep` | no | no | — |
| `TinySA`, `TinySA_Ultra` | `sweep` | no (single channel) | no | — |
| `SimulatedVNA`, `SimulatedDevice` | `sweep` | no | no | — |

Every opt-in listed above is **unverified on hardware**: the affected
firmware has never been observed answering the command in this repository.
Test a flag on the physical device before enabling it for a device you rely
on, and record the resulting `read_path` in the run's artifacts.

## Test sequences

A sequence is a versioned JSON document describing what to do for one unit.

```json
{
  "format": "nanovna-saver-sequence",
  "version": 1,
  "name": "Antenna unit test",
  "substitutions": { "name": "dipole" },
  "steps": [
    { "action": "sweep", "start": "14.0M", "stop": "14.3M", "points": 41,
      "name": "unit {serial}" },
    { "action": "check", "limits": "limits.nvlimits" },
    { "action": "save", "path": "{serial}-{name}.s1p" },
    { "action": "report", "path": "{serial}-{name}.json" }
  ]
}
```

```
python -m NanoVNASaver.cli --sequence unit.nvseq --serial SN42 \
    --output-dir results --json results/SN42-record.json
```

Actions: `sweep`, `check`, `save`, `report`, `log`, `prompt`.

Every action accepts `fail` and `unavailable` policies (`stop` by default,
or `continue`). The schema is strict — an unknown key is an error, because a
typo that is silently ignored measures the wrong band and still reports
success.

Substitutions are a closed set: `{serial}`, `{name}` and `{date}`. Anything
else is an error. Output paths are resolved against the output directory and
refused if they escape it.

A record is **passed** only when every step ran *and* every step succeeded. A
run that stopped early, or hit an unavailable check, records what happened and
is not passed.

## Verdict log

`--log FILE` appends one row per completed run, for both a plain sweep and a
sequence:

```
timestamp,run_id,trace_id,...,outcome,verdict,rules_passed,rules_failed,rules_unavailable,reason
2026-11-01T21:31:23+00:00,SN42-e5d4b1d8cd45,...,measured,pass,1,0,0,
```

`outcome` is `measured`, `cancelled` or `failed`; only a `measured` row with
`verdict=pass` and no failed or unavailable rule is a pass. A sequence row
covers the whole unit, so a run whose measurement passed but whose later step
failed is recorded as `failed` with the step number in `reason`.

The limit identity, calibration identity and processing identity in the row
are frozen when the run ends: editing the limit catalogue afterwards does not
relabel a row already written. A `.csv` file whose header does not match is
refused rather than appended to, and a logging failure is reported (exit 2)
after the device has been released — the measurement itself is unaffected and
the gap is visible.

## Local service

**Automation service** in the main window starts an HTTP and WebSocket service
on a loopback port, with the endpoints under `/v1/`. Nothing listens until you
start it. The status line shows the address and a token generated for that run:

```console
$ curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:$PORT/v1/info
$ curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"schema_version": 1, "sweep": {...}}' http://127.0.0.1:$PORT/v1/start
```

The service sweeps the device connected in the application:

- A run and a sweep exclude each other. A run is refused with `no_device` while
  the application is sweeping, and the application refuses to sweep during a
  run. After the run the device stays connected, with its point count restored.
- A run reads through `DeviceSession`, so it returns uncorrected samples and its
  snapshot says no correction was applied. The application's calibration is not
  used.

A loopback bind keeps other machines out but not a web page open in your browser,
so these are refused for every client:

| Refused | Status | Code |
| --- | --- | --- |
| Any request carrying an `Origin` header | 403 | `forbidden_request` |
| On a loopback bind, a `Host` that is not a loopback name (DNS rebinding) | 403 | `forbidden_request` |
| A request body that is not `Content-Type: application/json` | 415 | `unsupported_media_type` |

So a browser page cannot be a client of this service; use a script. The token
keeps out other accounts on the same machine.

The window starts the service without an export directory, so `/v1/export`
returns 409 `export_disabled` there. A script that starts `MeasurementService`
itself can pass `export_dir`. An export that the run's data cannot fit, such as a Touchstone file for a run
with no S11, returns 409 `export_refused` and writes nothing.
