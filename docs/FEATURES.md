# Feature entry points

This page is a map: one short section per user-visible feature, with the
button that opens it and the document that describes it in full. It exists so
you can find the right document without reading the source, and so the
feature documents have a single, maintained index.

Everything here is available without a physical VNA: the simulator
(`Simulated device (demo)` in the connection panel) exercises the same code
paths. See [OFFLINE_WALKTHROUGH.md](OFFLINE_WALKTHROUGH.md) for an
end-to-end session that uses only the simulator.

Button captions below are the literal captions in the current UI. If a
caption is missing from your build, the build is older than this page.

## Touchstone replay

Tick **Simulated device (demo)**, choose the scenario
**Touchstone replay (file)**, then press **Choose replay file ...** and pick a
`.s1p` or `.s2p` file. The control names the loaded file, its port count,
point count, span, and the channels it provides; **Rescan** re-reads the
source. Replay is a simulated-device scenario, so its data is labelled like
any other simulated data and is never presented as a new measurement.

Full description: [REPLAY.md](REPLAY.md).

## Antenna tuning

**Antenna tuning ...** in the main window's button grid opens a guided
overview -> minimum -> zoom -> repeat -> export workflow: **Overview sweep**,
**Use device range**, **Place a marker at the minimum**, **Zoom to minimum**,
**Repeat sweep**, **Stop sweep**, **Export S1P ...** and a
**Logarithmic steps** checkbox.

The window never connects a device and never sweeps on its own; every sweep
it requests goes through the normal sweep machinery, so a cancelled or failed
sweep cannot become exportable data.

Full description: [ANTENNA_TUNING.md](ANTENNA_TUNING.md).

## Measurement limits

**Limits ...** opens the limits window: **New ...**, **Edit ...**,
**Duplicate**, **Load ...**, **Export ...**, **Remove**, **Use for reports**,
**No limits**, and **Check captured traces**. A limit set is a versioned JSON
document, so it can be kept under version control next to the design it
belongs to. A checked trace reports PASS, FAIL or UNAVAILABLE; UNAVAILABLE
means the question could not be decided (missing channel, band not covered,
incomplete acquisition) and is never treated as a pass. Report previews and
HTML/CSV exports freeze the selected set together with its results.

Worked example: `tests/data/example.nvlimits` (schema version 1) and
`tests/data/example_v2.nvlimits` (version 2, optional per-hertz slopes and the
`s11_phase_deg` metric).

Full description: [LIMIT_SETS.md](LIMIT_SETS.md).

## Sessions

**Session ...** opens the session window: **New session**,
**Open session ...**, **Save session**, **Save session as ...**, plus the
recent-session list and the recovery state. A session is a ZIP archive of
JSON members (no pickle), so it restores traces, markers, charts, sweep
settings and provenance without the original Touchstone or calibration files
and without a device. Unsaved work is protected by a `.nvrecovery` snapshot
that never overwrites a deliberate save.

Full description: [SESSION_FORMAT.md](SESSION_FORMAT.md).

## Calibration bulk correction and coverage

Calibration correction is applied once per segment rather than once per
point (`Calibration.correct11_bulk`, `correct21_bulk`, `correct_delay_bulk`),
which is what keeps multi-thousand-point sweeps responsive. The calibration
status line in the sweep panel reports the span the loaded calibration
actually covers, and distinguishes fully covered, interpolated, held at the
span end, and uncorrected.

Coverage is a statement about frequencies only. It says nothing about
calibration age, connector wear, or measurement uncertainty.

Full description: [CALIBRATION_BULK.md](CALIBRATION_BULK.md); the
measurement-side contract is in the code docstrings and in the
[offline walkthrough](OFFLINE_WALKTHROUGH.md) section 3.

## Calibration age and verification

A calibration records when its standards were measured, and with what. An
optional reminder fires at an interval you choose; older files report an unknown
age rather than a guessed one. A connected short, open or load can be swept and
compared with the model the calibration assumes, against a tolerance you set.
These are prompts and checks, not uncertainty estimates.

Full description: [CALIBRATION_HEALTH.md](CALIBRATION_HEALTH.md).

## Provenance and simulated labelling

Every capture records where it came from: source, simulated flag, device,
application version, creation time, sweep and calibration identity,
completion state, simulator scenario, and the processing identity described
below. That record travels with traces into sessions, reports and Touchstone
comments.

Simulated data is marked in three places, all derived from the same record:

* the window title gains `[SIMULATED: <scenario>]`;
* Touchstone exports carry a `! NanoVNASaver export: SIMULATED DATA
  (scenario <name>)` comment, which is re-read when the file is loaded again;
* sessions and reports store the scenario with the trace.

A simulated sweep must never be presented as a device measurement.

## Post-calibration processing

Processing runs after calibration as an ordered, opt-in pipeline: port
extension, fixture de-embedding, reference impedance, running average, then
frequency smoothing. With no stages selected the pipeline is the identity, so
nothing is altered by default. The configuration is recorded as a stable
identity in provenance, which is what makes "these numbers were smoothed over
11 points" auditable later.

Smoothing and running average produce a *changed* measurement, never a more
accurate one, and they are documented that way.

Full description: [PROCESSING.md](PROCESSING.md).

## Two-port and replay restrictions

These are current limits, stated here so no document has to imply more:
* Only S11 and S21 are acquired in one pass (`Measurement.ACQUIRED_PARAMETERS`).
  A single sweep cannot produce S12 or S22. The **two-port reversal** capture
  (`Reversal.py`, left-column "Two-port reversal" box) gets them the physical
  way: measure the DUT, turn it around, measure again, then combine. The
  reverse sweep is re-labelled, not transformed — its S11 is the DUT's S22
  and its S21 is the DUT's S12 — and a reverse sweep that did not measure a
  channel leaves that channel absent rather than zero. Two sweeps are only
  combined when they agree on the grid, reference impedance, calibration
  identity, source and simulated flag; otherwise the reason is shown and both
  originals are kept. An imported `.s2p` can also carry those columns.
* Replay of a `.s1p` file provides S11 only. Absent means absent: a missing
  channel is never filled with zeros and treated as data.
* S2P export writes a channel the measurement never had as an explicit zero
  placeholder, names it in the file comments, and names it in the export
  button label. A zero there is a placeholder, not a measurement.
* Reports label a trace without S21 as `one-port data (no S21)`.
* PDF export is a Qt-drawn document; HTML (self-contained) and CSV remain the
  primary report formats.
* TDR gating (`TDRGating.py`) requires a sweep that starts at DC, has uniform
  spacing and at least 16 samples. The gate is a documented raised-cosine
  band with a stated transition width; an identity gate round-trips to
  floating-point precision. The **TDR window** offers it under "Time gating
  (derived)": keep-band or bandstop, four tapers, band edges and transition
  width in metres, off by default. The chart then draws an extra dashed
  curve — the gate's **weight profile** (1 = kept, 0 = rejected), not a
  second amplitude, because the gating engine and the window's legacy
  transform use different amplitude normalisations (measured on one sweep:
  0.497 against 0.006) and plotting both on one axis would invent a
  comparison. A gate the data cannot support is refused with the reason
  shown in words, never drawn on a guess; turning gating off clears the
  curve, and a new sweep re-evaluates it rather than leaving a stale one.
* Analytic simulator scenarios declare the channels they actually model. The
  antenna scenario is one-port: it offers S11 and **no S21 at all**, so code
  must not assume two channels exist.

## Verification status

The features above are covered by the test suite in `tests/` and by the
simulator, not by a physical instrument. See
[CONTRIBUTING.md](CONTRIBUTING.md) for what has been verified on which
platform, and [OFFLINE_WALKTHROUGH.md](OFFLINE_WALKTHROUGH.md) section 11 for
what an offline walkthrough deliberately does not prove.
