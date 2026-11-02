# Offline Walkthrough

This walkthrough exercises the whole application without a physical VNA. It
covers sessions, traces, comparison, calibration coverage, presets, reports,
and the export placeholders, and it is the manual counterpart of the automated
checks in `tests/`.

New to the project? [FEATURES.md](FEATURES.md) lists each workflow with the
button that opens it and the document that describes it; this page is the
single end-to-end run through them.

Everything here uses the built-in simulator. Data produced that way is always
labelled: the window title gains `[SIMULATED: <scenario>]`, Touchstone exports
carry a `! NanoVNASaver export: SIMULATED DATA ...` comment that survives a
reload, and sessions and reports record the same provenance. Never present a
simulated sweep as a device measurement.

## 1. Start the simulator

1. Open the connection panel.
2. Tick **Simulated device (demo)** and choose a scenario in the **Scenario**
   combo:
   * **Antenna resonance** — a 14.15 MHz resonance, 50 Ω radiation
     resistance, Q = 20; reflection-only, its S21 channel is identically
     zero rather than a transmission measurement;
   * **Butterworth low-pass filter** — 5th order, fc = 100 MHz;
   * **Coaxial cable** — a 10 m line, velocity factor 0.66, 10 dB/100 m at
     100 MHz;
   * **Fixed attenuator** — a flat 3 dB pad with 30 dB return loss;
   * **Touchstone replay (file)** — see [REPLAY.md](REPLAY.md); press
     **Choose replay file ...** first.
3. Press **Connect to device**. No serial port is opened and no hardware is
   touched.

## 2. Sweep

Set the sweep span in the **Start** / **Stop** / **Center** / **Span** fields
(or pick a preset), then press **Sweep** (Ctrl+W).

* The state label moves through `Idle`, `Preparing`, `Measuring`,
  `Averaging`, `Retrying`, and finally `Completed`, `Cancelled` or `Failed`.
* Progress shows the completed segment and average, plus an estimate once
  enough of the sweep has run.
* **Stop** (Esc) requests cancellation and returns immediately; the label
  passes through `Cancelling`. A cancelled sweep is never shown as a
  measurement: the display either returns to the last complete sweep and says
  so, or is emptied and labelled.
* A failed sweep reports the device's own error rather than leaving stale
  numbers on screen.

## 3. Calibration coverage

Load a calibration file. The coverage annotation tells you which measured
frequencies the calibration actually covers:

* fully covered;
* interpolated between calibration points;
* outside the span, where the endpoint value is held;
* uncorrected, when the calibration cannot correct that parameter.

Coverage is a span statement. It says nothing about calibration age,
connector wear, or measurement uncertainty, and must not be read as an error
bar. See `DEVELOPMENT.md` for the full list of what it cannot establish.

The correction itself is applied in bulk per segment; see
[CALIBRATION_BULK.md](CALIBRATION_BULK.md) for what changed and what did not.

## 4. Captured traces

Use the trace panel below the charts:

* **Capture current sweep** freezes the displayed measurement as a new trace.
* **Import file ...** adds a Touchstone file (`s1p`/`s2p`) as an imported
  trace.
* **Rename ...**, **Delete**, **Move up**, **Move down**, and the visibility
  checkbox next to each trace.
* **Use as reference** marks the selected trace as the comparison reference.

Several traces can be shown together on the frequency plots, each with its
own color in the chart legend. TDR currently displays a single trace; the
trace panel states this limitation rather than silently ignoring the extras.

## 5. Comparison

Pick a trace and a reference, then choose how to compare in the **Compare**
combo:

* magnitude in dB,
* phase in degrees,
* impedance in ohms,
* complex S-parameter.

The summary names both traces, the unit, the number of compared points, and
the compared span in MHz. Comparison only ever uses frequencies both traces
have data for — nothing is extrapolated. A pair with different reference
impedances, or with no channel in common, reports unavailable and why.

With a marker placed, the marker area reports the comparison at that exact
frequency, so a delta is never attributed to the wrong curve.

## 6. Sessions

* **New session**, **Open session ...**, **Save session** and
  **Save session as ...** live in the session window, with the recent-session
  list and the recovery state below them.
* A session is a ZIP of JSON files (see `SESSION_FORMAT.md`); there is no
  pickle and members are read by name, so a corrupted archive fails cleanly.
* Saving captures every trace with its own name, color, and provenance,
  plus markers, charts, sweep settings, and the selected reference.
* Reopening restores all of it without connecting or sweeping.
* Unsaved work prompts before closing or replacing a session.
* While work is unsaved, a recovery snapshot (`.nvrecovery`) is written next
  to the intended save. The recovery file never overwrites your deliberate
  save.

## 7. Presets

Presets capture a sweep configuration plus the charts and analyses that suit
it. Three built-ins ship: **Antenna tuning**, **Filter testing** and
**Cable testing**.

* Applying a preset only changes settings; it never connects or sweeps.
* A preset whose span or point count the connected device cannot support is
  rejected before anything is applied.
* Built-ins cannot be renamed or deleted. Your own presets can be saved,
  duplicated, renamed, deleted, exported, and imported.

## 8. Antenna tuning

**Antenna tuning ...** in the main window's button grid is a guided overview
-> minimum -> zoom -> repeat -> export workflow. See
[ANTENNA_TUNING.md](ANTENNA_TUNING.md) for the full description.

* The overview sweep runs inside the device range with a supported point
  count, then reports the lowest **measured** VSWR sample and places a marker
  on it.
* Zoom centres a narrow sweep on that sample; repeat re-runs the same window
  while you adjust the antenna.
* Export writes a one-port Touchstone file named after the resonance and the
  timestamp, with the measured minimum and its provenance in the comments.
* A cancelled sweep, a failed sweep, or a completion from a different span
  never advances the workflow or becomes exportable data.

## 9. Reports

The report window builds a document from the traces you select:

* title, DUT, and notes;
* per-trace provenance (device, simulated status, calibration identity);
* per-trace analyses: resonance, VSWR, insertion loss, cutoff, group delay,
  cable length;
* embedded plots.

The report is frozen when previewed: a later sweep cannot change a report you
have already previewed. Export is available as **HTML** and **CSV**.

Each trace carries its provenance labels in the report: simulated status,
completeness, calibration state, one-port data, and whether correction was
applied and by what. Read those before comparing two traces.

**Manage limits ...** opens the same limits window described above, so a
report can be built with its verdicts already frozen into it.

### Reading an incomplete two-port report

A forward-only measurement has S11 and S21 but no S12/S22. The report says so
in two ways: the trace is labelled `one-port data (no S21)` when S21 is
missing, and an S2P export writes the missing columns as explicit zeros while
naming them in the file comments and in the export button label.

A zero there is a placeholder, not a measurement. Reciprocity-based
inferences, full two-port de-embedding, and anything requiring S12/S22 are
not valid for such a file. Treat the zeros as absent data.

**PDF is not implemented.** There is no `QWebEngine` or `QPrinter`
dependency, so a PDF would require adding one; the HTML file is
self-contained and prints correctly from a browser.

## 10. Export

* `S1P` writes S11.
* `S2P` writes all four columns. A channel the measurement never had (for
  example S12/S22 on a one-port sweep) is written as an explicit zero
  placeholder, and the file comments and the button label both say so. This
  is not a complete two-port characterization.
* Export validates the whole dataset before opening the file, so a rejected
  export never leaves a truncated file behind.
* The simulation marker is re-read when the file is loaded again.

Provenance is written into every export; [FEATURES.md](FEATURES.md) lists
every surface on which simulated data is labelled.

## 11. What this walkthrough does not verify

* Any physical device: accuracy, calibration validity, protocol quirks, or
  firmware-specific behaviour.
* Connector or cable effects.
* Windows/macOS packaging of bundled resources, beyond the startup smoke
  check.

Those remain hardware and platform acceptance items and are recorded as
unverified rather than assumed.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the command-by-command matrix of
what has been verified on which platform.