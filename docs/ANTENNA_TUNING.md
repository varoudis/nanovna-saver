# Antenna Tuning Workflow

A guided, low-friction path for the most common NanoVNA job: finding an
antenna's resonance, watching the dip move while you adjust the antenna, and
saving the finished measurement as an honest one-port Touchstone file.

Open it with **Antenna tuning ...** in the main window's button grid, or from
the same grid after a session is loaded. The window is a thin front end over
the normal sweep machinery: it never connects a device, never starts a sweep
by itself, and never runs an unattended loop.

## 1. Broad overview sweep

The window starts prefilled with the current sweep span, the connected
device's point count, and the current segment count.

* **Start / Stop** — the wide span used to find the response. Frequencies
  outside the device range are clamped to it and the clamping is named in the
  "Next sweep" line; an empty or reversed span is refused instead of being
  quietly turned into another measurement.
* **Points** — only counts the connected device offers are listed. A point
  count that the device cannot produce is replaced by the nearest supported
  count and the substitution is reported.
* **Segments** — more segments mean finer steps. The planned grid is built
  from the device's real sweep arithmetic, not from a nominal
  `(points - 1) x segments` span, so no planned or measured frequency can
  leave the entered span or the device range. A segment count whose grid
  does not fit is reduced (and the reduction reported); the requested stop
  is kept whenever the measured grid reaches it, and only a stop the grid
  cannot reach is quantized onto the measured grid, with a note.
* **Logarithmic steps** — carried into the sweep. Note that a logarithmic
  sweep with a single segment still measures linearly: the window says so
  rather than pretending otherwise. Raise the segment count for real
  logarithmic spacing.
* **Use device range** — fills start and stop from the device.

Press **Overview sweep**. The configuration is applied as one change (span,
segments, point count, mode, logarithmic spacing) and the sweep runs once
through the normal worker, with the ordinary progress, cancellation and
provenance handling.

The lowest VSWR is then taken from the **measured S11 samples** — the winning
sample is chosen by value, so its index and frequency are whatever was
actually measured. A minimum on the first or last sample is reported as an
edge minimum, because the true dip may lie outside the measured span.
Non-finite samples are skipped and counted.

With **Place a marker at the minimum** enabled (default), marker 1 moves onto
the measured minimum sample after every completed sweep of this workflow.

## 2. Zoom, tune, repeat

* **Zoom span** — the width of the narrow sweep, centred on the measured
  minimum. It is seeded with one twentieth of the span that was just measured.
* **Zoom points** — point count inside the narrow sweep.
* **Zoom to minimum** — validates and sweeps the narrow window. Rejected
  (without measuring) when the span is zero, negative, wider than the device
  range, narrower than the requested point count allows, or when the minimum
  no longer lies inside the device range.
* **Repeat sweep** — repeats the *current* narrow window. Use it after each
  mechanical adjustment; the window stays put so the dip is visible moving.
* **Stop sweep** — cancels the running sweep. The last complete measurement
  stays on screen and exportable; this workflow stops advancing.

Clamping is only ever applied to the *window*, never to the measurement: a
zoom near a device edge is shifted into range, still contains the minimum, and
says that it was clamped. The window is also quantized onto the step grid the
device really measures, so the measured span equals the planned span, and a
request that cannot be met is refused rather than approximated.

## 3. Export

**Export S1P ...** writes the completed measurement using the existing
Touchstone machinery:

* the measured (corrected, if a calibration is loaded) S11 samples and the
  reference impedance of the measurement;
* comments naming the sweep, the sample count and the **measured** minimum
  (value, frequency, sample index, and a warning when it is at a span edge);
  the minimum's return loss is the standard `-20*log10(|S11|)`, i.e. positive
  for a passive match, `0 dB` at total reflection and infinite for a perfect
  match;
* correction provenance (applied or not, calibration identity) and the
  simulated-data marker, which survives a reload;
* a descriptive filename, e.g. `antenna_14p150MHz_20261101-123015.s1p`
  (prefix, measured resonance, timestamp). The suggested path never
  collides with an existing file. Only the file that is actually written is
  ever replaced, and only after you confirm that exact file: choosing
  `named.s2p` writes `named.s1p`, so the question names `named.s1p` and
  `named.s2p` is left untouched.

A measurement containing non-finite samples cannot be written; the window
says so as soon as the sweep completes rather than at export time.

## Status and safety

* All sweep-flavoured controls are disabled while a sweep is pending or
  running; **Stop sweep** is the only enabled action then.  The controls are
  handed back when the worker thread really ends, not when its terminal is
  announced: the terminal is published from inside the run, so a cancelled
  *first* sweep - which leaves no stored measurement to recover from - still
  returns to an enabled **Overview sweep** and a disabled **Stop sweep**.
* With no device connected the window says so and offers no sweep.
* Every sweep this window starts is bound to a fresh run token that is
  carried by the worker's own completion. A completion is adopted only when
  that exact token comes back *and* the completed snapshot proves it measured
  the planned span, point count, segment count and logarithmic setting. A
  completion queued by a cancelled run therefore cannot advance a newly
  started one, even when both plans are identical. A stored result is
  dropped when the displayed data no longer matches it.
* An export never clobbers a concurrently created file: publication is an
  atomic no-clobber hard link, or an exclusively created copy where the
  filesystem cannot link - never a rename over whatever is there.
* The window changes nothing else: charts, traces, markers, references,
  sessions and presets are left as they were (apart from the sweep settings it
  has to change to measure what it says it measures).

## Calibration preflight

The "Calibration" line describes the coverage of the *planned* grid using the
loaded calibration: in range, partially covered, outside, or absent. It is a
span statement only — it says nothing about calibration age, connector wear,
or measurement uncertainty.

## Not verified without hardware

Everything below is exercised automatically against the built-in simulator,
the same code path a physical device uses. It is **not** a claim about any
particular instrument:

* that a real device accepts the planned spans and point counts without
  clamping further, and that its sweep-to-sweep noise keeps the reported
  minimum stable to the sample;
* that the measured VSWR of a physical antenna responds the way the analytic
  model does while tuning;
* device-specific behaviour of `resetSweep` after a multi-segment sweep;
* calibration accuracy, connector repeatability and the effect of movement on
  a real measurement.

See `OFFLINE_WALKTHROUGH.md` for the rest of the simulator workflows.