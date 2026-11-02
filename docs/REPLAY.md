# Touchstone Replay

Touchstone replay is a *simulated device scenario*: instead of an analytic
model, the sweep pipeline answers every requested frequency from a recorded
`.s1p` or `.s2p` file. It is meant for demonstrations, UI work, report and
export checks, and regression comparisons against a known dataset — never for
presenting recorded data as a new measurement.

Everything produced by replay carries the normal simulated-device
provenance: the window title gains `[SIMULATED: replay]`, Touchstone exports
carry the `SIMULATED DATA` comment, and sessions and reports record the
scenario plus the file that was replayed.

## Workflow

1. Open the connection panel and tick **Simulated device**.
2. Choose the scenario **Touchstone replay (file)**.
3. Press **Choose replay file ...** and pick a `.s1p` or `.s2p` file. The
   control lists the loaded file, its port count, its point count, its
   frequency span, and the channels it provides.
4. Press **Connect**.
   * The sweep span is moved onto the file's own range automatically, so the
     default configuration never asks for a frequency the file cannot answer.
   * The first sweep starts immediately, exactly as for any other simulated
     scenario.
5. Sweep normally. Segments, averaging, cancellation, calibration
   (whatever you have loaded) and the display all behave as usual.

### The default sweep for a file

Connecting a replay file derives the default start/stop from the file span
and the sweep's own grid arithmetic, because `Sweep` rounds each segment
boundary to whole hertz and can otherwise request a few hertz past the file's
last sample:

* linear sweeps: the stop is computed from the largest whole-hertz step that
  fits the requested point and segment count, so the last requested
  frequency is the file's own last sample at most;
* logarithmic sweeps: the stop stays at the file's last sample and every
  rounded segment boundary is validated against the span;
* the requested point count and grid mode are always preserved. If the
  segment count cannot fit, it is reduced and the loaded-file label says so;
* a positive-span file never collapses into repeated samples of a single
  frequency, and a span that cannot hold the configured points even in one
  segment is refused before connecting, naming the settings that would fit;
* a genuine one-frequency file is replayed as an intentional start == stop
  sweep, and the label says that too.

A reduced segment count or a continuous-wave default is written next to the
loaded file (`Default sweep: ...`); nothing about the file itself changes.

A file is only used if it passes validation; see *Limits* below. A rejected
pick keeps the previously loaded file and reports why, so a bad selection
never changes what the next sweep replays.

While a sweep is running, changing the device or the replay file is refused:
the running worker owns the device conversation, and swapping the source
mid-acquisition would change the meaning of samples that were already read.
Disconnect first, then choose another file.

## Interpolation contract

* A requested frequency that is **exactly a sample frequency** in the file
  returns that sample's value unchanged — bit-identical, not re-derived.
* A requested frequency **between two samples** returns the complex value
  from independent linear interpolation of the real and imaginary parts (the
  Touchstone `slinear` interpretation).
* A requested frequency **outside the file's span is never extrapolated**.
  The sweep is rejected before any sample is read, the failure names the
  requested span and the file's span, and the display keeps the previous
  result instead of showing an invented one. Editing the sweep after
  connecting is allowed; it is checked again on the next sweep.

The interpolation is applied per channel, so S11 and S21 of an `.s2p` are
interpolated on the same frequency grid as the file's own samples.

## One-port and two-port honesty

* **`.s1p`** provides S11 only. No S21 is fabricated, zero-filled or carried
  over from another trace: the second channel is not read at all, the
  snapshot has no S21 at all, and any export therefore writes a one-port
  file.
* **`.s2p`** provides S11 and S21. A complete two-port file's S12 and S22
  are retained but never published as forward channels: the sweep pipeline
  carries S11 and S21 only, and the loaded-file label says the reverse pair
  is "available for reversal". They reach a measurement only through a
  *reverse view*, which re-labels them exactly as a physical reverse sweep
  is re-labelled (its S11 becomes S22, its S21 becomes S12). A one-port file
  cannot be reversed at all, because those channels were never in it.

### Declared channels are a contract

The sweep worker validates every channel the connected device *declares*.
An ordinary VNA and every analytic scenario declare S11 and S21, so a read
that comes back empty or short for either is a device fault: the segment is
rejected, retried a bounded number of times, and the sweep fails without
publishing any sample. In particular, an empty S21 read on a two-channel
device is never accepted as a one-port measurement. Only a source that
explicitly declares S11 alone (a `.s1p` replay) skips the S21 read, and then
S21 is never read, published or exported at all.

## Reference impedance

The simulated device is a 50 ohm system. **One-port** data in a file with a
different reference impedance is accepted through an explicit, recorded
renormalization: the file's own impedance is kept as `source_impedance`, the
samples are converted with the same closed form the rest of the application
uses, and both the loaded-file label and the recorded provenance name the
conversion (`renormalized from 75 ohm`). The physics is unchanged — a 75 ohm
resistor (Γ = 0 at 75 ohm) replays as Γ = 0.2, which is the same resistor in
a 50 ohm system.

**Two-port** non-50 ohm files are still rejected with a message naming both
impedances. Renormalizing S21 needs all four S-parameters, and converting S11
while passing S21 through untouched would mix two reference impedances in a
single snapshot. Export such a file at 50 ohm first (for example with
`scikit-rf`'s `Network.renormalize`) and replay that copy.

Touchstone 2.x is refused explicitly rather than parsed leniently: its
keyword blocks and option-line meaning are a separate parser scope.

## Limits

| Limit | Value | Reason |
| --- | --- | --- |
| File size | 32 MiB | Bounded memory while parsing |
| Samples per channel | 250 000 | Bounded sweep cost |
| Reference impedance | 50 ohm, or one-port data converted explicitly | Two-port conversion would need all four S-parameters |
| File types | `.s1p`, `.s2p`, S-parameters only, Touchstone 1.x | Only S11/S21 are published; 2.x is a separate parser |
| Frequencies | finite, positive, strictly ascending, unique | Deterministic interpolation |
| Extrapolation | none | A replayed value must originate from the file |

Files are read as UTF-8 Touchstone 1.x text. A file with out-of-order rows is
reordered by the normal Touchstone parser, then validated on the resulting
grid; duplicate frequencies are rejected.

## Provenance surfaces

The replay source is recorded everywhere a simulated measurement is recorded:

* the main window title: `[SIMULATED: replay]`;
* the display comment: `SIMULATED DATA (scenario replay): replay of <file>
  (<kind>, <channels>, complex linear interpolation, no extrapolation)`,
  including the ignored-parameter note for `.s2p` files;
* Touchstone exports: the `SIMULATED DATA` marker plus the same comment;
* stored sessions and generated reports.

See `OFFLINE_WALKTHROUGH.md` for the wider offline workflow.

## Tests

`tests/test_replay_scenario.py` covers file validation, the interpolation
contract, channel honesty, preflight rejection, sweep-worker acquisition,
export/session/report provenance, and GUI selection behaviour.

`TestDefaultRangePlan` proves the default range contract against the real
`Sweep` grid (rounding overshoot, narrow files, continuous-wave files,
linear/logarithmic/segmented spans). `TestDeclaredChannels` proves that a
declared-but-missing S21 fails the sweep without publishing a one-port
result, while a genuine one-port source is never asked for S21.