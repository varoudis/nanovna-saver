# Trace Processing: Stages, Smoothing and the Running Average

`src/NanoVNASaver/Processing.py` is the single ordered-processing contract
(DSP-01/02/03). Everything in it is Qt-free, deterministic and works only on
the samples that were actually measured: no sample is invented, interpolated
or extrapolated, and a stage that cannot be honoured raises instead of
quietly doing something else.

Two rules run through the whole module and the UI built on it:

* **A raw channel is never overwritten.** A processed result is a separate,
  named, derived artifact carrying the `ProcessingConfig` that produced it
  (`ProcessingResult.config`, and the frozen `Provenance.processing`
  description on a measurement snapshot). The acquisition stays available
  beside the processed view, so a standard, a report or a session can always
  show what was measured.
* **A missing channel is not a zero sample.** An absent `S21` is absent, not
  `0`, and nothing here fills one in.

## Stage order

```text
raw capture
  -> calibration            (Calibration.py, outside this module)
  -> port extension         port_extension
  -> fixture de-embedding   fixture
  -> reference impedance    renormalize
  -> running average        running_average
  -> frequency smoothing    smoothing
```

`STAGE_ORDER` fixes this order. A configuration may omit stages but may never
reorder them, because the order is part of what the pipeline means: smoothing
after averaging is not the same trace as averaging after smoothing. Each
stage appears at most once per configuration; `ProcessingConfig` refuses a
duplicate and refuses stages that arrive out of order.

`apply_processing(channels, config, *, reference_impedance=50.0,
running=None, average_context="")` applies the stages in order to a copy and
returns a `ProcessingResult`:

| Field | Meaning |
| --- | --- |
| `channels` | New tuples; the input mapping is untouched |
| `config` | The configuration that was applied |
| `reference_impedance` | Reference impedance after the `renormalize` stage |
| `notes` | One line per stage saying what was actually applied |
| `running` | The accumulator to carry into the next run, or `None` |

The caller keeps the previous result when a stage raises `ProcessingError`:
a half-processed trace is never published, and unprocessed samples are never
labelled as processed ones.

## Port extension

Parameters are per port: `port1_delay_ps`, `port2_delay_ps`,
`port1_loss_db_per_ghz`, `port2_loss_db_per_ghz`. All are optional and default
to zero; a supplied value must be finite and `>= 0`.

A reflection (`S11`, `S22`) travels through its fixture twice, so both the
delay and the loss are doubled. A transmission (`S21`, `S12`) passes through
each port once, so port 1 and port 2 contribute one pass each. The doubling
is geometry, not an option. Delay is applied as a phase rotation
`exp(+j*2*pi*f*delay*passes)` and loss as a gain of
`10**(loss_db_per_ghz * (f/1 GHz) * passes / 20)`.

Renormalization to a new reference impedance is exact for one-port data
(`G_new = (G + a)/(1 + a*G)` with `a = (Z_old - Z_new)/(Z_old + Z_new)`) and
is **refused** for transmission channels: a two-port measured with unequal
port impedances needs the full wave-cascading transform, and approximating it
would be worse than not doing it.

## Frequency smoothing (DSP-02)

```python
smooth_channel(points, aperture_points)
smoothing_aperture_for(points, percent)   # percent of span -> odd point count
smoothing_stage(aperture_points)          # validated stage
smoothing_stage_for(points, percent)      # None when it would be a no-op
effective_smoothing_aperture(samples, aperture_points)
```

**The aperture is counted in samples, never in hertz.** It is always odd, so
every window is centred on the sample it reports. `validate_smoothing_aperture`
accepts a positive odd integer (or a float holding a whole number) and refuses
`0`, negative, even, non-numeric, `None` and non-finite values. It validates
rather than coerces: a stored description such as `{"aperture_points": 0}`
means the operator asked for something impossible, and silently substituting
1 or 3 would make the recorded provenance lie about the numbers on screen.

**The aperture is clamped to the data, never padded.** A wider aperture is
reduced to the largest odd count the channel can supply
(`effective_smoothing_aperture`), and the applied count is what appears in the
notes. A channel of one or two samples can only support an aperture of 1,
which means "no smoothing": the samples come back unchanged and the caller can
say so instead of claiming a window was applied.

**Edge windows shrink.** Near either end the window contains only the samples
that exist — no wrap-around, no padding, no extrapolation — so nothing is
smoothed with data outside the measured band.

**The arithmetic is a centred complex moving average.** Real and imaginary
parts are averaged together, which is the only choice that keeps magnitude,
phase and Smith views of one point consistent: averaging magnitudes alone
biases a resonance toward a deeper null, and averaging phases alone is
meaningless across a ±180° wrap. The cost is linear in the number of samples
(a prefix sum makes the aperture width irrelevant), and memory is one extra
buffer.

**A non-finite sample raises** rather than spreading `NaN` over the whole
neighbourhood.

**Uneven bandwidth is reported, not hidden.** Because the aperture counts
samples, its frequency width is `aperture * step` only on a uniform grid. On a
logarithmic or otherwise irregular sweep the width varies across the trace;
`apply_processing` notes this for the affected channels.

> **Smoothing does not improve accuracy.** It widens narrow features and
> lowers peaks: the trace after smoothing is a *different measurement*, not a
> better one. It cannot recover information the sweep did not capture, and a
> smoothed null is never a truer null than the raw one.

## Running average (DSP-03)

The worker blends once per complete sweep, when the last segment arrives and
before that sample is published, so the sample, the snapshot and the charts all
hold the average. A partial sample has not been averaged and its provenance
does not claim the stage. Stages ordered after the average, such as smoothing,
run on the whole averaged sweep, while the accumulator keeps the unsmoothed
values. A new calibration, offset delay, device, grid or pipeline starts a new
average.

```python
RunningAverageState()                     # empty accumulator
state.blend(channels, *, context, mode=None, alpha=None)
state.weighting    state.effective_count
state.reset()      state.invalidate(reason)      state.with_policy(...)
```

The accumulator is immutable. `blend` returns `(new_state, published_points)`;
the new state is what the next sweep needs, and `apply_processing` carries it
out as `ProcessingResult.running`. **An integrator that drops
`result.running` restarts the average on every sweep** — the state is not
shared behind the caller's back.

### Weighting

| Mode | Recurrence | Meaning |
| --- | --- | --- |
| `cumulative` | `avg_n = avg_(n-1) + (x_n - avg_(n-1))/n` | Exact arithmetic mean of every sweep blended so far |
| `exponential` | `avg_n = (1-alpha)*avg_(n-1) + alpha*x_n` | EWMA; recent sweeps dominate |

Both are applied to the **complex** sample, so each averaged point is still a
single complex value and every representation derived from it agrees.

`cumulative` keeps one complex number per point regardless of how many sweeps
have been taken, so averaging for an hour costs no more memory than averaging
for a second.

`exponential` has the closed form
`sum(alpha*(1-alpha)**(n-k) * x_k) + (1-alpha)**n * avg_0`. Its weight is a
fixed `alpha` in `(0, 1]`; `alpha = 1` keeps only the newest sweep.

`effective_count` states the number of sweeps that contribute to the current
average, in one unit for both modes: the exact count for `cumulative`, the
equivalent count `1/alpha` for `exponential` (clamped to `[1, count]`). The UI
therefore has one unambiguous number to show.

`weighting` is the human-readable form of the same fact, e.g.
`cumulative mean of 12 sweep(s)` or
`exponential alpha=0.5 over 12 sweep(s), equivalent to 2 equal sweeps`.

The policy is validated, never repaired: an unknown mode, `alpha` on
`cumulative`, `alpha` outside `(0, 1]`, or `exponential` without an `alpha`
all raise `ProcessingError`.

### What may be blended

Only a **complete** sweep is blended. An empty channel set, a partially
populated one (say `S21` still empty) or a non-finite sample raises; the state
is left untouched, so one bad sweep cannot poison every later one.

### Reset policy

The accumulator is keyed by everything that makes an old sample incomparable
with a new one. Any of the following starts a new accumulation instead of
blending across the boundary:

* a different `context` string (the integrator's statement of source,
  calibration, channel set, frequency grid and processing configuration);
* a different channel set or a different frequency grid, detected by
  fingerprinting the channel names and their exact frequencies — a caller who
  forgets to change the context still cannot mix two grids;
* a different `mode` or `alpha`, because the old weights no longer apply
  (`with_policy` also drops the average when the policy changes);
* an explicit `reset()`, which clears the average and keeps the policy.

`invalidate(reason)` clears the average and records the reason in
`discarded`. It exists for cancelled, failed or superseded runs: an
integrator that already blended in-flight data calls it so those samples
neither contribute to the average nor disappear without explanation.

### Why average at all

A running average reduces the random part of the measurement — noise,
trigger jitter, a wandering source — by combining repeated sweeps of the same
grid. It does not remove drift (the average follows it, slowly or quickly
depending on the weighting), and it cannot fix a systematic error. As with
smoothing, the averaged trace is a different measurement: a transient that is
real but short will be attenuated, and that is the operator's trade, not an
accuracy improvement.

## Determinism and bounds

Both stages are pure functions of their inputs: the same samples and the same
configuration produce bit-identical output on any platform with the same
floating-point behaviour. Cost is linear in the number of samples for
smoothing and linear in samples per sweep for averaging, with memory
independent of the number of sweeps.

## Provenance

`ProcessingConfig.identity()` is a stable digest of the active stages and
their parameters, and `ProcessingStage.describe()` is its one-line human form.
Together with `ProcessingResult.notes` (what was actually applied, including
a clamped aperture or an uneven bandwidth) they let a stored trace be
reproduced from raw data: the configuration says what was asked for, the notes
say what happened.
