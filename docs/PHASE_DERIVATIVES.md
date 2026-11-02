# Phase Derivatives: Group Delay and Phase Curvature

Two quantities derived from the phase of one measured channel, computed in
`src/NanoVNASaver/PhaseDerivatives.py` (`ANALYSIS-04`). Everything is Qt-free
and works on the samples that were actually measured: no interpolated point
is ever produced, and a quantity that cannot be computed is reported as
unavailable with a reason instead of a number.

## Units and conventions

The independent variable is **angular frequency** `omega = 2*pi*f`, not
frequency in hertz. The classic mistake in this area is confusing the two,
which is a factor of `2*pi = 6.283`.

| Quantity | Definition | Unit |
| --- | --- | --- |
| Group delay | `tau_g = -d(phi)/d(omega)` | seconds (s) |
| Phase curvature | `d2(phi)/d(omega)**2` | square seconds (s^2) |

The minus sign in the group delay is the sign convention, not an error: a
line, cable or any other delay makes phase *decrease* with frequency, so
`-d(phi)/d(omega)` is positive for a delay. `d(phi)/d(f)` in hertz would be
`tau_g * 2*pi`; it is never computed here.

Phase is taken from the complex sample (`.phase`), unwrapped first, and is
reported in the derivative's unit — the intermediate unwrapped phase is in
radians, exposed by `unwrap_phase()` for charts that want to draw it.

## Group delay

**Meaning.** The slope of the phase response. For a pure delay line
`S21 = exp(-j*omega*tau)` it equals `tau` exactly. It is what "how long does
the signal take through this device" means for a linear network, and it is the
quantity that shows a cable's length (`tau = length / (velocity factor * c)`).

**Method** (`METHOD` in the module, repeated in every result):

* three-point quadratic Lagrange stencils that use the *actual local* step
  sizes;
* central stencil on interior samples, one-sided at both ends, and a plain
  two-point difference when the record has exactly two samples.

The stencils are exact for phase that is linear or quadratic in `omega`, and
second-order accurate otherwise. Every result names this method, so a stored
chart cannot be mistaken for one produced by a different scheme.

**Unwrapping.** Each sample is corrected by the nearest multiple of `2*pi`
relative to the previous defined sample. This is the same policy as
`MarkerSearch.unwrapped_phase`; it is reimplemented here in radians to avoid a
degree round trip and to be able to count the corrections. The policy's one
assumption is stated rather than hidden: a genuine phase step larger than `pi`
between adjacent samples is indistinguishable from a wrap, so on a sweep coarser
than the phase slope such data would be unwrapped wrongly. `wraps_corrected` in
every result makes it visible that a curve wrapped at all. A curve with no
wraps is unaffected by the policy.

The grid therefore has a hard ambiguity limit, `ambiguous_delay = pi / step_omega`
where `step_omega` is the smallest step in `2*pi*f` that the stencils actually
use: no delay at or beyond it can be distinguished from one shorter by that
step. A group-delay result whose values reach it sets `wrap_ambiguous` and says
so in `describe()` ("relies on the unwrapping assumption: the slope reaches the
N s wrap ambiguity of this grid"). The arithmetic still produces numbers on such
data — it cannot know the true phase — so the flag is the honest part: for
example, this module's 101-point 100–300 MHz test sweep has 2.25 MHz and
1.25 MHz spacing, so its limit is 400 ns; a quadratic phase whose delay reaches
377 ns there is aliased (its phase steps reached 5.29 rad) and is reported as
depending on the assumption rather than as measured. `ambiguous_delay` is
exposed unconditionally so a caller can compare a delay against the limit
without recomputing the grid.

## Phase curvature (a diagnostic)

`d2(phi)/d(omega)**2` in s^2. It is exposed, and labelled `diagnostic` in every
`describe()` and in the module docstring, because it has one legitimate use and
one common abuse.

*What it detects:* departure from constant group delay. A pure delay, or any
all-pass with exactly linear phase in `omega`, has zero curvature. A discrete
echo (two paths, one `dt` longer) adds a ripple `~sin(omega*dt)` to the phase,
so the curvature shows that ripple's period, whose inverse gives the echo delay
— that is the standard "ripple period equals path difference" reading of a
group-delay or phase trace, and it is how a spurious reflection in a fixture or
a connector is identified. Likewise, the skirt of a resonator has curvature
whose sign and shape distinguish a simple pole (one resonator) from coupled
resonators.

*What it does not give you:* a material constant, an impedance, or a noise-free
value on a raw trace. Second differences amplify sample-to-sample phase noise by
a factor of about `sqrt(6)/h**2` for uniform step `h`, so an unsmoothed
curvature trace mostly shows noise; that is why `describe()` calls an
unsmoothed result a raw curve with the noise not suppressed, and why curvature
is only meaningful with a stated aperture or on data with genuinely smooth
phase. Second-order structure that is narrower than the smoothing aperture is
destroyed by the smoothing, so the aperture must be reported together with any
curvature number — it always is.

A second derivative of a *measured* trace is therefore a diagnostic, not a
measurement. Nothing in the application should present it as a specification
value; group delay, which is a first derivative and much better conditioned, is
the headline quantity.

## Irregular frequency grids

Differentiation uses the real local steps, so a non-uniform sweep (segmented or
logarithmic) is supported as long as the grid is strictly increasing. A grid
with a duplicate frequency has a zero step and a grid that decreases has a
negative step; neither has a derivative, so such data is rejected as
**unavailable** with the offending sample indices named. The frequency grid of
the chosen channel is checked once, not per point.

## Smoothing

`aperture_points` (odd, at least 1, default 1 = no smoothing) is applied
*before* unwrapping and differentiation, using `Processing.smooth_channel`
itself — the same symmetric, odd-width, truncated-rather-than-padded complex
moving average the processing pipeline uses. It is not reimplemented. Three
consequences are made explicit rather than left to be discovered:

* Smoothing complex samples (not phases) means a wrapped curve is averaged
  continuously across `+/-180` instead of tearing at the branch.
* A window's mean is the phase at the window's **frequency centroid**, not at
  the middle sample's frequency. The two coincide only where the window is
  symmetric in frequency about that sample: true for an interior window of an
  equal-step grid, false for a window truncated at either end (it is centred on
  the samples that exist) and false where the local step changes, as at a
  segment boundary. Differentiating against the wrong abscissa tilts the slope:
  on a delay line at aperture 9 with 5 MHz steps the first samples are off by
  50%, while against the centroid they are off by 1e-14. So the derivative
  follows the centroid, `DerivativePoint.omega` says which abscissa each value
  really belongs to, and `off_centre` counts the values that differ from their
  sample's own frequency (8 of 101 at aperture 9). No sample is discarded and
  none is silently relocated.
* An aperture that covers the whole record is refused as unavailable: every
  smoothed value would be identical, the phase would be flat and the answer a
  confident "0 s", which is exactly the fabricated number this module must not
  produce. A structurally invalid aperture (0, negative, even) raises
  `PhaseDerivativeError` — that is a caller bug, not a property of the data.

## Unavailable and undefined

Three distinct outcomes, none of which is a fabricated number:

* `DerivativeStatus.UNAVAILABLE` — no trace at all: the channel is absent or
  empty, the grid has a duplicate or is non-monotonic, the requested aperture
  is unusable on this record, or there are too few samples (one for group
  delay, two for curvature). `reason` says which, and `points` is empty.
* `DerivativeStatus.PARTIAL` — the trace exists but some samples have no
  derivative, either because the sample has no phase (zero magnitude: `cmath`
  would invent 0.0 there) or because a neighbour needed by its stencil is
  missing. Those points carry `value = None`; `undefined_count` counts them and
  `describe()` reports them.
* `DerivativeStatus.OK` — every sample is defined.

`nan` and `inf` are rejected at construction (`PhaseDerivative.__post_init__`),
so a value that reaches a chart is always a real number that was computed.
`UNAVAILABLE` is not the only refusal: a trace that is numerically fine but
rests on the unwrapping assumption is reported as such via `wrap_ambiguous`
(see "Unwrapping"), because there the honest caveat is stronger than a status.

## Reading a result

* `median` is the median of the defined values (robust to a single notch),
  in the result's unit.
* `value_at(freq)` returns the value at exactly that measured frequency, or
  `None`; derivatives are never interpolated between samples.
* Each `DerivativePoint` carries the measured `freq` it belongs to and the
  `omega` its value was actually taken at; the two differ only for smoothed
  values that sit on their window's centroid.
* `window` and `describe()` name the aperture actually applied, how many
  values sit on window centroids, the wrap corrections, whether the curve was
  raw or smoothed, and for curvature that it is a diagnostic.
* `ambiguous_delay` is the grid's `pi / step_omega` limit and `wrap_ambiguous`
  says whether any value reached it, so a delay can be checked against the
  limit without recomputing the grid.
* `diagnostics()` returns group delay and curvature of the same samples with
  the same aperture, so the two can never be compared across different
  smoothing.

## Acceptance

`tests/test_phase_derivatives.py` checks every number against an analytic
reference written out in the test: a 3.7 ns delay line over 101 points
(3.7e-9 s to 1e-15 everywhere), constant phase (zero delay to 1e-18, zero
curvature), wrapped versus unwrapped delay, linear phase (zero curvature) and
`phi = K*omega**2/2` (curvature `K`, group delay `-K*omega`), an irregular
monotonic grid (both quantities, with the local steps), the aliasing limit
(`ambiguous_delay` 400 ns on that grid, flagged because the phase steps reach
5.29 rad > pi), and the refusals above.
