#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020 Rune B. Broberg
#  Copyright (C) 2020ff NanoVNA-Saver Authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""TDR-01: time-domain gating, checked against closed-form reflections.

Every expected number below is computed from the physics, not from the
implementation:

* a reflection of coefficient ``G`` at round-trip delay ``tau`` is the
  frequency response ``G * exp(-j*2*pi*f*tau)``; converting ``tau`` to a
  one-way distance is ``d = v * c * tau / 2``.  The transform has to put the
  impulse there, within one sample spacing.
* the forward transform is a discrete inverse Fourier transform of a
  Hermitian spectrum and the inverse is the matching forward DFT, so the
  identity gate must reproduce the input *exactly* up to rounding; that is
  asserted at ``1e-9`` (measured: ~1e-16).
* removing a band around a known echo must remove that echo's frequency
  component while leaving the rest of the trace alone.
"""

import math

import numpy as np
import pytest

from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.TDRGating import (
    C_LIGHT,
    DEFAULT_WINDOW,
    MIN_SAMPLES,
    GatedResult,
    GateKind,
    TDRGatingError,
    TimeDomainResult,
    apply_gate,
    gate_points,
    to_frequency_domain,
    to_time_domain,
)

#: 1 Hz resolution would be a 1 s record; 1 MHz gives a 1 us record, whose
#: unambiguous one-way range is ``v*c/(4*df)`` = ~49 m at v = 0.66.
DF = 1_000_000
VELOCITY = 0.66
PAD = 4


def _record_length(samples: int, pad: int = PAD) -> int:
    """Length of the circular record, ``2 * n * pad - 1`` bins."""
    return 2 * samples * pad - 1


def _delay_of(index: int, samples: int, pad: int = PAD) -> float:
    """Round-trip delay, in seconds, of time sample ``index``."""
    return index / (_record_length(samples, pad) * DF)


def _echo(samples: int, coefficient: float, delay: float) -> list[Datapoint]:
    """Uniform sweep from DC of ``coefficient * exp(-j*2*pi*f*delay)``."""
    return [
        Datapoint(
            index * DF,
            coefficient * math.cos(-2 * math.pi * index * DF * delay),
            coefficient * math.sin(-2 * math.pi * index * DF * delay),
        )
        for index in range(samples)
    ]


def _sum_of_echoes(
    samples: int, terms: list[tuple[float, float]]
) -> list[Datapoint]:
    """Sum of ``(coefficient, delay)`` echoes, evaluated analytically."""
    points = []
    for index in range(samples):
        freq = index * DF
        total = sum(
            coefficient
            * complex(
                math.cos(-2 * math.pi * freq * delay),
                math.sin(-2 * math.pi * freq * delay),
            )
            for coefficient, delay in terms
        )
        points.append(Datapoint(freq, total.real, total.imag))
    return points


def _one_way(delay: float, velocity: float = VELOCITY) -> float:
    """One-way distance, metres, of a round-trip delay."""
    return 0.5 * velocity * C_LIGHT * delay


def _residual(trace, expected) -> float:
    """Largest magnitude difference between two traces, sample by sample."""
    return max(
        abs(complex(point.re, point.im) - reference)
        for point, reference in zip(trace, expected, strict=True)
    )


# ----------------------------------------------------------------------
# Geometry of the transform
# ----------------------------------------------------------------------
def test_record_length_and_step_follow_the_stated_convention():
    samples = 201
    result = to_time_domain(
        _echo(samples, 0.5, 0.0), velocity_factor=VELOCITY, pad_factor=PAD
    )

    # L = 2*n*pad - 1 bins; dt = 1/(L*df); d = v*c*dt/2.
    assert result.sample_count == 2 * samples * PAD - 1 == 1607
    assert result.time_step == pytest.approx(1 / (1607 * DF), rel=1e-15)
    assert result.distance_step == pytest.approx(
        0.5 * VELOCITY * C_LIGHT / (1607 * DF), rel=1e-15
    )
    # 0.5 * 0.66 * c / (1607 * 1e6) = 0.061562856963285624 m.
    assert result.distance_step == pytest.approx(
        0.061562856963285624, rel=1e-12
    )
    # Unambiguous one-way range is a quarter wavelength of the step:
    # v*c/(4*df) = 49.46575557 m at v = 0.66, df = 1 MHz.
    assert result.max_distance == pytest.approx(49.46575557, rel=1e-9)
    assert result.max_distance == pytest.approx(VELOCITY * C_LIGHT / (4 * DF))


def test_metadata_names_the_transform_and_the_processing():
    samples = 201
    result = to_time_domain(
        _echo(samples, 0.5, 0.0), channel="S11", window="hann"
    )
    provenance = result.provenance()

    assert result.channel == "S11"
    assert result.input_samples == samples
    assert result.frequencies[0] == 0
    assert result.frequencies[-1] == (samples - 1) * DF
    assert provenance["window"] == "hann"
    assert provenance["pad_factor"] == 1
    assert provenance["velocity_factor"] == 1.0
    assert provenance["derived"] is True
    assert provenance["transform"] == "tdr_impulse_lowpass"
    assert "Hermitian" in provenance["convention"]
    assert "v*c*t/2" in provenance["convention"]


# ----------------------------------------------------------------------
# A known delayed reflection
# ----------------------------------------------------------------------
def test_single_echo_lands_at_its_hand_computed_distance():
    # 1.5 m one-way at v = 0.66:  tau = 2*d/(v*c) = 2*1.5/(0.66*c) = 15.162 ns.
    distance = 1.5
    delay = 2 * distance / (VELOCITY * C_LIGHT)
    assert delay == pytest.approx(1.516200432718873e-08, rel=1e-12)

    samples = 801
    result = to_time_domain(
        _echo(samples, 0.5, delay), velocity_factor=VELOCITY, pad_factor=PAD
    )
    peak = result.peak
    assert peak is not None
    assert peak.distance == pytest.approx(distance, abs=result.distance_step)
    # It really is one sample spacing, not a lucky wider tolerance:
    assert abs(peak.distance - distance) < result.distance_step
    assert abs(peak.distance - distance) == pytest.approx(
        0.0022074948368970837, abs=1e-9
    )
    assert peak.value == pytest.approx(0.5, abs=0.01)
    # Everything far from the echo is small: a single reflection, not a smear.
    far = [
        sample.value
        for sample in result.samples
        if abs(sample.distance - distance) > 1.0
    ]
    assert max(abs(value) for value in far) < 0.03


def test_echo_distance_scales_with_the_velocity_factor():
    # Same measured sweep, two dielectrics: the reported distance scales by
    # v exactly, because v enters only on the distance axis.
    samples = 801
    delay = 2 * 1.5 / (VELOCITY * C_LIGHT)
    points = _echo(samples, 0.5, delay)

    slow = to_time_domain(points, velocity_factor=0.66, pad_factor=PAD)
    fast = to_time_domain(points, velocity_factor=0.88, pad_factor=PAD)
    assert slow.peak is not None
    assert fast.peak is not None
    # The sample index is unchanged; only its distance label moved.
    assert slow.peak.time == pytest.approx(fast.peak.time, rel=1e-15)
    assert fast.peak.distance == pytest.approx(
        slow.peak.distance * 0.88 / 0.66, rel=1e-12
    )


def test_record_is_real_and_its_mean_returns_the_dc_sample():
    # A Hermitian spectrum gives a real record, and the inverse relation
    #   A[m] = (S/L) * sum_k x[k] e^(-j*2*pi*m*k/L),   S = 2*n - 1 (rect)
    # evaluated at m = 0 gives  gamma(0) = S * mean(x).  Here S is derived
    # from the physics of a one-sided sweep of n bins, not read back from the
    # implementation.
    samples = 201
    points = _sum_of_echoes(samples, [(0.5, 0.0), (0.2, 3e-9)])
    result = to_time_domain(points, pad_factor=PAD)
    values = np.asarray(result.values)
    assert np.isrealobj(values)
    assert np.isrealobj(np.asarray(result.samples[0].value))
    two_sided = 2 * samples - 1
    assert two_sided * values.mean() == pytest.approx(0.7, abs=1e-9)


# ----------------------------------------------------------------------
# Identity gate round trip: the strongest check that fwd/inverse are inverse
# ----------------------------------------------------------------------
def test_identity_gate_round_trips_the_measured_samples():
    samples = 201
    points = _sum_of_echoes(
        samples, [(0.4, 2.5e-9), (0.25, 11.3e-9), (-0.1, 0.0)]
    )
    result = to_time_domain(points, velocity_factor=VELOCITY, pad_factor=PAD)
    gated = apply_gate(result)
    assert gated.is_identity
    assert all(weight == 1.0 for weight in gated.gate)

    back = to_frequency_domain(gated)
    assert isinstance(back, tuple)
    assert len(back) == samples
    assert [point.freq for point in back] == [point.freq for point in points]
    # Measured: ~5.8e-16.  The stated tolerance is 1e-9.
    assert _residual(back, [point.z for point in points]) < 1e-9


@pytest.mark.parametrize("window", ["rectangular", "hann", "blackman"])
def test_identity_gate_round_trips_the_windowed_samples(window):
    samples = 201
    points = _sum_of_echoes(samples, [(0.5, 4.0e-9), (0.2, 0.0)])
    result = to_time_domain(
        points, window=window, velocity_factor=VELOCITY, pad_factor=PAD
    )
    back = to_frequency_domain(apply_gate(result))
    weights = np.asarray(
        {
            "rectangular": np.ones(samples),
            "hann": np.hanning(samples),
            "blackman": np.blackman(samples),
        }[window]
    )
    expected = [
        point.z * weight for point, weight in zip(points, weights, strict=True)
    ]
    assert _residual(back, expected) < 1e-9


def test_identity_gate_with_every_pad_factor_round_trips():
    samples = 101
    points = _echo(samples, 0.3, 6.0e-9)
    for pad in (1, 2, 3, 8):
        result = to_time_domain(points, pad_factor=pad)
        back = to_frequency_domain(apply_gate(result))
        assert _residual(back, [point.z for point in points]) < 1e-9, pad


def test_taper_holds_an_on_grid_echo_and_suppresses_its_sidelobes():
    # The two-sided normalisation S = 2*sum(w) - w[0] divides out the taper's
    # weight on the impulse, so an echo that lands on a record sample reads
    # its true coefficient G under ANY window.  This is exact, not empirical:
    # for an on-grid echo the record is x[k0] = G*S/L, so multiplying by L/S
    # returns G.  What the taper actually changes is the sidelobes.
    samples = 401
    delay = _delay_of(40, samples)
    points = _echo(samples, 0.8, delay)

    plain = to_time_domain(points, pad_factor=PAD)
    tapered = to_time_domain(points, window="hann", pad_factor=PAD)
    assert plain.peak is not None
    assert tapered.peak is not None
    assert plain.peak.value == pytest.approx(0.8, abs=1e-9)
    assert tapered.peak.value == pytest.approx(0.8, abs=1e-9)
    # mean(np.hanning(n)) = (n-1)/(2n) = 0.5 - 1/(2n), recorded for provenance.
    assert tapered.window_coherent_gain == pytest.approx(
        (samples - 1) / (2 * samples), rel=1e-12
    )
    assert plain.window_coherent_gain == 1.0
    # The taper does not move the echo.
    assert tapered.peak.time == pytest.approx(plain.peak.time, rel=1e-12)
    # Away from the echo the taper decides how fast the leakage decays.  The
    # Dirichlet kernel of a rectangular record has an f^-1 envelope, so its
    # sidelobes are flat in amplitude; a Hann record is a sum of three
    # Dirichlet kernels whose leading terms cancel, giving f^-3.  Quadrupling
    # the offset therefore leaves the rectangular leakage unchanged (ratio
    # ~1) while the Hann leakage drops by ~4^3 = 64.  Assert the *trend*,
    # which is the closed-form difference, not a fitted constant.
    near, far = 2 * 40, 8 * 40
    rectangular_ratio = abs(plain.values[far]) / abs(plain.values[near])
    hann_ratio = abs(tapered.values[far]) / abs(tapered.values[near])
    assert rectangular_ratio > 0.9
    assert hann_ratio < 1.0 / 16.0
    # And the tapered peak still stands clear of its own leakage.
    assert tapered.peak.value > 100 * max(
        abs(value) for value in tapered.values[far:]
    )


# ----------------------------------------------------------------------
# Removing an echo
# ----------------------------------------------------------------------
def test_bandstop_gate_removes_the_echo_and_keeps_the_rest():
    # A 0.5 echo at 1.5 m one-way (round trip 3 m) plus a constant 0.2
    # reflection from the reference plane.  Choosing df = 1 MHz and 801
    # samples makes the echo land near a sample, so the gate can separate it.
    samples = 801
    delay = 2 * 1.5 / (VELOCITY * C_LIGHT)
    points = _sum_of_echoes(samples, [(0.2, 0.0), (0.5, delay)])

    result = to_time_domain(
        points, window="hann", velocity_factor=VELOCITY, pad_factor=PAD
    )
    assert result.peak is not None
    assert result.peak.distance == pytest.approx(1.5, abs=0.01)

    gated = apply_gate(
        result,
        kind=GateKind.BANDSTOP,
        start_distance=1.2,
        stop_distance=1.8,
        edge=0.3,
    )
    assert gated.derived
    assert gated.kind is GateKind.BANDSTOP
    # The removed band is exactly where it says it is.
    for sample, weight in zip(
        gated.samples, gated.gate[: len(gated.samples)], strict=True
    ):
        if 1.5 - 0.05 <= sample.distance <= 1.5 + 0.05:
            assert weight < 1e-3

    back = to_frequency_domain(gated)
    weights = np.hanning(samples)
    # What survives is the tapered constant reflection at the reference
    # plane; the echo is gone.  Measured residual 6.8e-3.
    survivor = [0.2 * weight for weight in weights]
    assert _residual(back, survivor) < 0.01
    # Against the *original* trace the difference is the whole echo, so the
    # gate demonstrably did something.
    assert _residual(back, [point.z for point in points]) > 0.5


def test_bandstop_gate_leaves_a_distant_echo_intact():
    samples = 401
    near = _delay_of(40, samples)
    far = _delay_of(260, samples)
    points = _sum_of_echoes(samples, [(0.5, near), (0.4, far)])

    result = to_time_domain(points, velocity_factor=VELOCITY, pad_factor=PAD)
    far_distance = _one_way(far)
    gated = apply_gate(
        result,
        kind=GateKind.BANDSTOP,
        start_distance=_one_way(near) - 0.3,
        stop_distance=_one_way(near) + 0.3,
        edge=0.15,
    )
    gated_far = [
        sample.value
        for sample in gated.samples
        if abs(sample.distance - far_distance) < result.distance_step
    ]
    original_far = [
        sample.value
        for sample in result.samples
        if abs(sample.distance - far_distance) < result.distance_step
    ]
    assert gated_far
    assert max(abs(value) for value in gated_far) == pytest.approx(
        max(abs(value) for value in original_far), rel=1e-9
    )
    # ... and the near echo is gone from the gated record.
    gated_near = [
        sample.value
        for sample in gated.samples
        if abs(sample.distance - _one_way(near)) < result.distance_step
    ]
    assert max(abs(value) for value in gated_near) < 1e-12


def test_keep_gate_keeps_only_the_named_band():
    samples = 401
    near = _delay_of(40, samples)
    far = _delay_of(260, samples)
    points = _sum_of_echoes(samples, [(0.5, near), (0.4, far)])
    result = to_time_domain(points, velocity_factor=VELOCITY, pad_factor=PAD)

    gated = apply_gate(
        result,
        kind=GateKind.KEEP,
        start_distance=_one_way(near) - 0.2,
        stop_distance=_one_way(near) + 0.2,
        edge=0.1,
    )
    by_distance = {
        round(sample.distance, 12): abs(sample.value)
        for sample in gated.samples
    }
    assert by_distance[round(_one_way(far), 12)] < 1e-12
    near_samples = [
        abs(sample.value)
        for sample in gated.samples
        if abs(sample.distance - _one_way(near)) < 0.05
    ]
    assert max(near_samples) > 0.4


def test_gate_weights_are_raised_cosine_over_the_stated_edge():
    samples = 401
    result = to_time_domain(_echo(samples, 0.5, 0.0), pad_factor=PAD)
    start, stop, edge = 2.0, 6.0, 1.0
    gated = apply_gate(
        result, start_distance=start, stop_distance=stop, edge=edge
    )
    assert gated.shape == "raised_cosine"

    for sample, weight in zip(
        gated.samples, gated.gate[: len(gated.samples)], strict=True
    ):
        distance = sample.distance
        half = edge / 2
        if distance <= start - half or distance >= stop + half:
            expected = 0.0
        elif start + half <= distance <= stop - half:
            expected = 1.0
        elif distance < start + half:
            # Hann rise from start-half to start+half.
            t = (distance - (start - half)) / edge
            expected = 0.5 * (1 - math.cos(math.pi * t))
        else:
            t = ((stop + half) - distance) / edge
            expected = 0.5 * (1 - math.cos(math.pi * t))
        assert weight == pytest.approx(expected, abs=1e-12)
        if expected not in (0.0, 1.0):
            assert 0.0 < weight < 1.0


def test_rectangular_gate_is_documented_by_its_zero_edge():
    samples = 201
    result = to_time_domain(_echo(samples, 0.5, 0.0), pad_factor=PAD)
    gated = apply_gate(result, start_distance=1.0, stop_distance=2.0)
    assert gated.edge == 0.0
    assert gated.shape == "rectangular"
    weights = {round(weight, 12) for weight in gated.gate}
    assert weights == {0.0, 1.0}
    # Only two values: no transition, hence the ringing a caller accepts.
    assert gated.label.startswith("S11 gated (keep 1.00-2.00 m")


# ----------------------------------------------------------------------
# Provenance: a gated trace is processed, never raw
# ----------------------------------------------------------------------
def test_gated_result_carries_every_gate_parameter():
    samples = 201
    result = to_time_domain(
        _echo(samples, 0.5, 2e-9), velocity_factor=0.7, pad_factor=2
    )
    gated = apply_gate(
        result,
        kind="bandstop",
        start_distance=0.5,
        stop_distance=3.0,
        edge=0.4,
    )
    provenance = gated.provenance()

    assert isinstance(gated, GatedResult)
    assert provenance["derived"] is True
    assert provenance["gate_kind"] == "bandstop"
    assert provenance["gate_start_distance"] == 0.5
    assert provenance["gate_stop_distance"] == 3.0
    assert provenance["gate_edge"] == 0.4
    assert provenance["gate_shape"] == "raised_cosine"
    assert provenance["gate_identity"] is False
    assert provenance["window"] == DEFAULT_WINDOW
    assert provenance["pad_factor"] == 2
    assert provenance["velocity_factor"] == 0.7
    assert provenance["label"] == gated.label
    assert "bandstop" in gated.label


def test_gate_points_is_the_shortcut_for_the_same_result():
    samples = 201
    points = _sum_of_echoes(samples, [(0.3, 1e-9), (0.2, 0.0)])
    direct = to_frequency_domain(
        apply_gate(
            to_time_domain(points, velocity_factor=VELOCITY, pad_factor=PAD),
            kind="bandstop",
            start_distance=0.15,
            stop_distance=0.45,
            edge=0.1,
        )
    )
    shortcut = gate_points(
        points,
        velocity_factor=VELOCITY,
        pad_factor=PAD,
        kind="bandstop",
        start_distance=0.15,
        stop_distance=0.45,
        edge=0.1,
    )
    assert shortcut == direct


# ----------------------------------------------------------------------
# Refusals: each names its reason
# ----------------------------------------------------------------------
def test_too_few_samples_is_refused_with_a_reason():
    with pytest.raises(TDRGatingError) as caught:
        to_time_domain(_echo(MIN_SAMPLES - 1, 0.5, 0.0))
    assert f"fewer than the {MIN_SAMPLES} required" in str(caught.value)


def test_empty_sweep_is_refused():
    with pytest.raises(TDRGatingError, match="fewer than the"):
        to_time_domain([])


def test_non_uniform_spacing_is_refused():
    samples = 201
    points = _echo(samples, 0.5, 0.0)
    broken = list(points)
    # Bias one sample by half a step: the axis still increases, so the
    # monotonic check passes and the uniformity check is the one that fires.
    broken[100] = Datapoint(int(broken[100].freq + 0.5 * DF), *broken[100][1:])
    with pytest.raises(TDRGatingError) as caught:
        to_time_domain(broken)
    message = str(caught.value)
    assert "not uniformly spaced" in message
    assert "largest step deviation" in message


def test_sweep_that_does_not_start_at_dc_is_refused():
    samples = 201
    points = [
        Datapoint(5 * DF + index * DF, 0.5, 0.0) for index in range(samples)
    ]
    with pytest.raises(TDRGatingError) as caught:
        to_time_domain(points)
    message = str(caught.value)
    assert "does not start at DC" in message
    assert "tolerance" in message


def test_a_sweep_just_off_dc_is_accepted():
    # Floor division of the sample rate can put the first bin a little above
    # zero; half a step of slack is the documented limit.
    samples = 201
    points = [Datapoint(index * DF + 1, 0.5, 0.0) for index in range(samples)]
    result = to_time_domain(points)
    assert result.input_samples == samples


def test_non_finite_sample_is_refused():
    points = _echo(64, 0.5, 0.0)
    points[10] = Datapoint(points[10].freq, math.inf, 0.0)
    with pytest.raises(TDRGatingError, match="non-finite"):
        to_time_domain(points)


def test_duplicate_frequencies_are_refused():
    points = _echo(64, 0.5, 0.0)
    points[3] = Datapoint(points[2].freq, 0.5, 0.0)
    with pytest.raises(TDRGatingError, match="not strictly increasing"):
        to_time_domain(points)


@pytest.mark.parametrize("velocity", [0.0, -0.5, 1.5, math.nan])
def test_impossible_velocity_factor_is_refused(velocity):
    with pytest.raises(TDRGatingError) as caught:
        to_time_domain(_echo(64, 0.5, 0.0), velocity_factor=velocity)
    assert "velocity factor" in str(caught.value)


@pytest.mark.parametrize("pad", [0, -4])
def test_non_positive_pad_factor_is_refused(pad):
    with pytest.raises(TDRGatingError, match="pad_factor"):
        to_time_domain(_echo(64, 0.5, 0.0), pad_factor=pad)


def test_unknown_window_is_refused_and_lists_the_supported_ones():
    with pytest.raises(TDRGatingError) as caught:
        to_time_domain(_echo(64, 0.5, 0.0), window="tukey")
    message = str(caught.value)
    assert "unknown window" in message
    assert "rectangular" in message and "hann" in message


def test_start_not_below_stop_is_refused():
    result = to_time_domain(_echo(201, 0.5, 0.0), pad_factor=PAD)
    with pytest.raises(TDRGatingError) as caught:
        apply_gate(result, start_distance=5.0, stop_distance=5.0)
    assert "not below" in str(caught.value)

    with pytest.raises(TDRGatingError, match="not below"):
        apply_gate(result, start_distance=6.0, stop_distance=5.0)


def test_gate_entirely_outside_the_record_is_refused():
    result = to_time_domain(_echo(201, 0.5, 0.0), pad_factor=PAD)
    beyond = result.max_distance + result.distance_step
    with pytest.raises(TDRGatingError) as caught:
        apply_gate(result, start_distance=beyond, stop_distance=beyond + 5)
    message = str(caught.value)
    assert "outside the record" in message
    assert f"{result.max_distance:g}" in message

    with pytest.raises(TDRGatingError) as caught:
        apply_gate(
            result,
            start_distance=1.0,
            stop_distance=result.max_distance + 1.0,
        )
    assert "past the end of the record" in str(caught.value)


def test_negative_and_inverted_gate_bounds_are_refused():
    result = to_time_domain(_echo(201, 0.5, 0.0), pad_factor=PAD)
    with pytest.raises(TDRGatingError, match="negative"):
        apply_gate(result, start_distance=-1.0, stop_distance=2.0)
    with pytest.raises(TDRGatingError, match="negative"):
        apply_gate(result, start_distance=1.0, stop_distance=2.0, edge=-0.5)
    with pytest.raises(TDRGatingError, match="wider than the"):
        apply_gate(result, start_distance=1.0, stop_distance=2.0, edge=1.5)


def test_unknown_gate_kind_is_refused():
    result = to_time_domain(_echo(201, 0.5, 0.0), pad_factor=PAD)
    with pytest.raises(TDRGatingError) as caught:
        apply_gate(result, kind="notch")
    assert "unknown gate kind" in str(caught.value)


def test_gate_stop_defaults_to_the_end_of_the_record():
    result = to_time_domain(_echo(201, 0.5, 0.0), pad_factor=PAD)
    gated = apply_gate(result, start_distance=0.0)
    assert gated.stop_distance == result.max_distance
    assert gated.is_identity
    assert isinstance(result, TimeDomainResult)
