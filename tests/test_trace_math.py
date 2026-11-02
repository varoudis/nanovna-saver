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
"""ANALYSIS-02: complex memory-trace mathematics.

Every expected value is a closed-form expression written out here in the
test, not a re-run of the implementation:

* addition/subtraction are componentwise on ``(re, im)``;
* multiplication is ``(a+bi)(c+di) = (ac-bd) + (ad+bc)i``;
* division is ``(a+bi)/(c+di) = ((ac+bd) + (bc-ad)i)/(c^2+d^2)``.

Those are the textbook complex identities, computed below with plain floats,
so a bug in the module cannot be mirrored by the test.
"""

import cmath
import math
from dataclasses import replace

import pytest

from NanoVNASaver.Measurement import S11, S21, snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.TraceMath import (
    DERIVED_SOURCE_SEPARATOR,
    SourceAvailability,
    TraceMathStatus,
    TraceOperation,
    check_sources,
    compute_trace_math,
    derived_source_ids,
    plottable_points,
    recompute,
)
from NanoVNASaver.Traces import compare_traces

TOLERANCE = 1e-12


def _channel(freqs, reals, imags):
    return [
        Datapoint(freq, re, im)
        for freq, re, im in zip(freqs, reals, imags, strict=True)
    ]


def _snapshot(points, *, name="m", impedance=50.0, parameter=S11):
    kwargs = {parameter: points}
    return snapshot_from_datapoints(
        kwargs.get(S11),
        kwargs.get(S21),
        name=name,
        reference_impedance=impedance,
    )


# ----------------------------------------------------------------------
# Analytic references
# ----------------------------------------------------------------------
def _analytic_sum(a, b):
    """(ar + br) + (ai + bi)i -- componentwise."""
    (ar, ai), (br, bi) = a, b
    return (ar + br, ai + bi)


def _analytic_difference(a, b):
    """(ar - br) + (ai - bi)i -- componentwise."""
    (ar, ai), (br, bi) = a, b
    return (ar - br, ai - bi)


def _analytic_product(a, b):
    """(ar*br - ai*bi) + (ar*bi + ai*br)i."""
    (ar, ai), (br, bi) = a, b
    return (ar * br - ai * bi, ar * bi + ai * br)


def _analytic_quotient(a, b):
    """((ar*br + ai*bi) + (ai*br - ar*bi)i) / (br^2 + bi^2)."""
    (ar, ai), (br, bi) = a, b
    denominator = br * br + bi * bi
    return (
        (ar * br + ai * bi) / denominator,
        (ai * br - ar * bi) / denominator,
    )


#: Two arbitrary points with nothing special about them; the analytic helpers
#: above are the only source of truth for the expected results.
A = (0.37, -0.62)
B = (1.25, 0.75)


class TestAnalyticRoundTrips:
    @staticmethod
    @pytest.mark.parametrize(
        ("operation", "reference"),
        [
            (TraceOperation.ADD, _analytic_sum),
            (TraceOperation.SUBTRACT, _analytic_difference),
            (TraceOperation.MULTIPLY, _analytic_product),
            (TraceOperation.DIVIDE, _analytic_quotient),
        ],
    )
    def test_matches_closed_form(operation, reference):
        first = _snapshot(_channel([1_000_000], [A[0]], [A[1]]), name="a")
        second = _snapshot(_channel([1_000_000], [B[0]], [B[1]]), name="b")
        result = compute_trace_math(first, second, operation)
        expected_re, expected_im = reference(A, B)
        assert result.status is TraceMathStatus.OK
        assert result.used == 1
        point = result.points[0]
        assert point.defined
        assert point.value.real == pytest.approx(expected_re, abs=TOLERANCE)
        assert point.value.imag == pytest.approx(expected_im, abs=TOLERANCE)
        # The snapshot channel must carry the same numbers.
        derived = result.snapshot.channel(S11)
        assert derived[0].re == pytest.approx(expected_re, abs=TOLERANCE)
        assert derived[0].im == pytest.approx(expected_im, abs=TOLERANCE)

    @staticmethod
    def test_operations_differ_from_each_other():
        """Guards against every operation accidentally doing the same thing."""
        first = _snapshot(_channel([1], [A[0]], [A[1]]))
        second = _snapshot(_channel([1], [B[0]], [B[1]]))
        values = {
            operation: compute_trace_math(first, second, operation)
            .points[0]
            .value
            for operation in TraceOperation
        }
        assert len({round(value.real, 12) for value in values.values()}) == 4

    @staticmethod
    def test_division_is_not_reciprocal_of_multiplication():
        """(a/b) * b must return a, checked through the analytic quotient."""
        quotient = _analytic_quotient(A, B)
        recovered = _analytic_product(quotient, B)
        assert recovered[0] == pytest.approx(A[0], abs=1e-12)
        assert recovered[1] == pytest.approx(A[1], abs=1e-12)

    @staticmethod
    def test_channel_is_s21_when_requested():
        freqs = [1_000_000, 2_000_000]
        first = snapshot_from_datapoints(
            None,
            _channel(freqs, [0.5, 0.25], [0.5, -0.25]),
            name="a",
        )
        second = snapshot_from_datapoints(
            None,
            _channel(freqs, [0.5, 0.5], [0.0, 0.0]),
            name="b",
        )
        result = compute_trace_math(
            first, second, TraceOperation.ADD, channel=S21
        )
        assert result.channel == S21
        assert result.used == 2
        expected = _analytic_sum((0.5, 0.5), (0.5, 0.0))
        assert result.points[0].value.real == pytest.approx(
            expected[0], abs=TOLERANCE
        )
        assert result.points[0].value.imag == pytest.approx(
            expected[1], abs=TOLERANCE
        )

    @staticmethod
    def test_missing_channel_is_unavailable():
        first = _snapshot(_channel([1], [0.1], [0.0]))
        second = snapshot_from_datapoints(None, None, name="b")
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert result.status is TraceMathStatus.UNAVAILABLE
        assert "unavailable" in result.reason
        assert result.snapshot is None


class TestOverlapAndAlignment:
    @staticmethod
    def test_partial_overlap_uses_only_common_frequencies():
        freqs_a = [1_000_000, 2_000_000, 3_000_000]
        freqs_b = [2_000_000, 3_000_000, 4_000_000]
        first = _snapshot(
            _channel(freqs_a, [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]), name="a"
        )
        second = _snapshot(
            _channel(freqs_b, [0.05, 0.05, 0.05], [0.0, 0.0, 0.0]), name="b"
        )
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert result.status is TraceMathStatus.OK
        assert result.frequencies == (2_000_000, 3_000_000)
        assert result.overlap == 2
        assert result.used == 2
        assert result.dropped_first == 1
        assert result.dropped_second == 1
        # Hand-computed: 0.2 + 0.05 and 0.3 + 0.05.
        assert result.points[0].value.real == pytest.approx(0.25, abs=TOLERANCE)
        assert result.points[1].value.real == pytest.approx(0.35, abs=TOLERANCE)
        channel = result.snapshot.channel(S11)
        assert [dp.freq for dp in channel] == [2_000_000, 3_000_000]

    @staticmethod
    def test_partially_shifted_grid_drops_everything_off_grid():
        """No interpolation: one shared frequency out of four."""
        first = _snapshot(
            _channel(
                [1_000_000, 2_000_000, 3_000_000], [1.0, 1.0, 1.0], [0, 0, 0]
            )
        )
        second = _snapshot(
            _channel(
                [2_000_000, 5_000_000, 6_000_000], [2.0, 2.0, 2.0], [0, 0, 0]
            )
        )
        result = compute_trace_math(first, second, TraceOperation.MULTIPLY)
        assert result.frequencies == (2_000_000,)
        assert result.dropped_first == 2
        assert result.dropped_second == 2
        # Product of two purely real values.
        assert result.points[0].value == pytest.approx(complex(2.0, 0.0))

    @staticmethod
    def test_disjoint_grids_are_unavailable():
        first = _snapshot(_channel([1_000_000], [0.5], [0.5]))
        second = _snapshot(_channel([2_000_000], [0.5], [0.5]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert result.status is TraceMathStatus.UNAVAILABLE
        assert result.snapshot is None
        assert result.points == ()
        assert "no coincident frequencies" in result.reason
        assert "never interpolated" in result.reason

    @staticmethod
    def test_no_extrapolation_at_ends():
        """A narrower grid must not be stretched onto the wider one."""
        wide = _snapshot(
            _channel([1, 2, 3, 4, 5], [0.1] * 5, [0.0] * 5), name="wide"
        )
        narrow = _snapshot(
            _channel([2, 3, 4], [0.2] * 3, [0.0] * 3), name="narrow"
        )
        result = compute_trace_math(wide, narrow, TraceOperation.SUBTRACT)
        assert result.frequencies == (2, 3, 4)
        assert result.dropped_first == 2
        assert result.dropped_second == 0
        assert all(
            point.value.real == pytest.approx(-0.1, abs=TOLERANCE)
            for point in result.points
        )

    @staticmethod
    def test_describe_reports_counts_and_span():
        first = _snapshot(
            _channel([1_000_000, 2_000_000], [0.1, 0.2], [0.0, 0.0]), name="a"
        )
        second = _snapshot(
            _channel([2_000_000, 3_000_000], [0.1, 0.2], [0.0, 0.0]), name="b"
        )
        result = compute_trace_math(first, second, TraceOperation.ADD)
        text = result.describe()
        assert "S11 sum of a + b" in text
        assert "1 of 1 coincident points" in text
        assert "2 dropped as non-coincident" in text
        assert "2-2 MHz" in text

    @staticmethod
    def test_result_points_are_frequency_ordered():
        """Unsorted inputs must not produce an unsorted derived channel."""
        first = _snapshot(_channel([3, 1, 2], [0.1, 0.2, 0.3], [0, 0, 0]))
        second = _snapshot(_channel([2, 3, 1], [0.1, 0.1, 0.1], [0, 0, 0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert result.frequencies == (1, 2, 3)
        # Hand-computed sums by frequency: 0.2+0.1, 0.3+0.1, 0.1+0.1.
        assert [point.value.real for point in result.points] == pytest.approx(
            [0.3, 0.4, 0.2]
        )

    @staticmethod
    def test_duplicate_frequency_uses_last_occurrence():
        """Matches compare_traces: a repeated frequency resolves to the last."""
        first = _snapshot(
            [Datapoint(1, 0.1, 0.0), Datapoint(1, 0.5, 0.0)], name="dup"
        )
        second = _snapshot(_channel([1], [0.1], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert result.overlap == 1
        assert result.points[0].value.real == pytest.approx(0.6, abs=TOLERANCE)


class TestReferenceImpedance:
    @staticmethod
    def test_mismatched_impedance_is_refused_naming_both():
        first = _snapshot(_channel([1], [0.5], [0.0]), impedance=50.0)
        second = _snapshot(_channel([1], [0.5], [0.0]), impedance=75.0)
        for operation in TraceOperation:
            result = compute_trace_math(first, second, operation)
            assert result.status is TraceMathStatus.UNAVAILABLE
            assert result.snapshot is None
            assert "50" in result.reason
            assert "75" in result.reason
            assert "renormalize" in result.reason

    @staticmethod
    def test_matching_impedance_proceeds():
        first = _snapshot(_channel([1], [0.5], [0.0]), impedance=75.0)
        second = _snapshot(_channel([1], [0.25], [0.0]), impedance=75.0)
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        assert result.status is TraceMathStatus.OK
        assert result.points[0].value == pytest.approx(complex(2.0, 0.0))


class TestDivisionByZero:
    @staticmethod
    def test_zero_divisor_marks_one_point_undefined():
        freqs = [1, 2, 3]
        first = _snapshot(_channel(freqs, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))
        second = _snapshot(_channel(freqs, [2.0, 0.0, 4.0], [0.0, 0.0, 0.0]))
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        assert result.status is TraceMathStatus.OK
        assert result.overlap == 3
        assert result.used == 2
        undefined = result.undefined_points
        assert len(undefined) == 1
        assert undefined[0].freq == 2
        assert undefined[0].value is None
        assert "zero" in undefined[0].reason
        # Hand-computed quotients at the surviving frequencies.
        assert result.points[0].value == pytest.approx(complex(0.5, 0.0))
        assert result.points[2].value == pytest.approx(complex(0.25, 0.0))
        channel = result.snapshot.channel(S11)
        assert [dp.freq for dp in channel] == [1, 3]

    @staticmethod
    def test_no_nan_or_inf_reaches_the_snapshot():
        freqs = [1, 2]
        first = _snapshot(_channel(freqs, [1.0, 1.0], [0.0, 0.0]))
        second = _snapshot(_channel(freqs, [0.0, 0.0], [0.0, 0.0]))
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        assert result.status is TraceMathStatus.UNAVAILABLE
        assert result.snapshot is None
        assert "all 2 coincident points are undefined" in result.reason
        assert all(
            point.value is None and not point.defined for point in result.points
        )

    @staticmethod
    def test_every_point_finite_after_zero_division():
        freqs = list(range(1, 11))
        first = _snapshot(_channel(freqs, [1.0] * 10, [0.5] * 10))
        second = _snapshot(_channel(freqs, [0.0, 1.0] * 5, [0.0, 0.0] * 5))
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        assert result.snapshot is not None
        for point in result.snapshot.channel(S11):
            assert math.isfinite(point.re)
            assert math.isfinite(point.im)

    @staticmethod
    def test_non_finite_divisor_sample_is_undefined():
        freqs = [1, 2]
        first = _snapshot(_channel(freqs, [1.0, 1.0], [0.0, 0.0]))
        second = _snapshot(_channel(freqs, [math.nan, 2.0], [0.0, 0.0]))
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        assert result.used == 1
        assert result.undefined_points[0].freq == 1
        assert "not finite" in result.undefined_points[0].reason

    @staticmethod
    def test_near_zero_divisor_is_still_valid():
        """Only an exact zero is singular; small divisors are legitimate."""
        first = _snapshot(_channel([1], [1.0], [0.0]))
        second = _snapshot(_channel([1], [1e-9], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        assert result.status is TraceMathStatus.OK
        assert result.points[0].value.real == pytest.approx(1e9, rel=1e-12)


class TestDerivedProvenance:
    @staticmethod
    def test_result_is_derived_and_has_no_raw_data():
        first = _snapshot(_channel([1], [0.5], [0.5]), name="a")
        second = _snapshot(_channel([1], [0.25], [0.25]), name="b")
        assert first.raw and second.raw
        result = compute_trace_math(first, second, TraceOperation.ADD)
        snapshot = result.snapshot
        assert snapshot is not None
        assert snapshot.raw == {}
        assert snapshot.corrected
        assert snapshot.is_derived
        assert snapshot.provenance.derived_from == (
            f"{first.id}{DERIVED_SOURCE_SEPARATOR}{second.id}"
        )
        assert snapshot.id not in (first.id, second.id)
        assert snapshot.reference_impedance == 50.0

    @staticmethod
    def test_provenance_notes_record_operation_and_counts():
        first = _snapshot(_channel([1, 2], [0.5, 0.5], [0.0, 0.0]), name="a")
        second = _snapshot(_channel([2, 3], [0.5, 0.5], [0.0, 0.0]), name="b")
        result = compute_trace_math(first, second, TraceOperation.MULTIPLY)
        notes = result.snapshot.provenance.notes
        assert "ANALYSIS-02 multiply of S11" in notes
        assert f"sources: {first.id} * {second.id}" in notes
        assert "coincident points: 1 (used 1)" in notes
        assert "dropped as non-coincident: 1 from first, 1 from second" in notes
        assert "no interpolation" in notes

    @staticmethod
    def test_derived_snapshot_keeps_processing_provenance():
        first = _snapshot(_channel([1], [0.5], [0.0]), name="a")
        second = _snapshot(_channel([1], [0.5], [0.0]), name="b")
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert (
            result.snapshot.provenance.processing is first.provenance.processing
        )

    @staticmethod
    def test_derived_source_ids_round_trip():
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert derived_source_ids(result.snapshot) == (first.id, second.id)
        # An ordinary acquisition records no derived source.
        assert derived_source_ids(first) == ()


class TestSourceAvailability:
    @staticmethod
    def test_reproducible_while_both_sources_live():
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        availability = check_sources(result, [first.id, second.id])
        assert isinstance(availability, SourceAvailability)
        assert availability.reproducible
        assert availability.missing == ()
        assert len(plottable_points(result, [first.id, second.id])) == 1

    @staticmethod
    def test_deleted_source_makes_result_unrenderable():
        first = _snapshot(_channel([1], [0.5], [0.0]), name="a")
        second = _snapshot(_channel([1], [0.5], [0.0]), name="b")
        result = compute_trace_math(first, second, TraceOperation.ADD)
        availability = check_sources(result, [second.id])
        assert not availability.reproducible
        assert availability.missing == (first.id,)
        assert first.id in availability.reason
        assert plottable_points(result, [second.id]) == ()

    @staticmethod
    def test_replaced_source_is_not_reproducible():
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        replacement = _snapshot(_channel([1], [0.9], [0.0]))
        assert not check_sources(
            result, [replacement.id, second.id]
        ).reproducible
        assert check_sources(result, [replacement.id, second.id]).missing == (
            first.id,
        )

    @staticmethod
    def test_unavailable_result_is_reproducible_but_unplottable():
        """Disjoint grids are reproducible: replaying always refuses again."""
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([2], [0.5], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        assert result.status is TraceMathStatus.UNAVAILABLE
        assert check_sources(result, [first.id, second.id]).reproducible
        assert plottable_points(result, [first.id, second.id]) == ()
        replay = recompute(result, [first, second])
        assert replay.status is TraceMathStatus.UNAVAILABLE
        assert replay.reason == result.reason
        assert replay.snapshot is None

    @staticmethod
    def test_recompute_reproduces_with_live_sources():
        first = _snapshot(_channel([1, 2], [0.5, 0.25], [0.0, 0.0]), name="a")
        second = _snapshot(_channel([1, 2], [0.25, 0.5], [0.0, 0.0]), name="b")
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        again = recompute(result, [first, second])
        assert again.status is TraceMathStatus.OK
        assert again.frequencies == result.frequencies
        assert again.operation is TraceOperation.DIVIDE
        assert again.channel == result.channel
        # Same numbers, but a fresh derived snapshot (not the same object).
        assert again.snapshot.id != result.snapshot.id
        assert [point.value for point in again.points] == [
            point.value for point in result.points
        ]
        # Hand-computed: 0.5/0.25 = 2 at 1 Hz, 0.25/0.5 = 0.5 at 2 Hz.
        assert again.points[0].value == pytest.approx(complex(2.0, 0.0))
        assert again.points[1].value == pytest.approx(complex(0.5, 0.0))

    @staticmethod
    def test_recompute_with_deleted_source_is_unavailable():
        first = _snapshot(_channel([1], [0.5], [0.0]), name="a")
        second = _snapshot(_channel([1], [0.5], [0.0]), name="b")
        result = compute_trace_math(first, second, TraceOperation.ADD)
        again = recompute(result, [second])
        assert again.status is TraceMathStatus.UNAVAILABLE
        assert again.snapshot is None
        assert again.points == result.points
        assert first.id in again.reason


class TestDataLayer:
    @staticmethod
    def test_raw_layer_is_used_when_requested():
        freqs = [1]
        first = snapshot_from_datapoints(
            _channel(freqs, [0.9], [0.0]),
            None,
            name="a",
            raw11=_channel(freqs, [0.1], [0.0]),
        )
        second = snapshot_from_datapoints(
            _channel(freqs, [0.8], [0.0]),
            None,
            name="b",
            raw11=_channel(freqs, [0.2], [0.0]),
        )
        corrected = compute_trace_math(first, second, TraceOperation.ADD)
        raw = compute_trace_math(
            first, second, TraceOperation.ADD, corrected=False
        )
        # Corrected: 0.9 + 0.8.  Raw: 0.1 + 0.2.
        assert corrected.points[0].value.real == pytest.approx(
            1.7, abs=TOLERANCE
        )
        assert raw.points[0].value.real == pytest.approx(0.3, abs=TOLERANCE)
        assert raw.corrected is False
        assert "data layer: raw" in raw.snapshot.provenance.notes

    @staticmethod
    def test_derived_snapshot_survives_serialization():
        first = _snapshot(_channel([1], [0.5], [0.0]), name="a")
        second = _snapshot(_channel([1], [0.25], [0.0]), name="b")
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        restored = type(result.snapshot).from_dict(result.snapshot.to_dict())
        assert restored.raw == {}
        assert restored.is_derived
        assert restored.provenance.derived_from == (
            f"{first.id}{DERIVED_SOURCE_SEPARATOR}{second.id}"
        )
        assert restored.channel(S11)[0].re == pytest.approx(2.0)


class TestModuleContracts:
    @staticmethod
    def test_compare_traces_is_untouched_and_interpolating():
        """The display comparison still interpolates; these are not the same.

        ``compare_traces`` works on traces and may resample; complex memory
        math refuses.  This pins that the two entry points stay distinct.
        """
        from NanoVNASaver.Traces import Trace, TraceKind

        first = _snapshot(_channel([1, 2, 4], [1.0, 1.0, 1.0], [0, 0, 0]))
        second = _snapshot(_channel([2, 3, 4], [0.5, 0.5, 0.5], [0, 0, 0]))
        comparison = compare_traces(
            Trace(id="a", name="a", snapshot=first, kind=TraceKind.FROZEN),
            Trace(id="b", name="b", snapshot=second, kind=TraceKind.FROZEN),
            parameter=S11,
        )
        # The display comparison resamples 3 Hz onto the reference grid; the
        # complex memory math refuses to and combines only 2 Hz and 4 Hz.
        assert comparison.interpolated
        assert comparison.frequencies() == [2, 3, 4]
        memory = compute_trace_math(first, second, TraceOperation.SUBTRACT)
        assert memory.frequencies == (2, 4)

    @staticmethod
    def test_unsupported_parameter_raises():
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        with pytest.raises(ValueError, match="unsupported parameter"):
            compute_trace_math(first, second, TraceOperation.ADD, channel="S99")

    @staticmethod
    def test_unsupported_operation_raises():
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        with pytest.raises(TypeError, match="unsupported operation"):
            compute_trace_math(first, second, "add")

    @staticmethod
    def test_ambiguous_source_id_raises():
        """Ids must be encodable in ``derived_from`` without ambiguity."""
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        ambiguous = replace(first, id=f"a{DERIVED_SOURCE_SEPARATOR}b")
        with pytest.raises(ValueError, match="reserved separator"):
            compute_trace_math(ambiguous, second, TraceOperation.ADD)
        with pytest.raises(ValueError, match="must carry an id"):
            compute_trace_math(
                replace(first, id=""), second, TraceOperation.ADD
            )

    @staticmethod
    def test_operation_symbols_and_labels():
        assert [op.symbol for op in TraceOperation] == ["+", "-", "*", "/"]
        assert TraceOperation.DIVIDE.label == "quotient"

    @staticmethod
    def test_result_is_immutable():
        first = _snapshot(_channel([1], [0.5], [0.0]))
        second = _snapshot(_channel([1], [0.5], [0.0]))
        result = compute_trace_math(first, second, TraceOperation.ADD)
        with pytest.raises(AttributeError):
            result.status = TraceMathStatus.UNAVAILABLE
        with pytest.raises(AttributeError):
            result.points[0].freq = 5

    @staticmethod
    def test_division_matches_cmath_reference_for_arbitrary_values():
        """Second, independent reference for one non-trivial quotient."""
        a = complex(0.37, -0.62)
        b = complex(1.25, 0.75)
        first = _snapshot(_channel([7_100_000], [a.real], [a.imag]))
        second = _snapshot(_channel([7_100_000], [b.real], [b.imag]))
        result = compute_trace_math(first, second, TraceOperation.DIVIDE)
        expected = cmath.exp(cmath.log(a) - cmath.log(b))
        assert abs(result.points[0].value - expected) < TOLERANCE
