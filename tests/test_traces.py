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
"""TRACE-01/TRACE-03: trace collection independence and comparison maths."""

import math

import pytest

from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Traces import (
    ComparisonCache,
    ComparisonKind,
    TraceCollection,
    TraceKind,
    compare_traces,
    marker_delta,
)


def _circle(freqs, radius=0.5):
    return [
        Datapoint(
            freq, radius * math.cos(freq / 1e6), radius * math.sin(freq / 1e6)
        )
        for freq in freqs
    ]


def _flat(freqs, value=0.1):
    return [Datapoint(freq, value, 0.0) for freq in freqs]


def _snapshot(s11, s21=None, name="m", impedance=50.0):
    return snapshot_from_datapoints(
        s11, s21, name=name, reference_impedance=impedance
    )


class TestTraceCollection:
    @staticmethod
    def test_capturing_does_not_overwrite_other_traces():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100]), name="a"), name="one")
        second = collection.add(
            _snapshot(_flat([100], 0.2), name="b"), name="two"
        )
        assert len(collection) == 2
        assert collection.get(first.id).channel()[0].re == 0.1
        assert collection.get(second.id).channel()[0].re == 0.2

    @staticmethod
    def test_live_updates_never_touch_frozen_traces():
        collection = TraceCollection()
        frozen = collection.add(
            _snapshot(_flat([100]), name="f"), name="frozen"
        )
        collection.add(
            _snapshot(_flat([100], 0.3), name="live"), kind=TraceKind.LIVE
        )
        collection.update_live(_snapshot(_flat([100], 0.9), name="live2"))
        assert collection.get(frozen.id).channel()[0].re == 0.1

    @staticmethod
    def test_live_trace_is_created_then_reused():
        collection = TraceCollection()
        collection.update_live(_snapshot(_flat([100])))
        assert len(collection) == 1
        collection.update_live(_snapshot(_flat([100], 0.4)))
        assert len(collection) == 1
        assert collection.live_trace.channel()[0].re == 0.4

    @staticmethod
    def test_ids_are_unique():
        collection = TraceCollection()
        ids = {
            collection.add(_snapshot(_flat([100])), trace_id="dup").id
            for _ in range(3)
        }
        assert len(ids) == 3
        assert len(collection) == 3

    @staticmethod
    def test_first_trace_becomes_comparison_reference():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        assert collection.reference is first
        second = collection.add(_snapshot(_flat([100])))
        collection.set_reference(second.id)
        assert collection.reference is second

    @staticmethod
    def test_removing_the_reference_selects_another():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        second = collection.add(_snapshot(_flat([100])))
        collection.remove(first.id)
        assert collection.reference is second

    @staticmethod
    def test_removing_the_last_reference_clears_it():
        collection = TraceCollection()
        only = collection.add(_snapshot(_flat([100])))
        collection.remove(only.id)
        assert collection.reference is None

    @staticmethod
    def test_reorder_updates_positions():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        second = collection.add(_snapshot(_flat([100])))
        collection.reorder(second.id, 0)
        assert collection.ordered_ids() == [second.id, first.id]
        assert collection.get(second.id).order == 0

    @staticmethod
    def test_visibility_and_colour_are_per_trace():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        second = collection.add(_snapshot(_flat([100])))
        collection.set_visible(first.id, False)
        collection.set_color(second.id, "#000000")
        assert collection.visible() == [second]
        assert collection.get(second.id).color == "#000000"

    @staticmethod
    def test_rename_rejects_empty_names():
        collection = TraceCollection()
        trace = collection.add(_snapshot(_flat([100])))
        collection.rename(trace.id, "Antenna")
        assert collection.get(trace.id).name == "Antenna"
        with pytest.raises(ValueError):
            collection.rename(trace.id, "  ")

    @staticmethod
    def test_frozen_trace_is_independent_of_later_snapshot_mutation():
        collection = TraceCollection()
        trace = collection.add(_snapshot(_flat([100]), name="frozen"))
        collection.update_live(_snapshot(_flat([100], 0.7)))
        assert trace.channel()[0].re == 0.1

    @staticmethod
    def test_several_traces_can_be_compared():
        collection = TraceCollection()
        for index in range(4):
            collection.add(
                _snapshot(_circle([1e6, 2e6, 3e6]), name=f"t{index}"),
                name=f"Trace {index}",
            )
        assert len(collection) == 4
        assert len(collection.comparison_candidates()) == 4

    @staticmethod
    def test_candidates_exclude_traces_without_the_channel():
        collection = TraceCollection()
        collection.add(_snapshot(_flat([100])), name="one-port")
        collection.add(_snapshot(_flat([100]), _flat([100])), name="two-port")
        assert len(collection.comparison_candidates("S11")) == 2
        assert len(collection.comparison_candidates("S21")) == 1


class TestComparisonMath:
    @staticmethod
    def test_identical_traces_differ_by_zero():
        collection = TraceCollection()
        data = _circle([1e6, 2e6, 3e6, 4e6])
        first = collection.add(_snapshot(data))
        second = collection.add(_snapshot(data))
        for kind in ComparisonKind:
            result = compare_traces(first, second, kind=kind)
            assert result.available
            for value in result.values():
                assert value == pytest.approx(0.0, abs=1e-12)

    @staticmethod
    def test_db_subtraction_is_a_ratio_not_a_vector_difference():
        collection = TraceCollection()
        loud = collection.add(_snapshot([Datapoint(100, 0.5, 0.0)]))
        quiet = collection.add(_snapshot([Datapoint(100, 0.25, 0.0)]))
        result = compare_traces(loud, quiet, kind=ComparisonKind.MAGNITUDE_DB)
        # 20*log10(0.5/0.25) == 6.02 dB
        assert result.values()[0] == pytest.approx(6.0206, abs=1e-3)
        vector = compare_traces(loud, quiet, kind=ComparisonKind.COMPLEX)
        assert vector.values()[0] == pytest.approx(0.25)

    @staticmethod
    def test_phase_wraps_at_the_boundary():
        collection = TraceCollection()
        plus = collection.add(
            _snapshot(
                [
                    Datapoint(
                        100,
                        math.cos(math.radians(179)),
                        math.sin(math.radians(179)),
                    )
                ]
            )
        )
        minus = collection.add(
            _snapshot(
                [
                    Datapoint(
                        100,
                        math.cos(math.radians(-179)),
                        math.sin(math.radians(-179)),
                    )
                ]
            )
        )
        result = compare_traces(plus, minus, kind=ComparisonKind.PHASE)
        # 179 - (-179) = 358 degrees == -2 degrees after wrapping.
        assert result.values()[0] == pytest.approx(-2.0, abs=1e-6)

    @staticmethod
    def test_shifted_resonance_over_a_shared_grid():
        """Same frequencies, different resonance: only differences remain."""
        collection = TraceCollection()
        freqs = list(range(1_000_000, 5_000_001, 100_000))

        def resonance(centre):
            return [
                Datapoint(freq, abs(freq - centre) / (10 * 1_000_000), 0.0)
                for freq in freqs
            ]

        low = collection.add(_snapshot(resonance(2_000_000)), name="low")
        high = collection.add(_snapshot(resonance(3_000_000)), name="high")
        result = compare_traces(low, high, kind=ComparisonKind.MAGNITUDE_DB)
        assert result.available
        assert not result.interpolated
        # The two models coincide exactly at the midpoint.
        midpoint = result.points[15]
        assert midpoint.freq == 2_500_000
        assert midpoint.scalar == pytest.approx(0.0, abs=1e-12)
        # Away from it the difference is real.
        assert result.points[0].scalar != pytest.approx(0.0)

    @staticmethod
    def test_disjoint_spans_are_unavailable_not_fabricated():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([1e6, 2e6])))
        second = collection.add(_snapshot(_flat([3e6, 4e6])))
        result = compare_traces(first, second)
        assert not result.available
        assert result.points == ()
        assert "overlapping" in result.unavailable_reason

    @staticmethod
    def test_missing_channel_is_unavailable():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100]), _flat([100])))
        second = collection.add(_snapshot(_flat([100])))
        result = compare_traces(first, second, parameter="S21")
        assert not result.available
        assert "S21" in result.unavailable_reason

    @staticmethod
    def test_unequal_grids_are_interpolated_not_extrapolated():
        collection = TraceCollection()
        # The comparison grid is the reference (fine) grid.  The source only
        # has every other frequency, so its values must be interpolated.
        coarse = collection.add(
            _snapshot(_circle(range(1_000_000, 4_000_001, 500_000))),
            name="coarse",
        )
        fine = collection.add(
            _snapshot(_circle(range(1_000_000, 4_000_001, 250_000))),
            name="fine",
        )
        result = compare_traces(coarse, fine, kind=ComparisonKind.COMPLEX)
        assert result.available
        assert result.interpolated
        # The compared grid is the reference grid inside the shared span.
        assert result.frequencies() == list(
            range(1_000_000, 4_000_001, 250_000)
        )

    @staticmethod
    def test_incompatible_reference_impedance_is_unavailable():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100]), impedance=50.0))
        second = collection.add(_snapshot(_flat([100]), impedance=75.0))
        result = compare_traces(first, second, kind=ComparisonKind.IMPEDANCE)
        assert not result.available
        assert "reference impedance" in result.unavailable_reason

    @staticmethod
    def test_impedance_difference_uses_each_reference_impedance():
        collection = TraceCollection()
        # gamma 0 -> 50 ohm, gamma 1/3 -> 100 ohm
        first = collection.add(_snapshot([Datapoint(100, 0.0, 0.0)]))
        second = collection.add(_snapshot([Datapoint(100, 1 / 3, 0.0)]))
        result = compare_traces(first, second, kind=ComparisonKind.IMPEDANCE)
        assert result.values()[0] == pytest.approx(-50.0, abs=1e-6)

    @staticmethod
    def test_zero_magnitude_is_handled_explicitly():
        collection = TraceCollection()
        silent = collection.add(_snapshot([Datapoint(100, 0.0, 0.0)]))
        loud = collection.add(_snapshot([Datapoint(100, 0.5, 0.0)]))
        result = compare_traces(silent, loud, kind=ComparisonKind.MAGNITUDE_DB)
        assert math.isinf(result.values()[0])
        equal = compare_traces(silent, silent, kind=ComparisonKind.MAGNITUDE_DB)
        assert equal.values()[0] == 0.0

    @staticmethod
    def test_duplicate_frequencies_use_last_occurrence():
        collection = TraceCollection()
        first = collection.add(
            _snapshot([Datapoint(100, 0.1, 0.0), Datapoint(100, 0.2, 0.0)])
        )
        second = collection.add(_snapshot([Datapoint(100, 0.1, 0.0)]))
        result = compare_traces(first, second, kind=ComparisonKind.COMPLEX)
        assert result.values()[0] == pytest.approx(0.1)

    @staticmethod
    def test_summary_labels_both_traces_units_and_interpolation():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])), name="A")
        second = collection.add(_snapshot(_flat([100], 0.2)), name="B")
        result = compare_traces(first, second, kind=ComparisonKind.MAGNITUDE_DB)
        summary = result.summary()
        assert "A" in summary
        assert "B" in summary
        assert "dB" in summary

    @staticmethod
    def test_marker_delta_picks_nearest_frequency():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100, 200, 300])))
        second = collection.add(_snapshot(_flat([100, 200, 300], 0.2)))
        result = compare_traces(first, second)
        assert marker_delta(result, 210).freq == 200
        assert marker_delta(result, 1000).freq == 300

    @staticmethod
    def test_marker_delta_on_unavailable_result_is_none():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        second = collection.add(_snapshot(_flat([500])))
        assert marker_delta(compare_traces(first, second), 100) is None


class TestComparisonCache:
    @staticmethod
    def test_cache_reuses_results_until_revision_changes():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        second = collection.add(_snapshot(_flat([100])))
        cache = ComparisonCache()
        before = cache.get(first, second, "S11", ComparisonKind.COMPLEX)
        again = cache.get(first, second, "S11", ComparisonKind.COMPLEX)
        assert before is again

        collection.replace(first.id, _snapshot(_flat([100], 0.3)))
        after = cache.get(first, second, "S11", ComparisonKind.COMPLEX)
        assert after is not before
        assert after.values()[0] == pytest.approx(0.2)

    @staticmethod
    def test_invalidate_by_trace_id():
        collection = TraceCollection()
        first = collection.add(_snapshot(_flat([100])))
        second = collection.add(_snapshot(_flat([100])))
        cache = ComparisonCache()
        cache.get(first, second, "S11", ComparisonKind.COMPLEX)
        cache.invalidate(first.id)
        assert cache._entries == {}


class TestComparisonAcrossDifferentGrids:
    """TRACE-04.1: overlap is a span, not a shared grid.

    Written after a real defect: comparison required at least one *identically
    sampled* frequency, so two traces on offset grids (a very common case when
    a sweep's start or point count changed) were reported as having no
    overlapping frequencies even though their spans overlapped completely.
    """

    @staticmethod
    def _pair(source_freqs, reference_freqs):
        collection = TraceCollection()
        reference = collection.add(
            _snapshot(_flat(reference_freqs, 0.1), name="ref"), name="ref"
        )
        source = collection.add(
            _snapshot(_flat(source_freqs, 0.1), name="src"), name="src"
        )
        return source, reference

    @staticmethod
    def test_offset_grids_are_comparable():
        source, reference = TestComparisonAcrossDifferentGrids._pair(
            [100, 200, 300], [150, 250, 350]
        )
        result = compare_traces(source, reference, kind=ComparisonKind.COMPLEX)
        assert result.available
        freqs = [point.freq for point in result.points]
        # The reference trace's own grid, clipped to the shared span
        # [max(min), min(max)] = [150, 300].
        assert freqs == [150, 250]
        assert all(abs(point.scalar) < 1e-9 for point in result.points)

    @staticmethod
    def test_no_shared_frequency_is_not_a_failure():
        source, reference = TestComparisonAcrossDifferentGrids._pair(
            [1_000_000, 1_100_000, 1_200_000],
            [1_050_000, 1_150_000, 1_250_000],
        )
        result = compare_traces(source, reference, kind=ComparisonKind.COMPLEX)
        assert result.available
        assert result.points

    @staticmethod
    def test_disjoint_spans_are_still_unavailable():
        source, reference = TestComparisonAcrossDifferentGrids._pair(
            [900, 950], [100, 200, 300]
        )
        result = compare_traces(source, reference, kind=ComparisonKind.COMPLEX)
        assert not result.available
        assert "overlapping" in result.unavailable_reason

    @staticmethod
    def test_only_the_shared_span_is_compared():
        source, reference = TestComparisonAcrossDifferentGrids._pair(
            [100, 200, 300, 400], [250, 350, 450]
        )
        result = compare_traces(source, reference, kind=ComparisonKind.COMPLEX)
        assert result.available
        # Reference grid [250, 350, 450] clipped to the shared span
        # [250, 400]: 450 is outside the source's data and is not used.
        assert [point.freq for point in result.points] == [250, 350]
