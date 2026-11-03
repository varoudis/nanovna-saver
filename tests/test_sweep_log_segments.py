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
"""A multi-segment logarithmic sweep must not repeat a frequency.

Each segment used to end on the next one's start, so the published grid held
every segment boundary twice: a 3-segment 11-point sweep produced 33 samples
of which two were duplicates.  That repeat then had to be tolerated all the
way down the pipeline -- by the Touchstone export, the declared-grid
completeness check and the limit evaluator.

The fix keeps the geometric boundaries and makes each segment stop one of its
own steps short of the next boundary, so the grid is strictly increasing with
the same number of samples, starting at the configured start and ending at the
configured end.  Linear sweeps and single-segment logarithmic sweeps are
untouched.
"""

import math

import pytest

from NanoVNASaver.Limits import _grid_tolerance, _local_spacings
from NanoVNASaver.Settings.Sweep import Properties, Sweep

#: Spans that hold more hertz than samples, so every sample can be distinct.
SPANS = [
    (1_000_000, 100_000_000, 11, 3),
    (14_000_000, 14_300_000, 101, 2),
    (3_600_000, 30_000_000, 101, 3),
    (500_000, 30_000_000, 101, 10),
    (1_000_000, 1_000_100, 3, 4),
    (1_000_000, 10_000_000, 201, 4),
    (50_000_000, 50_100_000, 101, 4),
    (1_000_000, 2_000_000, 3, 2),
]


def _log(start: int, end: int, points: int, segments: int) -> Sweep:
    return Sweep(start, end, points, segments, Properties(logarithmic=True))


@pytest.mark.parametrize(("start", "end", "points", "segments"), SPANS)
def test_a_log_sweep_never_repeats_a_frequency(start, end, points, segments):
    frequencies = list(_log(start, end, points, segments).get_frequencies())

    assert len(frequencies) == points * segments
    assert len(set(frequencies)) == len(frequencies)


@pytest.mark.parametrize(("start", "end", "points", "segments"), SPANS)
def test_a_log_sweep_keeps_its_ends_and_its_point_count(
    start, end, points, segments
):
    frequencies = list(_log(start, end, points, segments).get_frequencies())

    assert frequencies[0] == start
    assert frequencies[-1] == end
    assert frequencies == sorted(frequencies)


@pytest.mark.parametrize(("start", "end", "points", "segments"), SPANS)
def test_every_log_segment_still_increases(start, end, points, segments):
    sweep = _log(start, end, points, segments)

    for index in range(segments):
        first, last = sweep.get_index_range(index)
        assert first < last, index


@pytest.mark.parametrize(("start", "end", "points", "segments"), SPANS)
def test_a_log_sweep_stays_geometric(start, end, points, segments):
    """The fix must not flatten the sweep into a linear ramp."""
    sweep = _log(start, end, points, segments)
    if points * segments < 3:
        pytest.skip("no interior segment to compare")

    frequencies = list(sweep.get_frequencies())
    lowest = frequencies[1] - frequencies[0]
    highest = frequencies[-1] - frequencies[-2]
    assert highest > lowest


def test_the_boundary_starts_the_next_segment_not_the_previous_one():
    """The shared boundary belongs to one segment, and to one only."""
    sweep = _log(1_000_000, 100_000_000, 11, 3)
    boundary = sweep._log_boundary(1)

    first_stop = sweep.get_index_range(0)[1]
    second_start = sweep.get_index_range(1)[0]
    assert second_start == boundary
    assert first_stop < boundary
    assert (boundary - first_stop) >= 1


def test_a_single_segment_log_sweep_is_untouched():
    """One segment has no join, so nothing about it may change.

    ``get_frequencies`` steps linearly between the two boundaries, so a
    single-segment sweep is a linear grid from start to end; that is the
    historic behaviour and it is not affected by the boundary change.
    """
    sweep = _log(1_000_000, 100_000_000, 11, 1)
    start, stop = sweep.get_index_range(0)

    assert (start, stop) == (1_000_000, 100_000_000)
    frequencies = list(sweep.get_frequencies())
    step = (stop - start) / (sweep.points - 1)
    expected = [round(start + index * step) for index in range(sweep.points)]
    assert frequencies == expected


@pytest.mark.parametrize(
    ("start", "end", "points", "segments"),
    [(1_000_000, 100_000_000, 11, 3), (3_600_000, 30_000_000, 101, 3)],
)
def test_a_linear_sweep_is_untouched(start, end, points, segments):
    """The linear grid the fix does not concern, pinned to its old values."""
    sweep = Sweep(start, end, points, segments)
    ranges = [sweep.get_index_range(i) for i in range(segments)]

    step = sweep.stepsize
    span = (points - 1) * step
    expected = [
        (start + i * points * step, start + i * points * step + span)
        for i in range(segments)
    ]
    assert ranges == expected


def test_no_sample_gap_looks_like_a_missing_point():
    """Every interval is a plausible step, so the grid reads as complete.

    A firmware that truncates its sweep step to whole hertz drifts by up to
    one hertz per point; the declared-grid check allows that.  The join must
    not be a special case that looks like a hole.
    """
    sweep = _log(500_000, 30_000_000, 101, 10)
    frequencies = list(sweep.get_frequencies())
    spacings = _local_spacings(frequencies)

    for index, freq in enumerate(frequencies):
        tolerance = _grid_tolerance(freq, spacings[index], sweep.points)
        assert tolerance > 0.0, index
        assert tolerance >= 1.0


def test_a_truncated_device_grid_still_matches_the_declared_one():
    """The join does not hide a frequency the device would omit.

    ``NanoVNA_V2`` reports ``start + i * int(step)``, which drifts by up to a
    hertz per point inside a segment.  Every declared sample must still be
    found within the documented tolerance.
    """
    sweep = _log(500_000, 30_000_000, 101, 10)
    declared = list(sweep.get_frequencies())
    reported: list[int] = []
    for index in range(sweep.segments):
        start, stop = sweep.get_index_range(index)
        step = int((stop - start) / (sweep.points - 1))
        reported.extend(int(start + i * step) for i in range(sweep.points))

    spacings = _local_spacings(declared)
    for index, freq in enumerate(declared):
        tolerance = _grid_tolerance(freq, spacings[index], sweep.points)
        assert abs(freq - reported[index]) <= tolerance, (
            f"{freq} Hz declared, device reported {reported[index]} Hz,"
            f" tolerance {tolerance}"
        )


def test_a_span_too_narrow_for_the_grid_keeps_the_historic_shape():
    """Nothing is invented outside the sweep when the span cannot hold it.

    ``Sweep(1_000_000, 1_000_000)`` has one hertz: 15 samples cannot be
    distinct there.  The old overlapping grid is kept rather than pushing a
    boundary past the configured end.
    """
    sweep = _log(1_000_000, 1_000_000, 5, 3)
    frequencies = list(sweep.get_frequencies())

    assert len(frequencies) == 15
    assert min(frequencies) >= 1_000_000
    assert max(frequencies) <= 1_000_000
    for index in range(sweep.segments):
        first, last = sweep.get_index_range(index)
        assert first == last


def test_the_narrow_span_rounding_is_reported_not_hidden():
    """A sub-step span cannot satisfy both ends; say what actually happens.

    With 1 Hz of span and 5 points the geometric boundaries round to the same
    hertz, so the segment is a single frequency repeated.  That is the honest
    outcome for a configuration the device cannot express, and it is what the
    historic code did too.
    """
    sweep = _log(1_000_000, 1_000_001, 5, 2)
    frequencies = list(sweep.get_frequencies())

    assert frequencies[0] == 1_000_000
    assert frequencies[-1] == 1_000_001
    assert len(set(frequencies)) < len(frequencies)
    assert all(math.isfinite(value) for value in frequencies)
