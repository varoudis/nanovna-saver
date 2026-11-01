#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
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
"""LIMIT-01/02: limit set format, numeric boundaries and coverage policy."""

import json
import math

import pytest

from NanoVNASaver.Limits import (
    GAP_FACTOR,
    LIMIT_SET_FORMAT,
    LimitCatalogue,
    LimitError,
    LimitFormatError,
    LimitRule,
    LimitSegment,
    LimitSet,
    LimitValidationError,
    Metric,
    Verdict,
    evaluate_rule,
    load,
    loads,
)
from NanoVNASaver.Measurement import CompletionState, snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Settings.Sweep import Properties, Sweep

STEP = 1_000_000
BAND_START = 10_000_000
BAND_STOP = 20_000_000


def _magnitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _reflection_for_vswr(vswr: float) -> float:
    """Reflection magnitude that yields ``vswr``."""
    return (vswr - 1.0) / (vswr + 1.0)


def _db_for_vswr(vswr: float) -> float:
    return 20.0 * math.log10(_reflection_for_vswr(vswr))


def _channel(freqs, values):
    """Datapoints whose magnitude in dB is given by ``values``."""
    return [
        Datapoint(int(freq), _magnitude(value), 0.0)
        for freq, value in zip(freqs, values, strict=True)
    ]


def _grid(start=BAND_START, stop=BAND_STOP, step=STEP):
    return list(range(start, stop + step, step))


def _snapshot(s11=None, s21=None, *, completion=CompletionState.COMPLETE):
    common = {
        "name": "test",
        "source": "test",
        "completion": completion,
    }
    if s11 is None and s21 is not None:
        # A one-port acquisition still needs an S11 channel for the adapter.
        s11 = _channel(_grid(), [-20.0] * len(_grid()))
    return snapshot_from_datapoints(s11, s21, **common)


def _rule(metric, segments, rule_id="r1"):
    return LimitRule(metric=metric, segments=tuple(segments), id=rule_id)


def _evaluate(snapshot, rule):
    return evaluate_rule(rule, snapshot, trace_id="t", trace_name="T")


def _vswr_rule(maximum=2.0):
    return _rule(
        Metric.S11_VSWR,
        [LimitSegment(BAND_START, BAND_STOP, maximum=maximum)],
    )


class TestFormatRoundTrip:
    @staticmethod
    def test_document_round_trips_byte_stable():
        text = open("tests/data/example.nvlimits", encoding="utf-8").read()
        limit_set = loads(text)
        again = loads(json.dumps(limit_set.to_dict()))
        assert again.to_dict() == limit_set.to_dict()

    @staticmethod
    def test_example_documents_every_metric():
        limit_set = load("tests/data/example.nvlimits")
        assert limit_set.version == 1
        assert {rule.metric for rule in limit_set.rules} == {
            Metric.S11_VSWR,
            Metric.S11_RETURN_LOSS_DB,
            Metric.S21_INSERTION_LOSS_DB,
        }
        assert limit_set.rules[0].segments[1].maximum == 1.5

    @staticmethod
    def test_segments_are_sorted_by_start():
        rule = _rule(
            Metric.S11_VSWR,
            [
                LimitSegment(20_000_000, 30_000_000, maximum=2.0),
                LimitSegment(10_000_000, 20_000_000, maximum=3.0),
            ],
        )
        assert [segment.start for segment in rule.segments] == [
            10_000_000,
            20_000_000,
        ]


class TestMalformedDocuments:
    @staticmethod
    def test_invalid_json_reports_position():
        with pytest.raises(LimitFormatError) as info:
            loads('{"format": "nanovna-saver-limits", ')
        assert "line" in str(info.value) and "column" in str(info.value)

    @staticmethod
    def test_wrong_format_is_rejected():
        with pytest.raises(LimitFormatError) as info:
            loads(json.dumps({"format": "something-else", "version": 1}))
        assert "unexpected limit set format" in str(info.value)

    @staticmethod
    def test_newer_version_is_rejected():
        with pytest.raises(LimitFormatError) as info:
            loads(json.dumps({"format": LIMIT_SET_FORMAT, "version": 2}))
        assert "newer than supported" in str(info.value)

    @staticmethod
    def test_non_integer_version_is_rejected():
        with pytest.raises(LimitFormatError):
            loads(json.dumps({"format": LIMIT_SET_FORMAT, "version": "1"}))

    @staticmethod
    def test_unknown_metric_lists_known_metrics():
        document = {
            "format": LIMIT_SET_FORMAT,
            "version": 1,
            "name": "x",
            "rules": [
                {
                    "id": "r",
                    "metric": "s33_magic",
                    "segments": [{"start_hz": 1, "stop_hz": 2, "maximum": 1.0}],
                }
            ],
        }
        with pytest.raises(LimitValidationError) as info:
            LimitSet.from_dict(document)
        message = str(info.value)
        assert "unknown metric" in message
        assert "s11_vswr" in message

    @staticmethod
    def test_unknown_key_is_rejected_actionably():
        document = {
            "format": LIMIT_SET_FORMAT,
            "version": 1,
            "name": "x",
            "rules": [
                {
                    "id": "r",
                    "metric": "s11_vswr",
                    "segments": [
                        {
                            "start_hz": 1,
                            "stop_hz": 2,
                            "maximum": 1.0,
                            "unit": "dB",
                        }
                    ],
                }
            ],
        }
        with pytest.raises(LimitValidationError) as info:
            LimitSet.from_dict(document)
        assert "unknown key 'unit'" in str(info.value)

    @staticmethod
    def test_reversed_range_is_rejected():
        with pytest.raises(LimitValidationError) as info:
            LimitSegment(2_000_000, 1_000_000, maximum=2.0)
        assert "must be greater than" in str(info.value)

    @staticmethod
    def test_zero_width_segment_is_rejected():
        with pytest.raises(LimitValidationError):
            LimitSegment(1_000_000, 1_000_000, maximum=2.0)

    @staticmethod
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_nonfinite_bound_is_rejected(value):
        document = {
            "format": LIMIT_SET_FORMAT,
            "version": 1,
            "name": "x",
            "rules": [
                {
                    "id": "r",
                    "metric": "s11_vswr",
                    "segments": [
                        {"start_hz": 1, "stop_hz": 2, "maximum": value}
                    ],
                }
            ],
        }
        with pytest.raises(LimitValidationError) as info:
            LimitSet.from_dict(json.loads(json.dumps(document)))
        assert "finite" in str(info.value)

    @staticmethod
    def test_string_bound_is_rejected():
        with pytest.raises(LimitValidationError) as info:
            LimitSegment(1, 2, maximum="2.0")
        assert "must be a number" in str(info.value)

    @staticmethod
    def test_conflicting_bounds_are_rejected():
        with pytest.raises(LimitValidationError) as info:
            LimitSegment(1, 2, minimum=5.0, maximum=2.0)
        assert "conflicting bounds" in str(info.value)

    @staticmethod
    def test_empty_segment_bounds_are_rejected():
        with pytest.raises(LimitValidationError) as info:
            LimitSegment(1, 2)
        assert "minimum, a maximum, or both" in str(info.value)

    @staticmethod
    def test_segments_must_tile_without_gaps():
        with pytest.raises(LimitValidationError) as info:
            _rule(
                Metric.S11_VSWR,
                [
                    LimitSegment(1, 2, maximum=2.0),
                    LimitSegment(3, 4, maximum=2.0),
                ],
            )
        assert "gap" in str(info.value)

    @staticmethod
    def test_segments_must_tile_without_overlap():
        with pytest.raises(LimitValidationError) as info:
            _rule(
                Metric.S11_VSWR,
                [
                    LimitSegment(1, 5, maximum=2.0),
                    LimitSegment(4, 8, maximum=2.0),
                ],
            )
        assert "overlap" in str(info.value)

    @staticmethod
    def test_overlapping_rules_for_one_metric_are_rejected():
        with pytest.raises(LimitValidationError) as info:
            LimitSet(
                "x",
                (
                    _rule(
                        Metric.S11_VSWR,
                        [LimitSegment(1, 5, maximum=2.0)],
                        rule_id="a",
                    ),
                    _rule(
                        Metric.S11_VSWR,
                        [LimitSegment(3, 8, maximum=3.0)],
                        rule_id="b",
                    ),
                ),
            )
        assert "overlapping bands" in str(info.value)

    @staticmethod
    def test_rules_for_different_metrics_may_share_a_band():
        limit_set = LimitSet(
            "x",
            (
                _rule(
                    Metric.S11_VSWR,
                    [LimitSegment(1, 5, maximum=2.0)],
                    rule_id="a",
                ),
                _rule(
                    Metric.S11_RETURN_LOSS_DB,
                    [LimitSegment(1, 5, minimum=10.0)],
                    rule_id="b",
                ),
            ),
        )
        assert len(limit_set.rules) == 2

    @staticmethod
    def test_duplicate_rule_ids_are_rejected():
        with pytest.raises(LimitValidationError) as info:
            LimitSet(
                "x",
                (
                    _rule(
                        Metric.S11_VSWR,
                        [LimitSegment(1, 5, maximum=2.0)],
                        rule_id="a",
                    ),
                    _rule(
                        Metric.S11_RETURN_LOSS_DB,
                        [LimitSegment(1, 5, minimum=10.0)],
                        rule_id="a",
                    ),
                ),
            )
        assert "duplicate rule id" in str(info.value)

    @staticmethod
    def test_empty_rules_are_rejected():
        with pytest.raises(LimitValidationError):
            LimitSet("x", ())

    @staticmethod
    def test_empty_name_is_rejected():
        with pytest.raises(LimitValidationError):
            LimitSet("   ", (_vswr_rule(),))

    @staticmethod
    def test_fractional_hertz_is_rejected():
        document = {
            "format": LIMIT_SET_FORMAT,
            "version": 1,
            "name": "x",
            "rules": [
                {
                    "id": "r",
                    "metric": "s11_vswr",
                    "segments": [
                        {"start_hz": 1.5, "stop_hz": 2, "maximum": 1.0}
                    ],
                }
            ],
        }
        with pytest.raises(LimitValidationError) as info:
            LimitSet.from_dict(document)
        assert "whole number of hertz" in str(info.value)

    @staticmethod
    def test_missing_metadata_file_reports_bad_path(tmp_path):
        with pytest.raises(LimitFormatError) as info:
            load(str(tmp_path / "absent.nvlimits"))
        assert "could not read" in str(info.value)

    @staticmethod
    def test_missing_rule_id_gets_a_stable_name():
        document = {
            "format": LIMIT_SET_FORMAT,
            "version": 1,
            "name": "x",
            "rules": [
                {
                    "metric": "s11_vswr",
                    "segments": [{"start_hz": 1, "stop_hz": 2, "maximum": 1.0}],
                }
            ],
        }
        assert LimitSet.from_dict(document).rules[0].id == "rule-1"


class TestCatalogueKeepsValidSettings:
    @staticmethod
    def test_rejected_load_leaves_the_catalogue_unchanged(tmp_path):
        good = tmp_path / "good.nvlimits"
        good.write_text(
            json.dumps(
                {
                    "format": LIMIT_SET_FORMAT,
                    "version": 1,
                    "name": "good",
                    "rules": [
                        {
                            "id": "r",
                            "metric": "s11_vswr",
                            "segments": [
                                {
                                    "start_hz": 1,
                                    "stop_hz": 2,
                                    "maximum": 2.0,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        catalogue = LimitCatalogue()
        catalogue.load(str(good))
        assert catalogue.names() == ["good"]
        assert catalogue.active_name == "good"

        broken = tmp_path / "broken.nvlimits"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(LimitFormatError):
            catalogue.load(str(broken))
        assert catalogue.names() == ["good"]
        assert catalogue.active().name == "good"

    @staticmethod
    def test_activating_a_missing_set_raises():
        with pytest.raises(LimitError):
            LimitCatalogue().activate("absent")

    @staticmethod
    def test_removing_the_active_set_elects_another():
        catalogue = LimitCatalogue()
        catalogue.add(_set("a"))
        catalogue.add(_set("b"))
        catalogue.activate("b")
        catalogue.remove("b")
        assert catalogue.active_name == "a"


def _set(name, maximum=2.0):
    return LimitSet(
        name,
        (_rule(Metric.S11_VSWR, [LimitSegment(1, 2, maximum=maximum)]),),
    )


class TestInclusiveBoundaries:
    @staticmethod
    def test_value_equal_to_maximum_passes():
        freqs = [10_000_000, 11_000_000]
        snapshot = _snapshot(_channel(freqs, [-20.0, _db_for_vswr(2.0)]))
        rule = _rule(
            Metric.S11_VSWR,
            [LimitSegment(10_000_000, 11_000_000, maximum=2.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.PASS
        assert result.worst_value == pytest.approx(2.0, rel=1e-6)

    @staticmethod
    def test_value_just_beyond_maximum_fails():
        freqs = [10_000_000, 11_000_000]
        snapshot = _snapshot(_channel(freqs, [-20.0, _db_for_vswr(2.1)]))
        rule = _rule(
            Metric.S11_VSWR,
            [LimitSegment(10_000_000, 11_000_000, maximum=2.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.FAIL
        assert result.worst_frequency == 11_000_000
        assert result.worst_value > 2.0
        assert result.worst_limit == 2.0
        assert result.worst_kind == "maximum"
        assert result.worst_violation == pytest.approx(result.worst_value - 2.0)

    @staticmethod
    def test_value_equal_to_minimum_passes():
        freqs = [10_000_000, 11_000_000]
        snapshot = _snapshot(_channel(freqs, [-10.0, -12.0]))
        rule = _rule(
            Metric.S11_RETURN_LOSS_DB,
            [LimitSegment(10_000_000, 11_000_000, minimum=10.0)],
        )
        assert _evaluate(snapshot, rule).verdict is Verdict.PASS

    @staticmethod
    def test_minimum_violation_answers_with_the_lowest_value():
        freqs = [10_000_000, 11_000_000]
        snapshot = _snapshot(_channel(freqs, [-5.0, -12.0]))
        rule = _rule(
            Metric.S11_RETURN_LOSS_DB,
            [LimitSegment(10_000_000, 11_000_000, minimum=10.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.FAIL
        assert result.worst_value == pytest.approx(5.0)
        assert result.worst_frequency == 10_000_000
        assert result.worst_kind == "minimum"


class TestMetricSigns:
    @staticmethod
    def test_return_loss_is_positive_for_a_good_match():
        point = Datapoint(1_000_000, 0.1, 0.0)
        assert Metric.S11_RETURN_LOSS_DB.measure(point) == pytest.approx(20.0)

    @staticmethod
    def test_insertion_loss_is_positive_for_loss():
        point = Datapoint(1_000_000, 0.5, 0.0)
        assert Metric.S21_INSERTION_LOSS_DB.measure(point) == pytest.approx(
            6.0206, rel=1e-4
        )

    @staticmethod
    def test_gain_has_negative_insertion_loss():
        point = Datapoint(1_000_000, 2.0, 0.0)
        assert Metric.S21_INSERTION_LOSS_DB.measure(point) == pytest.approx(
            -6.0206, rel=1e-4
        )

    @staticmethod
    def test_vswr_is_infinite_at_total_reflection():
        point = Datapoint(1_000_000, 1.0, 0.0)
        assert math.isinf(Metric.S11_VSWR.measure(point))

    @staticmethod
    def test_units_are_a_property_of_the_metric():
        assert Metric.S11_VSWR.unit == ""
        assert Metric.S11_RETURN_LOSS_DB.unit == "dB"
        assert Metric.S21_INSERTION_LOSS_DB.parameter == "S21"
        assert Metric.S11_VSWR.parameter == "S11"


class TestTotalReflection:
    @staticmethod
    def test_infinite_vswr_fails_a_finite_maximum():
        snapshot = _snapshot(_channel(_grid(), [-30.0] * len(_grid())))
        points = list(snapshot.channel("S11"))
        points[1] = Datapoint(points[1].freq, 1.0, 0.0)
        snapshot = _snapshot(points)
        result = _evaluate(snapshot, _vswr_rule(2.0))
        assert result.verdict is Verdict.FAIL
        assert math.isinf(result.worst_value)
        assert result.worst_frequency == points[1].freq

    @staticmethod
    def test_total_reflection_is_not_confused_with_missing_data():
        points = [Datapoint(10_000_000, 1.0, 0.0), Datapoint(11_000_000, 1, 0)]
        result = _evaluate(
            _snapshot(points),
            _rule(
                Metric.S11_VSWR,
                [LimitSegment(10_000_000, 11_000_000, maximum=1e9)],
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert result.reason == ""


class TestUnavailableCases:
    @staticmethod
    def test_missing_channel_is_unavailable_not_pass():
        snapshot = _snapshot(_channel(_grid(), [-30.0] * len(_grid())))
        rule = _rule(
            Metric.S21_INSERTION_LOSS_DB,
            [LimitSegment(BAND_START, BAND_STOP, maximum=3.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.UNAVAILABLE
        assert "no S21 channel" in result.reason

    @staticmethod
    def test_partial_acquisition_is_unavailable():
        snapshot = _snapshot(
            _channel(_grid(), [-30.0] * len(_grid())),
            completion=CompletionState.PARTIAL,
        )
        result = _evaluate(snapshot, _vswr_rule(3.0))
        assert result.verdict is Verdict.UNAVAILABLE
        assert "partial" in result.reason

    @staticmethod
    def test_interrupted_acquisition_is_unavailable():
        snapshot = _snapshot(
            _channel(_grid(), [-30.0] * len(_grid())),
            completion=CompletionState.INTERRUPTED,
        )
        assert (
            _evaluate(snapshot, _vswr_rule(3.0)).verdict is Verdict.UNAVAILABLE
        )

    @staticmethod
    def test_band_wider_than_the_sweep_is_unavailable():
        snapshot = _snapshot(_channel(_grid(), [-30.0] * len(_grid())))
        rule = _rule(
            Metric.S11_VSWR,
            [LimitSegment(BAND_START, BAND_STOP + 5_000_000, maximum=3.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.UNAVAILABLE
        assert "not covered" in result.reason

    @staticmethod
    def test_band_starting_inside_a_missing_chunk_is_unavailable():
        freqs = [10_000_000, 11_000_000, 19_000_000, 20_000_000]
        snapshot = _snapshot(_channel(freqs, [-30.0] * len(freqs)))
        # The sweep nominally spans the band but has a large hole in it.
        rule = _rule(
            Metric.S11_VSWR,
            [LimitSegment(12_000_000, 20_000_000, maximum=3.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.UNAVAILABLE
        assert "no samples between" in result.reason

    @staticmethod
    def test_a_hole_of_one_missing_sample_is_unavailable():
        freqs = _grid()
        del freqs[3]
        snapshot = _snapshot(_channel(freqs, [-30.0] * len(freqs)))
        result = _evaluate(snapshot, _vswr_rule(3.0))
        assert result.verdict is Verdict.UNAVAILABLE
        assert GAP_FACTOR < 2

    @staticmethod
    def test_absent_segment_coverage_is_unavailable():
        freqs = [10_000_000, 11_000_000]
        snapshot = _snapshot(_channel(freqs, [-30.0] * len(freqs)))
        # Both segment bounds are inside the data span, but the second
        # segment has no samples at all.
        rule = _rule(
            Metric.S11_VSWR,
            [
                LimitSegment(10_000_000, 11_000_000, maximum=3.0),
                LimitSegment(11_000_000, 12_000_000, maximum=3.0),
            ],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.UNAVAILABLE

    @staticmethod
    def test_nonfinite_sample_is_unavailable():
        points = [
            Datapoint(10_000_000, 0.1, 0.0),
            Datapoint(11_000_000, math.nan, 0.0),
            Datapoint(12_000_000, 0.1, 0.0),
        ]
        snapshot = _snapshot(points)
        result = _evaluate(
            snapshot,
            _rule(
                Metric.S11_VSWR,
                [LimitSegment(10_000_000, 12_000_000, maximum=3.0)],
            ),
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "non-finite" in result.reason

    @staticmethod
    def test_empty_channel_is_unavailable():
        snapshot = snapshot_from_datapoints(
            _channel(_grid(), [-20.0] * len(_grid())),
            None,
            name="t",
        )
        rule = _rule(
            Metric.S21_INSERTION_LOSS_DB,
            [LimitSegment(BAND_START, BAND_STOP, maximum=3.0)],
        )
        assert _evaluate(snapshot, rule).verdict is Verdict.UNAVAILABLE


class TestCapturedGridCoverage:
    """Coverage is judged against the grid that was actually captured.

    A logarithmic sweep, and a multi-segment sweep whose step changes at a
    segment join, are legitimate non-uniform grids: a rule inside such a
    capture must be decided.  A genuinely missing sample or chunk in the
    same grids must still be reported as a hole.
    """

    @staticmethod
    def _log_snapshot(segments=10):
        sweep = Sweep(
            500_000,
            30_000_000,
            101,
            segments,
            Properties(logarithmic=True),
        )
        frequencies = list(sweep.get_frequencies())
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in frequencies],
            None,
            sweep=sweep,
        )
        return frequencies, snapshot

    @staticmethod
    def _band_rule(start, stop, maximum=2.0):
        return _rule(
            Metric.S11_VSWR, [LimitSegment(start, stop, maximum=maximum)]
        )

    @staticmethod
    def _inside(frequencies, start, stop):
        return [
            index
            for index, freq in enumerate(frequencies)
            if start <= freq <= stop
        ]

    @pytest.mark.parametrize("segments", [1, 2, 4, 10, 20])
    def test_a_complete_logarithmic_sweep_is_decided(self, segments):
        frequencies, snapshot = self._log_snapshot(segments)
        band = [f for f in frequencies if 14_000_000 <= f <= 15_000_000]
        assert band, "the log grid must sample the rule band"
        result = _evaluate(snapshot, self._band_rule(14_000_000, 15_000_000))
        assert result.verdict is Verdict.PASS, result.reason
        assert result.samples >= len(band)

    @staticmethod
    def test_a_band_across_a_log_segment_join_is_decided():
        _freqs, snapshot = TestCapturedGridCoverage._log_snapshot(2)
        # 3-5 MHz straddles the join between the two log segments, where the
        # captured step size changes by roughly a factor of eight.
        result = _evaluate(
            snapshot,
            TestCapturedGridCoverage._band_rule(3_000_000, 5_000_000),
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_a_missing_log_sample_inside_the_band_is_a_hole():
        frequencies, _snapshot = TestCapturedGridCoverage._log_snapshot(10)
        inside = TestCapturedGridCoverage._inside(
            frequencies, 14_000_000, 15_000_000
        )
        holed = frequencies[: inside[5]] + frequencies[inside[5] + 1 :]
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in holed], None
        )
        result = _evaluate(
            snapshot,
            TestCapturedGridCoverage._band_rule(14_000_000, 15_000_000),
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert result.reason.startswith("no samples between")

    @staticmethod
    def test_a_missing_log_chunk_is_a_hole():
        frequencies, _snapshot = TestCapturedGridCoverage._log_snapshot(10)
        inside = TestCapturedGridCoverage._inside(
            frequencies, 14_000_000, 15_000_000
        )
        holed = frequencies[: inside[5]] + frequencies[inside[5] + 20 :]
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in holed], None
        )
        result = _evaluate(
            snapshot,
            TestCapturedGridCoverage._band_rule(14_000_000, 15_000_000),
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert result.reason.startswith("no samples between")

    @staticmethod
    def test_an_emptied_band_middle_in_a_log_capture_is_a_hole():
        frequencies, _snapshot = TestCapturedGridCoverage._log_snapshot(10)
        inside = TestCapturedGridCoverage._inside(
            frequencies, 14_000_000, 15_000_000
        )
        # The band edges are still sampled; everything between them is gone.
        holed = frequencies[: inside[1]] + frequencies[inside[-1] :]
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in holed], None
        )
        result = _evaluate(
            snapshot,
            TestCapturedGridCoverage._band_rule(14_000_000, 15_000_000),
        )
        assert result.verdict is Verdict.UNAVAILABLE

    @staticmethod
    def test_a_multi_segment_linear_sweep_is_decided():
        sweep = Sweep(1_000_000, 10_000_000, 101, 5)
        frequencies = list(sweep.get_frequencies())
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in frequencies],
            None,
            sweep=sweep,
        )
        result = _evaluate(
            snapshot, TestCapturedGridCoverage._band_rule(2_000_000, 8_000_000)
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_a_missing_sample_at_a_segment_join_is_a_hole():
        sweep = Sweep(1_000_000, 10_000_000, 101, 5)
        frequencies = list(sweep.get_frequencies())
        holed = frequencies[:101] + frequencies[102:]
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in holed], None
        )
        result = _evaluate(
            snapshot, TestCapturedGridCoverage._band_rule(2_000_000, 8_000_000)
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert result.reason.startswith("no samples between")

    @staticmethod
    def test_a_non_uniform_grid_reports_a_missing_channel_as_unavailable():
        _freqs, snapshot = TestCapturedGridCoverage._log_snapshot(10)
        rule = _rule(
            Metric.S21_INSERTION_LOSS_DB,
            [LimitSegment(14_000_000, 15_000_000, maximum=3.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.UNAVAILABLE
        assert "no S21 channel" in result.reason

    @staticmethod
    def test_a_hole_never_turns_into_a_pass():
        frequencies, _snapshot = TestCapturedGridCoverage._log_snapshot(10)
        inside = TestCapturedGridCoverage._inside(
            frequencies, 14_000_000, 15_000_000
        )
        holed = frequencies[: inside[5]] + frequencies[inside[5] + 1 :]
        # Every retained sample would pass; the hole still blocks the check.
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in holed], None
        )
        result = _evaluate(
            snapshot,
            TestCapturedGridCoverage._band_rule(14_000_000, 15_000_000),
        )
        assert result.verdict is Verdict.UNAVAILABLE


class TestDeclaredGridCompleteness:
    """A capture is checked against the grid its own settings declare.

    The snapshot provenance records the acquisition ``SweepInfo``.  That is
    authoritative: the completed capture must hold exactly one sample per
    declared frequency, so a cluster of missing samples cannot validate
    itself and a duplicate cannot stand in for a different frequency.
    """

    @staticmethod
    def _snapshot(frequencies, sweep=None):
        return snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in frequencies],
            None,
            sweep=sweep,
            simulated=sweep is not None,
            source="simulated" if sweep is not None else "",
        )

    @staticmethod
    def _rule(start, stop, maximum=2.0):
        return _rule(
            Metric.S11_VSWR, [LimitSegment(start, stop, maximum=maximum)]
        )

    @staticmethod
    def _device_grid(sweep):
        """Frequencies a real device reports for ``sweep``'s segments."""
        from NanoVNASaver.Hardware.Simulated import make_simulated_vna

        vna = make_simulated_vna("antenna")
        vna.datapoints = sweep.points
        frequencies: list[int] = []
        for index in range(sweep.segments):
            start, stop = sweep.get_index_range(index)
            vna.setSweep(start, stop)
            frequencies.extend(vna.read_frequencies())
        return frequencies

    @staticmethod
    def test_the_reported_clustered_hole_is_unavailable():
        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        full = list(sweep.get_frequencies())
        assert full == list(range(1_000_000, 1_000_011))
        retained = [
            freq
            for freq in full
            if freq not in {1_000_002, 1_000_003, 1_000_005, 1_000_006}
        ]
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in retained],
            None,
            sweep=sweep,
            simulated=True,
            source="simulated",
        )
        result = evaluate_rule(
            LimitRule(
                Metric.S11_VSWR,
                (LimitSegment(1_000_000, 1_000_010, maximum=2),),
                id="known-grid",
            ),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "missing 4 of 11 samples" in result.reason
        assert "1000002 Hz" in result.reason
        assert "11 point(s) x 1 segment(s)" in result.reason

    @staticmethod
    def test_the_complete_declared_grid_passes():
        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        snapshot = TestDeclaredGridCompleteness._snapshot(
            sweep.get_frequencies(), sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 1_000_010),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason
        assert result.reason == ""

    @staticmethod
    def test_a_clustered_hole_on_a_known_uniform_grid_is_unavailable():
        sweep = Sweep(13_000_000, 15_000_000, 101, 1)
        frequencies = list(sweep.get_frequencies())
        removed = set(frequencies[20:28])  # eight adjacent samples
        snapshot = TestDeclaredGridCompleteness._snapshot(
            [freq for freq in frequencies if freq not in removed], sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(13_000_000, 15_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "missing 8 of 101 samples" in result.reason

    @staticmethod
    def test_a_clustered_hole_on_a_known_log_grid_is_unavailable():
        sweep = Sweep(
            500_000, 30_000_000, 101, 10, Properties(logarithmic=True)
        )
        frequencies = list(sweep.get_frequencies())
        inside = [
            index
            for index, freq in enumerate(frequencies)
            if 14_000_000 <= freq <= 15_000_000
        ]
        removed = set(frequencies[inside[2] : inside[9]])
        snapshot = TestDeclaredGridCompleteness._snapshot(
            [freq for freq in frequencies if freq not in removed], sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(14_000_000, 15_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "declared by the recorded sweep" in result.reason
        assert "logarithmic" in result.reason

    @staticmethod
    def test_a_hole_at_a_known_segment_join_is_unavailable():
        sweep = Sweep(1_000_000, 10_000_000, 101, 5)
        frequencies = list(sweep.get_frequencies())
        snapshot = TestDeclaredGridCompleteness._snapshot(
            frequencies[:101] + frequencies[102:], sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(2_000_000, 8_000_000), snapshot
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "missing 1 of" in result.reason

    @pytest.mark.parametrize("segments", [1, 2, 5])
    def test_a_complete_multi_segment_grid_passes(self, segments):
        sweep = Sweep(1_000_000, 10_000_000, 51, segments)
        frequencies = list(sweep.get_frequencies())
        snapshot = TestDeclaredGridCompleteness._snapshot(frequencies, sweep)
        # The settings arithmetic ends a rounded grid slightly short of the
        # requested stop, so the band is taken from the declared samples.
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(frequencies[0], frequencies[-1]),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_legitimate_join_duplicates_are_accepted():
        # A logarithmic multi-segment grid repeats the joint frequency: the
        # declared grid itself contains it twice, so it is not an extra.
        sweep = Sweep(500_000, 30_000_000, 101, 2, Properties(logarithmic=True))
        frequencies = list(sweep.get_frequencies())
        assert len(frequencies) != len(set(frequencies))
        snapshot = TestDeclaredGridCompleteness._snapshot(frequencies, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 10_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_a_duplicate_cannot_fill_a_missing_frequency():
        sweep = Sweep(10_000_000, 20_000_000, 11, 1)
        frequencies = list(sweep.get_frequencies())
        # Duplicate 13 MHz where 14 MHz should be.
        holed = frequencies[:3] + [frequencies[3]] + frequencies[5:]
        snapshot = TestDeclaredGridCompleteness._snapshot(holed, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(10_000_000, 20_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "first missing sample at 14000000 Hz" in result.reason

    @staticmethod
    def test_an_off_grid_sample_is_not_accepted():
        sweep = Sweep(10_000_000, 20_000_000, 11, 1)
        frequencies = list(sweep.get_frequencies())
        replaced = sorted(
            [f for f in frequencies if f != 15_000_000] + [15_500_000]
        )
        snapshot = TestDeclaredGridCompleteness._snapshot(replaced, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(10_000_000, 20_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "first missing sample at 15000000 Hz" in result.reason

    @staticmethod
    def test_device_rounding_of_the_declared_grid_is_accepted():
        # The device computes its own integers; they differ from the settings
        # arithmetic in the last hertz.  That is rounding, not a new grid.
        sweep = Sweep(
            500_000, 30_000_000, 101, 10, Properties(logarithmic=True)
        )
        device = TestDeclaredGridCompleteness._device_grid(sweep)
        declared = list(sweep.get_frequencies())
        assert device != declared  # the two really do differ
        assert (
            max(abs(a - b) for a, b in zip(device, declared, strict=True)) <= 1
        )
        snapshot = TestDeclaredGridCompleteness._snapshot(device, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(14_000_000, 15_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_integer_step_truncation_like_a_v2_device_is_accepted():
        # NanoVNA_V2 reports ``start + i * int(step)``, so its grid drifts by
        # up to one hertz per point inside a segment.  That is the device's
        # real grid, not a hole.
        sweep = Sweep(
            500_000, 30_000_000, 101, 10, Properties(logarithmic=True)
        )
        truncated: list[int] = []
        for index in range(sweep.segments):
            start, stop = sweep.get_index_range(index)
            step = int((stop - start) / (sweep.points - 1))
            truncated.extend(
                int(start + offset * step) for offset in range(sweep.points)
            )
        declared = list(sweep.get_frequencies())
        drift = max(
            abs(a - b) for a, b in zip(truncated, declared, strict=True)
        )
        assert drift > 1  # integer step truncation really shifts the grid
        snapshot = TestDeclaredGridCompleteness._snapshot(truncated, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 25_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason
        # A hole in that same device grid is still reported.
        holed = truncated[:300] + truncated[301:]  # inside the rule band
        snapshot = TestDeclaredGridCompleteness._snapshot(holed, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 25_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "missing 1 of" in result.reason

    @staticmethod
    def test_a_hole_outside_the_rule_band_does_not_block_it():
        sweep = Sweep(1_000_000, 10_000_000, 101, 1)
        frequencies = list(sweep.get_frequencies())
        holed = frequencies[:5] + frequencies[8:]
        snapshot = TestDeclaredGridCompleteness._snapshot(holed, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(5_000_000, 10_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_a_band_edge_does_not_split_a_rounded_sample_pair():
        # The device rounds a sample that sits just below the band edge to
        # just above it.  The declared point above the edge may consume it, so
        # the capture is not "extra" merely because its hz value crossed the
        # rule's start.
        sweep = Sweep(
            500_000, 30_000_000, 101, 10, Properties(logarithmic=True)
        )
        declared = list(sweep.get_frequencies())
        device = TestDeclaredGridCompleteness._device_grid(sweep)
        edge_index = next(
            index
            for index, (want, got) in enumerate(
                zip(declared, device, strict=True)
            )
            if want != got and got > want
        )
        assert device[edge_index] > declared[edge_index]
        snapshot = TestDeclaredGridCompleteness._snapshot(device, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(device[edge_index], device[-1]),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_a_sample_outside_the_band_cannot_satisfy_an_in_band_point():
        # The only sample near the in-band declared point sits below the band
        # edge, so the in-band point is genuinely uncovered.
        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        frequencies = [
            freq for freq in sweep.get_frequencies() if freq != 1_000_005
        ]
        snapshot = TestDeclaredGridCompleteness._snapshot(frequencies, sweep)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_005, 1_000_010),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "missing 1 of" in result.reason

    @staticmethod
    def test_metadata_free_clustered_hole_keeps_the_data_only_policy():
        # Without recorded acquisition settings nothing declares the intended
        # grid.  A cluster whose intervals are each justified by their
        # neighbours is accepted, and that limit is documented rather than
        # papered over by inventing a resolution.
        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        full = list(sweep.get_frequencies())
        retained = [
            freq
            for freq in full
            if freq not in {1_000_002, 1_000_003, 1_000_005, 1_000_006}
        ]
        snapshot = TestDeclaredGridCompleteness._snapshot(retained)
        assert snapshot.provenance.sweep is None
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 1_000_010), snapshot
        )
        assert result.verdict is Verdict.PASS

    @staticmethod
    def test_metadata_free_single_hole_is_still_a_hole():
        sweep = Sweep(10_000_000, 20_000_000, 11, 1)
        frequencies = list(sweep.get_frequencies())
        holed = [f for f in frequencies if f != 15_000_000]
        snapshot = TestDeclaredGridCompleteness._snapshot(holed)
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(10_000_000, 20_000_000),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert result.reason.startswith("no samples between")

    @staticmethod
    def test_a_partial_acquisition_is_still_unavailable():
        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        snapshot = snapshot_from_datapoints(
            [Datapoint(freq, 0.1, 0.0) for freq in sweep.get_frequencies()],
            None,
            sweep=sweep,
            completion=CompletionState.PARTIAL,
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 1_000_010),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "partial" in result.reason

    @staticmethod
    def test_a_missing_channel_is_still_unavailable_with_metadata():
        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        snapshot = TestDeclaredGridCompleteness._snapshot(
            sweep.get_frequencies(), sweep
        )
        result = evaluate_rule(
            _rule(
                Metric.S21_INSERTION_LOSS_DB,
                [LimitSegment(1_000_000, 1_000_010, maximum=3.0)],
            ),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "no S21 channel" in result.reason

    @staticmethod
    def test_a_half_step_duplicate_is_ambiguous_not_a_pass():
        # Declared [1_000_000, 1_000_002, 1_000_004, 1_000_006]; the capture
        # reports the midpoint 1_000_003 twice and nothing at 1_000_002 or
        # 1_000_004.  The tolerance cap is exactly half a step, so that single
        # frequency is equally consistent with both declared points.  It must
        # not be credited to one of them and silently pass.
        sweep = Sweep(1_000_000, 1_000_006, 4, 1)
        assert list(sweep.get_frequencies()) == [
            1_000_000,
            1_000_002,
            1_000_004,
            1_000_006,
        ]
        snapshot = TestDeclaredGridCompleteness._snapshot(
            [1_000_000, 1_000_003, 1_000_003, 1_000_006], sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 1_000_006),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert "1000003 Hz" in result.reason
        assert "1000002 Hz" in result.reason or "1 MHz" in result.reason
        assert "cannot be decided" in result.reason
        # The same grid captured faithfully still passes.
        complete = TestDeclaredGridCompleteness._snapshot(
            sweep.get_frequencies(), sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 1_000_006),
            complete,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_an_ambiguous_sample_outside_the_band_does_not_block_it():
        sweep = Sweep(1_000_000, 1_000_006, 4, 1)
        snapshot = TestDeclaredGridCompleteness._snapshot(
            [1_000_000, 1_000_003, 1_000_003, 1_000_006], sweep
        )
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_004, 1_000_006),
            snapshot,
        )
        assert result.verdict is Verdict.UNAVAILABLE  # 1_000_004 still absent
        assert "ambigu" not in result.reason

    @staticmethod
    def test_declared_frequencies_survive_degenerate_point_counts():
        from NanoVNASaver.Measurement import SweepInfo

        # One point per segment divides by zero while generating the grid; the
        # documented fallback must cover grid generation, not only the
        # constructor.
        for info in (
            SweepInfo(start=1_000_000, end=1_000_010, points=1, segments=2),
            SweepInfo(start=1_000_000, end=1_000_010, points=0, segments=1),
            SweepInfo(start=1_000_010, end=1_000_000, points=11, segments=1),
        ):
            assert info.declared_frequencies() == ()

    @staticmethod
    def test_unusable_settings_fall_back_to_the_data_only_policy():
        # Recorded settings that cannot describe a grid (one point per
        # segment) must fall back to the data-only policy, not raise and not
        # declare a grid.
        import dataclasses

        from NanoVNASaver.Measurement import SweepInfo

        sweep = Sweep(1_000_000, 1_000_010, 11, 1)
        unusable = SweepInfo(
            start=1_000_000, end=1_000_010, points=1, segments=2
        )
        base = TestDeclaredGridCompleteness._snapshot(
            sweep.get_frequencies(), sweep
        )
        snapshot = dataclasses.replace(
            base,
            provenance=dataclasses.replace(base.provenance, sweep=unusable),
        )
        assert snapshot.provenance.sweep.declared_frequencies() == ()
        result = evaluate_rule(
            TestDeclaredGridCompleteness._rule(1_000_000, 1_000_010),
            snapshot,
        )
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_known_grids_round_trip_through_declared_frequencies():
        from NanoVNASaver.Measurement import SweepInfo

        sweep = Sweep(
            500_000, 30_000_000, 101, 10, Properties(logarithmic=True)
        )
        info = SweepInfo.from_sweep(sweep)
        assert info.declared_frequencies() == tuple(sweep.get_frequencies())
        assert len(info.declared_frequencies()) == 1010

    @staticmethod
    def test_declared_frequencies_are_empty_for_unusable_settings():
        from NanoVNASaver.Measurement import SweepInfo

        info = SweepInfo(start=1_000_000, end=1_000_010, points=1, segments=1)
        assert info.declared_frequencies() == ()


class TestPiecewiseBands:
    @staticmethod
    def test_a_violation_only_in_the_second_segment_fails():
        freqs = _grid(10_000_000, 20_000_000)
        values = [-30.0] * len(freqs)
        values[-2] = -3.0  # VSWR 1.41 at 19 MHz; bound will be 1.2
        snapshot = _snapshot(_channel(freqs, values))
        rule = _rule(
            Metric.S11_VSWR,
            [
                LimitSegment(10_000_000, 15_000_000, maximum=3.0),
                LimitSegment(15_000_000, 20_000_000, maximum=1.2),
            ],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.FAIL
        assert result.worst_frequency == 19_000_000
        assert [item.verdict for item in result.segments] == [
            Verdict.PASS,
            Verdict.FAIL,
        ]

    @staticmethod
    def test_boundary_sample_belongs_to_the_later_segment():
        # VSWR 2.0 is a pass at the later segment's maximum of 2.0 and would
        # fail the earlier segment's maximum of 1.5.
        points = [
            Datapoint(10_000_000, 0.1, 0.0),
            Datapoint(15_000_000, _magnitude(_db_for_vswr(2.0)), 0.0),
            Datapoint(20_000_000, 0.1, 0.0),
        ]
        snapshot = _snapshot(points)
        rule = _rule(
            Metric.S11_VSWR,
            [
                LimitSegment(10_000_000, 15_000_000, maximum=1.5),
                LimitSegment(15_000_000, 20_000_000, maximum=2.0),
            ],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.PASS
        assert result.segments[0].samples == 1
        assert result.segments[1].samples == 2

    @staticmethod
    def test_the_last_segment_includes_its_stop_frequency():
        points = [
            Datapoint(10_000_000, 0.1, 0.0),
            Datapoint(20_000_000, 0.9, 0.0),
        ]
        snapshot = _snapshot(points)
        result = _evaluate(snapshot, _vswr_rule(2.0))
        assert result.verdict is Verdict.FAIL
        assert result.worst_frequency == 20_000_000

    @staticmethod
    def test_worst_violation_wins_over_the_worst_value():
        # The minimum bound is violated harder than the maximum bound.
        points = [
            Datapoint(10_000_000, _magnitude(-1.0), 0.0),  # RL 1 dB
            Datapoint(11_000_000, _magnitude(-12.0), 0.0),  # RL 12 dB
        ]
        snapshot = _snapshot(points)
        rule = _rule(
            Metric.S11_RETURN_LOSS_DB,
            [LimitSegment(10_000_000, 11_000_000, minimum=6.0, maximum=11.0)],
        )
        result = _evaluate(snapshot, rule)
        assert result.verdict is Verdict.FAIL
        assert result.worst_kind == "minimum"
        assert result.worst_value == pytest.approx(1.0)


class TestSetEvaluation:
    @staticmethod
    def test_every_rule_yields_one_result():
        limit_set = LimitSet(
            "s",
            (
                _rule(
                    Metric.S11_VSWR,
                    [LimitSegment(BAND_START, BAND_STOP, maximum=2.0)],
                    rule_id="a",
                ),
                _rule(
                    Metric.S21_INSERTION_LOSS_DB,
                    [LimitSegment(BAND_START, BAND_STOP, maximum=3.0)],
                    rule_id="b",
                ),
            ),
        )
        snapshot = _snapshot(_channel(_grid(), [-20.0] * len(_grid())))
        results = limit_set.evaluate(snapshot, trace_id="t", trace_name="T")
        assert [item.rule_id for item in results] == ["a", "b"]
        assert [item.verdict for item in results] == [
            Verdict.PASS,
            Verdict.UNAVAILABLE,
        ]
        assert all(item.trace_id == "t" for item in results)

    @staticmethod
    def test_result_dictionary_is_json_safe():
        points = [Datapoint(10_000_000, 1.0, 0.0), Datapoint(11_000_000, 1, 0)]
        result = _evaluate(
            _snapshot(points),
            _rule(
                Metric.S11_VSWR,
                [LimitSegment(10_000_000, 11_000_000, maximum=2.0)],
            ),
        )
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["verdict"] == "fail"
        assert payload["worst_value"] == "inf"
        assert payload["metric_unit"] == ""
