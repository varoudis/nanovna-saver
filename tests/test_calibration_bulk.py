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
"""Bulk calibration correction: numerical equivalence with the scalar oracle.

``Calibration.correct11``/``correct21``/``correct_delay`` stay the reference
implementation.  The bulk path evaluates the same equations over NumPy
arrays; these tests pin that it agrees numerically (including the endpoint
fill outside the calibrated span), that it rejects ambiguous input instead of
broadcasting silently, and that the sweep worker actually uses it.
"""

import math
import threading
from dataclasses import dataclass, field

import numpy as np
import pytest

from NanoVNASaver.Calibration import (
    Calibration,
    arrays_to_datapoints,
    correct_delay,
    correct_delay_bulk,
    datapoints_to_arrays,
)
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import compute_calibration_identity
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.SweepState import SweepState, SweepStateMachine
from NanoVNASaver.SweepWorker import SweepCapture, SweepWorker

TWO_PORT_CAL = "tests/data/test_2port_long.cal"
ONE_PORT_CAL = "tests/data/sol_27_30.cal"
FIXED_1PORT_CAL = "tests/data/test_1port_fixed.cal"

#: Values this far apart are treated as disagreement between the two paths.
VALUE_TOLERANCE = 1e-12


def _calibration(filename: str) -> Calibration:
    calibration = Calibration()
    calibration.load(filename)
    calibration.calc_corrections()
    return calibration


def _span(calibration: Calibration) -> tuple[int, int]:
    frequencies = calibration.dataset.frequencies()
    return frequencies[0], frequencies[-1]


def _grid(
    calibration: Calibration, count: int, *, edge: int = 0
) -> list[Datapoint]:
    """Frequency grid covering the calibrated span.

    The reflection values are deliberately frequency dependent (a resonant
    load with delay, plus a lossy transmission path) so that interpolation
    error, not just the endpoint fill, is exercised.  ``edge`` adds that many
    points below and above the span, which must be held at the endpoints.
    """
    start, stop = _span(calibration)
    step = (stop - start) / max(count - 1, 1)
    frequencies = [round(start + index * step) for index in range(count)]
    if edge:
        below = max(1, round(step))
        frequencies = [
            start - (edge - index) * below for index in range(edge)
        ] + frequencies
        frequencies += [stop + (index + 1) * below for index in range(edge)]
    points = []
    for freq in frequencies:
        phase = -2 * math.pi * freq * 2.4e-9
        magnitude = 0.15 + 0.35 / (1 + ((freq - (start + stop) / 2) / 1e8) ** 2)
        reflection = magnitude * complex(math.cos(phase), math.sin(phase))
        points.append(Datapoint(freq, reflection.real, reflection.imag))
    return points


def _transmission(points: list[Datapoint]) -> list[Datapoint]:
    """A DUT's S21: attenuated, delayed and frequency dependent."""
    result = []
    for point in points:
        phase = -2 * math.pi * point.freq * 4.7e-9
        magnitude = 0.7 - 0.2 / (1 + (point.freq / 1e9))
        value = magnitude * complex(math.cos(phase), math.sin(phase))
        result.append(Datapoint(point.freq, value.real, value.imag))
    return result


def _scalar11(calibration, points) -> np.ndarray:
    return np.array([calibration.correct11(point).z for point in points])


def _scalar21(calibration, points21, points11) -> np.ndarray:
    return np.array(
        [
            calibration.correct21(point, point11).z
            for point, point11 in zip(points21, points11, strict=True)
        ]
    )


class TestCorrect11Bulk:
    @staticmethod
    @pytest.mark.parametrize(
        "filename", (TWO_PORT_CAL, ONE_PORT_CAL, FIXED_1PORT_CAL)
    )
    def test_matches_the_scalar_oracle_inside_the_span(filename):
        calibration = _calibration(filename)
        points = _grid(calibration, 501)
        freq, values = datapoints_to_arrays(points)

        bulk = calibration.correct11_bulk(freq, values)
        scalar = _scalar11(calibration, points)

        assert bulk.shape == scalar.shape
        assert np.max(np.abs(bulk - scalar)) < VALUE_TOLERANCE

    @staticmethod
    def test_matches_the_scalar_oracle_over_large_sweeps():
        calibration = _calibration(TWO_PORT_CAL)
        points = _grid(calibration, 20_000)
        freq, values = datapoints_to_arrays(points)

        bulk = calibration.correct11_bulk(freq, values)
        scalar = _scalar11(calibration, points)

        assert len(bulk) == len(scalar) == 20_000
        assert np.max(np.abs(bulk - scalar)) < VALUE_TOLERANCE

    @staticmethod
    def test_frequencies_outside_the_span_are_held_at_the_edges():
        calibration = _calibration(TWO_PORT_CAL)
        points = _grid(calibration, 9, edge=3)
        start, stop = _span(calibration)
        freq, values = datapoints_to_arrays(points)

        bulk = calibration.correct11_bulk(freq, values)

        low = calibration.dataset.get(start)
        high = calibration.dataset.get(stop)
        for index, point in enumerate(points):
            if point.freq < start:
                expected = (values[index] - low.e00) / (
                    values[index] * low.e11 - low.delta_e
                )
                assert bulk[index] == pytest.approx(expected, abs=1e-15)
            elif point.freq > stop:
                expected = (values[index] - high.e00) / (
                    values[index] * high.e11 - high.delta_e
                )
                assert bulk[index] == pytest.approx(expected, abs=1e-15)

    @staticmethod
    def test_matches_the_scalar_oracle_at_the_exact_edge_frequencies():
        calibration = _calibration(TWO_PORT_CAL)
        start, stop = _span(calibration)
        points = [
            Datapoint(start, 0.31, -0.42),
            Datapoint(stop, -0.12, 0.05),
        ]
        freq, values = datapoints_to_arrays(points)

        bulk = calibration.correct11_bulk(freq, values)

        assert np.allclose(bulk, _scalar11(calibration, points), atol=0, rtol=0)


class TestCorrect21Bulk:
    @staticmethod
    def test_matches_the_scalar_oracle_for_two_port_calibration():
        calibration = _calibration(TWO_PORT_CAL)
        points11 = _grid(calibration, 501)
        points21 = _transmission(points11)
        freq, values11 = datapoints_to_arrays(points11)
        _, values21 = datapoints_to_arrays(points21)

        bulk = calibration.correct21_bulk(freq, values21, values11)
        scalar = _scalar21(calibration, points21, points11)

        assert np.max(np.abs(bulk - scalar)) < VALUE_TOLERANCE

    @staticmethod
    def test_uses_the_raw_port_1_reflection():
        """correct21 expects raw S11; corrected S11 must give a different
        answer, which proves the bulk path does not reuse its own output."""
        calibration = _calibration(TWO_PORT_CAL)
        points11 = _grid(calibration, 101)
        points21 = _transmission(points11)
        freq, raw11 = datapoints_to_arrays(points11)
        _, raw21 = datapoints_to_arrays(points21)

        corrected11 = calibration.correct11_bulk(freq, raw11)
        with_raw = calibration.correct21_bulk(freq, raw21, raw11)
        with_corrected = calibration.correct21_bulk(freq, raw21, corrected11)

        assert np.max(np.abs(with_raw - with_corrected)) > 1e-6

    @staticmethod
    def test_two_port_correction_holds_outside_the_span():
        calibration = _calibration(TWO_PORT_CAL)
        points11 = _grid(calibration, 5, edge=2)
        points21 = _transmission(points11)
        freq, raw11 = datapoints_to_arrays(points11)
        _, raw21 = datapoints_to_arrays(points21)

        bulk = calibration.correct21_bulk(freq, raw21, raw11)
        scalar = _scalar21(calibration, points21, points11)

        assert np.max(np.abs(bulk - scalar)) < VALUE_TOLERANCE


class TestInvalidInput:
    @staticmethod
    def test_uncorrelated_lengths_are_rejected_instead_of_broadcast():
        calibration = _calibration(TWO_PORT_CAL)
        freq = np.array([200_000_000, 201_000_000], dtype=np.int64)
        with pytest.raises(ValueError, match="needs 2 values, got 1"):
            calibration.correct11_bulk(freq, np.array([0.1 + 0.2j]))

        with pytest.raises(ValueError, match="needs 2 values, got 3"):
            calibration.correct21_bulk(
                freq,
                np.array([0.1 + 0.2j, 0.2 + 0.3j]),
                np.array([0.1 + 0.2j, 0.3 + 0.1j, 0.5 + 0.5j]),
            )

    @staticmethod
    def test_two_dimensional_input_is_rejected():
        calibration = _calibration(TWO_PORT_CAL)
        freq = np.array([[200_000_000, 201_000_000]], dtype=np.int64)
        with pytest.raises(ValueError, match="1-D arrays"):
            calibration.correct11_bulk(freq, np.array([[0.1 + 0.2j, 0.3j]]))

    @staticmethod
    def test_empty_input_is_rejected():
        calibration = _calibration(TWO_PORT_CAL)
        with pytest.raises(ValueError, match="at least 1 point"):
            calibration.correct11_bulk(np.array([]), np.array([], complex))

    @staticmethod
    def test_requires_a_calculated_calibration():
        calibration = Calibration()
        calibration.load(TWO_PORT_CAL)
        with pytest.raises(ValueError, match="not calculated"):
            calibration.correct11_bulk(
                np.array([200_000_000]), np.array([0.1 + 0.2j])
            )

    @staticmethod
    def test_undefined_correction_is_reported_not_published():
        calibration = _calibration(TWO_PORT_CAL)
        freq = np.array([200_000_000, 250_000_000], dtype=np.int64)
        values = np.array([0.1 + 0.2j, complex(np.nan, 0.0)])

        with pytest.raises(ValueError, match="250000000 Hz"):
            calibration.correct11_bulk(freq, values)

    @staticmethod
    def test_non_finite_correction_is_reported_not_published():
        """The bulk path must not hand back NaN/inf as corrected data."""
        calibration = _calibration(TWO_PORT_CAL)
        freq = np.array([200_000_000, 250_000_000], dtype=np.int64)
        values = np.array([0.1 + 0.2j, complex(np.nan, 0.0)])

        with pytest.raises(ValueError, match="250000000 Hz"):
            calibration.correct11_bulk(freq, values)

    @staticmethod
    def test_large_but_finite_corrections_are_reported_as_is():
        """Near-singular data yields large numbers in the scalar oracle too;
        the bulk path must not clamp or reject them."""
        calibration = _calibration(TWO_PORT_CAL)
        freq = np.array([200_000_000], dtype=np.int64)
        e11 = np.asarray(calibration.interp["e11"](freq))
        delta_e = np.asarray(calibration.interp["delta_e"](freq))
        values = np.array([delta_e[0] / e11[0]], complex)

        bulk = calibration.correct11_bulk(freq, values)
        scalar = calibration.correct11(
            Datapoint(int(freq[0]), values[0].real, values[0].imag)
        ).z

        assert abs(bulk[0]) > 1e10
        assert bulk[0] == pytest.approx(scalar, rel=1e-12)


class TestDelayBulk:
    @staticmethod
    def test_matches_the_scalar_oracle():
        calibration = _calibration(TWO_PORT_CAL)
        points = _grid(calibration, 257, edge=2)
        freq, values = datapoints_to_arrays(points)

        for reflect in (True, False):
            bulk = correct_delay_bulk(freq, values, 3.3e-11, reflect=reflect)
            scalar = np.array(
                [
                    correct_delay(point, 3.3e-11, reflect=reflect).z
                    for point in points
                ]
            )
            assert np.max(np.abs(bulk - scalar)) < 1e-15

    @staticmethod
    def test_zero_delay_returns_equal_values_without_scaling():
        freq = np.array([1_000_000, 2_000_000], dtype=np.int64)
        values = np.array([0.5 - 0.25j, -0.1 + 0.7j], complex)
        result = correct_delay_bulk(freq, values, 0.0)
        assert np.array_equal(result, values)
        assert result is not values

    @staticmethod
    def test_mismatched_lengths_are_rejected():
        with pytest.raises(ValueError, match="needs 1 frequencies, got 2"):
            correct_delay_bulk(
                np.array([1_000_000], dtype=np.int64),
                np.array([0.5 + 0.1j, 0.2 + 0.2j], complex),
                1e-9,
            )


class TestArrayConversion:
    @staticmethod
    def test_frequencies_survive_the_round_trip_exactly():
        points = [
            Datapoint(freq, 0.1 * index, -0.2 * index)
            for index, freq in enumerate(
                (1, 27_000_001, 300_000_000, 6_000_000_000)
            )
        ]
        freq, values = datapoints_to_arrays(points)
        restored = arrays_to_datapoints(freq, values)

        assert restored == points
        assert all(isinstance(point.freq, int) for point in restored)

    @staticmethod
    def test_round_trip_rejects_non_integer_frequencies():
        with pytest.raises(ValueError, match="integer frequencies"):
            arrays_to_datapoints(
                np.array([1_000_000.5]), np.array([0.1 + 0.2j])
            )

    @staticmethod
    @pytest.mark.parametrize(
        "freq",
        [math.inf, -math.inf, math.nan, 1_000_000.5],
        ids=["inf", "-inf", "nan", "fractional"],
    )
    def test_non_finite_and_fractional_frequencies_are_rejected(freq):
        """A bad frequency must never become an arbitrary integer."""
        with pytest.raises(ValueError):
            arrays_to_datapoints(np.array([freq]), np.array([0.1 + 0.2j]))

    @staticmethod
    @pytest.mark.parametrize(
        "freq",
        [2**63, -(2**63) - 1],
        ids=["too-large", "too-small"],
    )
    def test_frequencies_outside_the_integer_range_are_rejected(freq):
        with pytest.raises(ValueError, match="64-bit integer"):
            arrays_to_datapoints(np.array([freq]), np.array([0.1 + 0.2j]))

    @staticmethod
    def test_float_frequencies_that_are_whole_hertz_are_accepted():
        restored = arrays_to_datapoints(
            np.array([27_000_000.0, 300_000_000.0]),
            np.array([0.1 + 0.2j, 0.3 - 0.4j]),
        )
        assert [point.freq for point in restored] == [27_000_000, 300_000_000]
        assert all(isinstance(point.freq, int) for point in restored)

    @staticmethod
    @pytest.mark.parametrize(
        "freq",
        [math.inf, math.nan, 1_000_000.5, 2**63],
        ids=["inf", "nan", "fractional", "too-large"],
    )
    def test_datapoints_to_arrays_rejects_unrepresentable_frequencies(freq):
        """Conversion into the bulk path must not silently truncate."""
        with pytest.raises(ValueError):
            datapoints_to_arrays([Datapoint(freq, 0.1, 0.2)])


# -- worker integration ------------------------------------------------


@dataclass
class Recorder:
    sample_ready: list = field(default_factory=list)
    run_finished: list = field(default_factory=list)
    run_request_finished: list = field(default_factory=list)
    state_changed: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    finished: int = 0
    sweep_error: int = 0


class Emitter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *payload):
        # ``run_request_finished`` carries two arguments; record every
        # payload shape without discarding the correlation token.
        getattr(self._recorder, self._name).append(
            payload[0] if len(payload) == 1 else payload
        )


class Counter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *_args):
        setattr(
            self._recorder,
            self._name,
            getattr(self._recorder, self._name) + 1,
        )


class Signals:
    def __init__(self, recorder):
        self._recorder = recorder

    state_changed = property(
        lambda self: Emitter(self._recorder, "state_changed")
    )
    progress = property(lambda self: Emitter(self._recorder, "progress"))
    sample_ready = property(
        lambda self: Emitter(self._recorder, "sample_ready")
    )
    run_finished = property(
        lambda self: Emitter(self._recorder, "run_finished")
    )
    run_request_finished = property(
        lambda self: Emitter(self._recorder, "run_request_finished")
    )
    finished = property(lambda self: Counter(self._recorder, "finished"))
    sweep_error = property(lambda self: Counter(self._recorder, "sweep_error"))
    updated = property(lambda self: Counter(self._recorder, "updated"))


@dataclass
class FakeApp:
    vna: object
    sweep: Sweep
    calibration: Calibration


def _make_worker(start=200_000_000, stop=300_000_000, points=101):
    app = FakeApp(
        vna=make_simulated_vna("cable"),
        sweep=Sweep(start=start, end=stop, points=points, segments=1),
        calibration=Calibration(),
    )
    worker = SweepWorker.__new__(SweepWorker)
    worker.app = app
    worker.sweep = Sweep()
    worker.percentage = 0.0
    worker.data11 = []
    worker.data21 = []
    worker.rawData11 = []
    worker.rawData21 = []
    worker.error_message = ""
    worker.offsetDelay = 0.0
    worker._cancel = threading.Event()
    worker._state_machine = SweepStateMachine()
    worker._capture = None
    worker._sweep_counter = 0
    worker._started_at = 0.0
    worker._retry_count = 0
    worker._retry_reason = ""
    worker._last_terminal = None
    recorder = Recorder()
    worker.signals = Signals(recorder)
    return worker, recorder


def _oracle(calibration, raw11, raw21, delay):
    """Scalar reference for the worker's corrected channels."""
    data11 = [
        correct_delay(calibration.correct11(point), delay, reflect=True)
        for point in raw11
    ]
    data21 = [
        correct_delay(calibration.correct21(point21, point11), delay)
        for point21, point11 in zip(raw21, raw11, strict=True)
    ]
    return data11, data21


class TestWorkerIntegration:
    @staticmethod
    def test_sweep_matches_the_scalar_oracle_with_offset_delay():
        worker, recorder = _make_worker()
        calibration = _calibration(TWO_PORT_CAL)
        worker.app.calibration = calibration
        worker.offsetDelay = 4.0e-11

        worker.run_once()

        assert recorder.run_finished, "sweep did not finish"
        raw11 = worker.rawData11
        raw21 = worker.rawData21
        expected11, expected21 = _oracle(
            calibration, raw11, raw21, worker.offsetDelay
        )
        assert len(worker.data11) == len(expected11) == len(raw11)
        assert worker.data11[0].freq == raw11[0].freq
        for actual, expected in zip(worker.data11, expected11, strict=True):
            assert actual.freq == expected.freq
            assert actual.z == pytest.approx(expected.z, abs=1e-12)
        for actual, expected in zip(worker.data21, expected21, strict=True):
            assert actual.freq == expected.freq
            assert actual.z == pytest.approx(expected.z, abs=1e-12)

        snapshot = recorder.sample_ready[-1].snapshot
        assert snapshot.correction.applied is True
        assert snapshot.correction.offset_delay == pytest.approx(4.0e-11)
        assert snapshot.correction.identity
        assert snapshot.correction.source.value == "application"

    @staticmethod
    def test_raw_provenance_stays_uncalibrated():
        worker, recorder = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        worker.run_once()

        snapshot = recorder.sample_ready[-1].snapshot
        raw = snapshot.channel("S11", corrected=False)
        corrected = snapshot.channel("S11")
        assert raw == tuple(worker.rawData11)
        assert corrected != raw

    @staticmethod
    def test_apply_calibration_uses_raw_s11_for_s21():
        worker, _ = _make_worker()
        calibration = _calibration(TWO_PORT_CAL)
        worker.app.calibration = calibration
        freq = list(range(200_000_000, 300_000_001, 1_000_000))
        raw11 = [
            Datapoint(
                value, 0.3 * math.cos(value / 1e8), 0.2 * math.sin(value / 1e8)
            )
            for value in freq
        ]
        raw21 = [
            Datapoint(value, 0.5 * math.cos(value / 2e8), -0.4)
            for value in freq
        ]

        data11, data21 = worker.applyCalibration(raw11, raw21)

        expected11 = [calibration.correct11(point).z for point in raw11]
        expected21 = [
            calibration.correct21(point21, point11).z
            for point21, point11 in zip(raw21, raw11, strict=True)
        ]
        assert [point.freq for point in data11] == freq
        assert [point.freq for point in data21] == freq
        assert (
            np.max(np.abs(np.array([point.z for point in data11]) - expected11))
            < VALUE_TOLERANCE
        )
        assert (
            np.max(np.abs(np.array([point.z for point in data21]) - expected21))
            < VALUE_TOLERANCE
        )

    @staticmethod
    def test_one_port_calibration_leaves_s21_uncorrected():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(ONE_PORT_CAL)
        freq = [27_000_000, 28_000_000, 30_000_000]
        raw11 = [Datapoint(value, 0.4, -0.1) for value in freq]
        raw21 = [Datapoint(value, 0.9, 0.05) for value in freq]

        data11, data21 = worker.applyCalibration(raw11, raw21)

        assert [point.z for point in data21] == [point.z for point in raw21]
        assert [point.z for point in data11] != [point.z for point in raw11]

    @staticmethod
    def test_uncalculated_calibration_passes_samples_through():
        worker, _ = _make_worker()
        calibration = Calibration()
        calibration.load(TWO_PORT_CAL)
        worker.app.calibration = calibration
        freq = [200_000_000, 250_000_000]
        raw11 = [Datapoint(value, 0.4, -0.1) for value in freq]
        raw21 = [Datapoint(value, 0.9, 0.05) for value in freq]

        data11, data21 = worker.applyCalibration(raw11, raw21)

        assert data11 == raw11
        assert data21 == raw21
        assert data11 is not raw11

    @staticmethod
    def test_mismatched_channels_are_rejected():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        with pytest.raises(ValueError, match="S21 has 1"):
            worker.applyCalibration(
                [
                    Datapoint(200_000_000, 0.1, 0.1),
                    Datapoint(201_000_000, 0.2, 0.2),
                ],
                [Datapoint(200_000_000, 0.1, 0.1)],
            )

    @staticmethod
    def test_shifted_frequency_grid_is_rejected():
        """Equal lengths but different frequencies must not be paired."""
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        with pytest.raises(ValueError, match="frequency grids differ"):
            worker.applyCalibration(
                [
                    Datapoint(200_000_000, 0.1, 0.1),
                    Datapoint(201_000_000, 0.2, 0.2),
                ],
                [
                    Datapoint(200_000_000, 0.3, 0.3),
                    Datapoint(202_000_000, 0.4, 0.4),
                ],
            )

    @staticmethod
    def test_empty_input_returns_empty_channels():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        assert worker.applyCalibration([], []) == ([], [])

    @staticmethod
    def test_one_port_sweep_without_s21_corrects_s11_only():
        worker, _ = _make_worker()
        calibration = _calibration(TWO_PORT_CAL)
        worker.app.calibration = calibration
        freq = [200_000_000, 250_000_000, 300_000_000]
        raw11 = [Datapoint(value, 0.4, -0.1) for value in freq]

        data11, data21 = worker.applyCalibration(raw11, [])

        expected = [calibration.correct11(point).z for point in raw11]
        assert data21 == []
        assert [point.freq for point in data11] == freq
        assert (
            np.max(np.abs(np.array([point.z for point in data11]) - expected))
            < VALUE_TOLERANCE
        )

    @staticmethod
    def test_s21_without_raw_s11_is_rejected_for_a_two_port_calibration():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        with pytest.raises(ValueError, match="matching raw S11"):
            worker.applyCalibration([], [Datapoint(200_000_000, 0.5, 0.1)])

    @staticmethod
    def test_s21_without_raw_s11_passes_through_for_a_one_port_calibration():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(ONE_PORT_CAL)
        raw21 = [Datapoint(27_000_000, 0.5, 0.1)]

        data11, data21 = worker.applyCalibration([], raw21)

        assert data11 == []
        assert data21 == raw21

    @staticmethod
    def test_captured_offset_delay_wins_over_the_live_value():
        worker, _ = _make_worker()
        calibration = _calibration(TWO_PORT_CAL)
        worker.app.calibration = calibration
        worker.offsetDelay = 1.0e-9
        capture = SweepCapture(
            sweep_id=1,
            sweep=Sweep(start=200_000_000, end=300_000_000, points=3),
            calibration=calibration,
            offset_delay=2.0e-11,
            averages=1,
            truncates=0,
            continuous=False,
        )
        freq = [200_000_000, 250_000_000, 300_000_000]
        raw11 = [Datapoint(value, 0.4, -0.1) for value in freq]
        raw21 = [Datapoint(value, 0.9, 0.05) for value in freq]

        data11, _ = worker.applyCalibration(raw11, raw21, capture=capture)

        expected11, _ = _oracle(calibration, raw11, raw21, 2.0e-11)
        for actual, expected in zip(data11, expected11, strict=True):
            assert actual.z == pytest.approx(expected.z, abs=1e-12)


class TestDelayOnlyCorrection:
    """An offset delay is a correction in its own right.

    Regression: applying only an offset delay (no calculated calibration)
    changed the published values while the snapshot claimed no correction
    was applied and reported a zero delay.
    """

    @staticmethod
    def test_delay_only_sweep_publishes_the_applied_delay():
        worker, recorder = _make_worker()
        worker.offsetDelay = 1.0e-9

        worker.run_once()

        assert worker.state is SweepState.COMPLETED
        snapshot = recorder.sample_ready[-1].snapshot
        correction = snapshot.correction
        assert correction.applied is True
        assert correction.offset_delay == pytest.approx(1.0e-9)
        # No calibration took part, so there is no calibration identity.
        assert correction.identity == ""
        assert correction.source.value == "application"

    @staticmethod
    def test_delay_only_sweep_corrects_the_samples():
        worker, recorder = _make_worker()
        worker.offsetDelay = 1.0e-9

        worker.run_once()

        snapshot = recorder.sample_ready[-1].snapshot
        raw = snapshot.channel("S11", corrected=False)
        corrected = snapshot.channel("S11")
        assert raw != corrected, "offset delay did not change the samples"
        expected = [correct_delay(point, 1.0e-9, reflect=True) for point in raw]
        for actual, reference in zip(corrected, expected, strict=True):
            assert actual.freq == reference.freq
            assert actual.z == pytest.approx(reference.z, abs=1e-15)

    @staticmethod
    def test_no_calibration_and_no_delay_reports_no_correction():
        worker, recorder = _make_worker()

        worker.run_once()

        correction = recorder.sample_ready[-1].snapshot.correction
        assert correction.applied is False
        assert correction.identity == ""
        assert correction.source.value == "none"
        assert correction.offset_delay == 0.0

    @staticmethod
    def test_calibration_and_delay_report_both():
        worker, recorder = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        worker.offsetDelay = 4.0e-11

        worker.run_once()

        correction = recorder.sample_ready[-1].snapshot.correction
        assert correction.applied is True
        assert correction.identity
        assert correction.offset_delay == pytest.approx(4.0e-11)

    @staticmethod
    @pytest.mark.parametrize(
        "delay",
        [math.inf, -math.inf, math.nan],
        ids=["inf", "-inf", "nan"],
    )
    def test_non_finite_delay_fails_without_publishing(delay):
        """A non-finite delay must not publish a COMPLETE NaN measurement."""
        worker, recorder = _make_worker()
        worker.offsetDelay = delay

        worker.run_once()

        assert worker.state is SweepState.FAILED
        assert recorder.run_finished == [SweepState.FAILED]
        assert recorder.sweep_error == 1
        assert recorder.sample_ready == []
        assert "finite" in worker.error_message

    @staticmethod
    def test_delay_that_overflows_the_phase_term_is_rejected():
        """A finite but absurd delay still cannot produce finite samples."""
        worker, _ = _make_worker()

        with pytest.raises(ValueError, match="non-finite"):
            worker.applyCalibration(
                [Datapoint(200_000_000, 0.4, -0.1)],
                [],
                capture=_capture(1e300),
            )

    @staticmethod
    def test_overflowing_delay_does_not_publish_a_measurement():
        worker, recorder = _make_worker()
        worker.offsetDelay = 1e300

        worker.run_once()

        assert worker.state is SweepState.FAILED
        assert recorder.sample_ready == []

    @staticmethod
    @pytest.mark.parametrize(
        "calibration_file", [None, ONE_PORT_CAL, TWO_PORT_CAL]
    )
    def test_non_finite_delay_is_rejected_with_and_without_calibration(
        calibration_file,
    ):
        worker, _ = _make_worker()
        if calibration_file:
            worker.app.calibration = _calibration(calibration_file)
        raw11, raw21 = _measured_grid()

        with pytest.raises(ValueError, match="finite"):
            worker.applyCalibration(raw11, raw21, capture=_capture(math.inf))

    @staticmethod
    @pytest.mark.parametrize(
        ("raw11", "raw21"),
        [([], []), ([Datapoint(200_000_000, 0.4, -0.1)], [])],
        ids=["empty", "one-channel"],
    )
    def test_delay_validation_covers_empty_and_single_channel_input(
        raw11, raw21
    ):
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)

        with pytest.raises(ValueError, match="finite"):
            worker.applyCalibration(raw11, raw21, capture=_capture(math.nan))

    @staticmethod
    def test_captured_non_finite_delay_wins_over_the_live_value():
        """The captured delay is what gets validated, not the live one."""
        worker, _ = _make_worker()
        worker.offsetDelay = 1.0e-9

        with pytest.raises(ValueError, match="finite"):
            worker.applyCalibration(
                [Datapoint(200_000_000, 0.4, -0.1)],
                [],
                capture=_capture(math.inf),
            )


class TestCapturedCalibrationIsolation:
    """A run keeps one coherent calibration for every published sample.

    Regression: ``SweepCapture.calibration`` referenced the live object, so
    reloading or recalculating the calibration in place during a run changed
    the correction outputs, the validity and the published identity of
    samples that had already been acquired.
    """

    @staticmethod
    def test_in_place_reload_and_recalculate_cannot_change_a_run():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        worker._capture = worker._capture_settings()
        capture = worker._capture
        assert capture is not None
        raw11, raw21 = _measured_grid()

        before = worker.applyCalibration(raw11, raw21, capture=capture)
        before_identity = worker._correction_state(capture).identity

        # The user reloads and recalculates the *same* calibration object
        # through the public API while the run is in progress.
        worker.app.calibration.load(ONE_PORT_CAL)
        worker.app.calibration.calc_corrections()

        after = worker.applyCalibration(raw11, raw21, capture=capture)
        after_state = worker._correction_state(capture)

        assert after == before
        assert after_state.identity == before_identity
        assert capture.calibration.isValid2Port()
        assert not worker.app.calibration.isValid2Port()

    @staticmethod
    def test_in_place_reload_mid_sweep_keeps_published_identity():
        worker, recorder = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        worker.app.sweep.update(200_000_000, 300_000_000, 2, 11)

        original_update = worker.update_data
        calls = {"count": 0}

        def update_data(frequencies, values11, values21, index):
            calls["count"] += 1
            if calls["count"] == 1:
                # Reload + recalculate the live object during the run.
                worker.app.calibration.load(ONE_PORT_CAL)
                worker.app.calibration.calc_corrections()
            original_update(frequencies, values11, values21, index)

        worker.update_data = update_data

        worker.run_once()

        assert worker.state is SweepState.COMPLETED
        snapshots = [sample.snapshot for sample in recorder.sample_ready]
        assert len(snapshots) == 2
        identities = {snapshot.correction.identity for snapshot in snapshots}
        provenance = {
            snapshot.provenance.calibration.identity for snapshot in snapshots
        }
        assert len(identities) == 1
        assert identities == provenance
        for snapshot in snapshots:
            assert snapshot.correction.applied is True
            assert snapshot.correction.offset_delay == 0.0

    @staticmethod
    def test_a_later_run_uses_the_newly_selected_calibration():
        worker, recorder = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        worker.run_once()
        first = recorder.sample_ready[-1].snapshot.correction.identity

        worker.app.calibration = _calibration(ONE_PORT_CAL)
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        second = snapshot.correction.identity

        assert first and second and first != second
        assert snapshot.provenance.calibration.one_port
        assert not snapshot.provenance.calibration.two_port

    @staticmethod
    def test_capture_does_not_alias_the_live_calibration():
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)

        capture = worker._capture_settings()

        assert capture.calibration is not worker.app.calibration
        assert capture.calibration.dataset is not worker.app.calibration.dataset

    @staticmethod
    def test_freezing_a_calibration_preserves_its_numbers():
        calibration = _calibration(TWO_PORT_CAL)
        frozen = calibration.frozen_copy()
        points = _grid(calibration, 33)
        freq, values = datapoints_to_arrays(points)

        assert np.array_equal(
            frozen.correct11_bulk(freq, values),
            calibration.correct11_bulk(freq, values),
        )
        assert frozen.isValid2Port() == calibration.isValid2Port()
        assert compute_calibration_identity(
            frozen
        ) == compute_calibration_identity(calibration)

    @staticmethod
    def test_captureless_recorrection_uses_the_live_calibration():
        """The calibration window re-corrects raw data with current settings."""
        worker, _ = _make_worker()
        worker.app.calibration = _calibration(TWO_PORT_CAL)
        worker._capture = worker._capture_settings()
        raw11, raw21 = _measured_grid()

        worker.app.calibration = _calibration(ONE_PORT_CAL)

        data11, data21 = worker.applyCalibration(raw11, raw21)

        expected11 = [
            worker.app.calibration.correct11(point) for point in raw11
        ]
        for actual, expected in zip(data11, expected11, strict=True):
            assert actual.z == pytest.approx(expected.z, abs=1e-12)
        # The one-port calibration leaves S21 untouched.
        assert [point.z for point in data21] == [point.z for point in raw21]


def _capture(offset_delay: float) -> SweepCapture:
    return SweepCapture(
        sweep_id=1,
        sweep=Sweep(start=200_000_000, end=300_000_000, points=3),
        calibration=Calibration(),
        offset_delay=offset_delay,
        averages=1,
        truncates=0,
        continuous=False,
    )


def _measured_grid() -> tuple[list[Datapoint], list[Datapoint]]:
    frequencies = [200_000_000, 250_000_000, 300_000_000]
    raw11 = [
        Datapoint(freq, 0.4 - index * 0.05, -0.1 + index * 0.02)
        for index, freq in enumerate(frequencies)
    ]
    raw21 = [
        Datapoint(freq, 0.9 - index * 0.1, 0.05 + index * 0.03)
        for index, freq in enumerate(frequencies)
    ]
    return raw11, raw21
