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
"""PRESET-01: structured analysis results against known simulator data."""

import pytest

from NanoVNASaver.Analysis.Results import (
    AnalysisKind,
    AnalysisStatus,
    analyse_cable,
    analyse_cutoff,
    analyse_group_delay,
    analyse_insertion_loss,
    analyse_resonance,
    analyse_snapshot,
    analyse_vswr,
    summarize,
)
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint


def _acquire(scenario, start, stop, points=201):
    vna = make_simulated_vna(scenario)
    vna.datapoints = points
    vna.setSweep(start, stop)
    frequencies = vna.read_frequencies()
    s11 = vna.readValues("data 0")
    s21 = vna.readValues("data 1")
    convert = lambda values: [  # noqa: E731
        Datapoint(freq, value.real, value.imag)
        for freq, value in zip(frequencies, values, strict=True)
    ]
    return snapshot_from_datapoints(convert(s11), convert(s21))


class TestResonanceAnalysis:
    @staticmethod
    def test_resonance_matches_the_modelled_frequency():
        snapshot = _acquire("antenna", 13e6, 15e6)
        result = analyse_resonance(snapshot, trace_name="antenna")
        assert result.available
        assert result.trace_name == "antenna"
        assert result.parameter == "S11"
        resonance = result.value("Resonance")
        assert resonance.available
        # Simulator models a 14.15 MHz resonance.
        assert resonance.frequency == pytest.approx(14_150_000, abs=20_000)

    @staticmethod
    def test_single_port_snapshot_has_no_transmission_analyses():
        snapshot = snapshot_from_datapoints(
            [
                Datapoint(freq, 0.31, -0.44)
                for freq in range(13_000_000, 15_000_001, 1_000_000)
            ],
            None,
        )
        assert snapshot.available == ("S11",)
        for function in (
            analyse_insertion_loss,
            analyse_cutoff,
            analyse_group_delay,
        ):
            result = function(snapshot)
            assert not result.available
            assert result.status is AnalysisStatus.UNAVAILABLE
            assert "S21" in result.reason

    @staticmethod
    def test_missing_data_gives_a_reason_not_zero():
        empty = snapshot_from_datapoints(None, None)
        result = analyse_resonance(empty)
        assert not result.available
        assert result.reason == "no S11 data"
        assert result.values == ()

    @staticmethod
    def test_too_few_points_is_reported():
        snapshot = snapshot_from_datapoints(
            [Datapoint(100, 0.1, 0.0), Datapoint(200, 0.2, 0.0)], None
        )
        result = analyse_resonance(snapshot)
        assert not result.available
        assert "not enough points" in result.reason


class TestVSWRAnalysis:
    @staticmethod
    def test_minimum_vswr_and_bandwidth():
        snapshot = _acquire("antenna", 13e6, 15e6)
        result = analyse_vswr(snapshot, limit=2.0)
        assert result.available
        assert result.parameters["limit"] == 2.0
        minimum = result.value("Minimum VSWR")
        assert minimum.available
        # The model is matched at resonance.
        assert minimum.value == pytest.approx(1.0, abs=1e-3)
        bandwidth = result.value("Bandwidth")
        assert bandwidth.available
        assert bandwidth.value > 0
        start = result.value("Start")
        end = result.value("End")
        assert start.value < minimum.frequency < end.value

    @staticmethod
    def test_bandwidth_matches_the_analytic_crossing_spacing():
        snapshot = _acquire("antenna", 13e6, 15e6, points=401)
        result = analyse_vswr(snapshot, limit=2.0)
        bandwidth = result.value("Bandwidth").value
        # f0 = 14.15 MHz, Q = 20 => BW = 707.5 kHz; VSWR 2 crossings are
        # BW/sqrt(2) = 500.3 kHz apart.  The 5 kHz grid bounds the accuracy.
        assert bandwidth == pytest.approx(500_278, abs=15_000)

    @staticmethod
    def test_no_dip_below_limit_is_unavailable_not_empty_success():
        snapshot = _acquire("filter", 10e6, 200e6)
        result = analyse_vswr(snapshot, limit=1.5)
        assert not result.available
        assert "no dip below VSWR" in result.reason

    @staticmethod
    def test_invalid_limit_is_rejected():
        snapshot = _acquire("antenna", 13e6, 15e6)
        result = analyse_vswr(snapshot, limit=1.0)
        assert not result.available
        assert "greater than 1" in result.reason


class TestTransmissionAnalyses:
    @staticmethod
    def test_cutoff_near_the_butterworth_frequency():
        snapshot = _acquire("filter", 10e6, 200e6, points=401)
        result = analyse_cutoff(snapshot, attenuation=3.0)
        assert result.available
        high = result.value("Passband edge (high)").value
        assert high == pytest.approx(100e6, rel=0.03)

    @staticmethod
    def test_insertion_loss_reports_min_max_and_mean():
        snapshot = _acquire("attenuator", 1e6, 100e6)
        result = analyse_insertion_loss(snapshot)
        assert result.available
        # The attenuator model is 3 dB flat.
        assert result.value("Mean gain").value == pytest.approx(-3.0, abs=0.01)
        assert result.value("Attenuation").value == pytest.approx(3.0, abs=0.01)

    @staticmethod
    def test_group_delay_is_finite_for_a_delaying_cable():
        snapshot = _acquire("cable", 1e6, 100e6, points=401)
        result = analyse_group_delay(snapshot)
        assert result.available
        mean = result.value("Mean group delay").value
        # 10 m at vf 0.66 is 50.54 ns one way.
        assert mean == pytest.approx(50.54e-9, rel=0.05)


class TestCableAnalysis:
    @staticmethod
    def test_cable_reports_impedance_magnitudes_for_a_matched_line():
        snapshot = _acquire("cable", 1e6, 100e6)
        result = analyse_cable(snapshot)
        assert result.available
        assert result.parameters["reference_impedance"] == 50.0
        # The simulated cable returns a 40 dB return loss residual, so |Z|
        # stays close to 50 ohm.
        assert result.value("Mean |Z|").value == pytest.approx(50.0, rel=0.01)

    @staticmethod
    def test_invalid_velocity_factor_is_rejected():
        snapshot = _acquire("cable", 1e6, 100e6)
        result = analyse_cable(snapshot, velocity_factor=0.0)
        assert not result.available
        assert "velocity factor" in result.reason


class TestBundles:
    @staticmethod
    def test_analyse_snapshot_runs_every_requested_kind():
        snapshot = _acquire("filter", 10e6, 200e6, points=201)
        results = analyse_snapshot(
            snapshot,
            [
                AnalysisKind.RESONANCE,
                AnalysisKind.VSWR,
                AnalysisKind.INSERTION_LOSS,
                AnalysisKind.CUTOFF,
                AnalysisKind.GROUP_DELAY,
                AnalysisKind.CABLE,
            ],
            trace_id="t1",
            trace_name="filter",
        )
        assert len(results) == 6
        assert all(result.trace_id == "t1" for result in results)
        assert all(result.trace_name == "filter" for result in results)

    @staticmethod
    def test_results_serialize_for_reports():
        snapshot = _acquire("antenna", 13e6, 15e6)
        payload = analyse_resonance(snapshot).to_dict()
        assert payload["kind"] == "resonance"
        assert payload["status"] == "ok"
        assert payload["values"][0]["label"] == "Resonance"
        assert payload["values"][0]["unit"] == "Hz"

    @staticmethod
    def test_non_finite_values_serialize_symbolically():
        snapshot = snapshot_from_datapoints(
            [Datapoint(100, 0.0, 0.0)], [Datapoint(100, 0.0, 0.0)]
        )
        payload = analyse_group_delay(snapshot).to_dict()
        assert payload["status"] == "unavailable"
        assert payload["reason"]
        # Unavailable results never fabricate a numeric value.
        assert payload["values"] == []

    @staticmethod
    def test_summary_mentions_unavailable_reasons():
        snapshot = _acquire("antenna", 13e6, 15e6)
        summary = summarize([analyse_insertion_loss(snapshot)])
        assert "No analysis available" in summary
        assert "S21" in summary

    @staticmethod
    def test_summary_lists_available_values():
        snapshot = _acquire("antenna", 13e6, 15e6)
        summary = summarize([analyse_resonance(snapshot)])
        assert "resonance" in summary
        assert "MHz" in summary

    @staticmethod
    def test_empty_result_list_is_handled():
        assert summarize([]) == "No analysis results"
