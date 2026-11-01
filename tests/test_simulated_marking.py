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
"""SIM-02.2: simulated data stays identifiable after saving and reopening."""

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Touchstone import SIMULATED_MARKER, Touchstone


def _simulated_touchstone(scenario="antenna"):
    vna = make_simulated_vna(scenario)
    vna.setSweep(13_000_000, 15_000_000)
    frequencies = vna.read_frequencies()
    values = vna.readValues("data 0")
    dataset = Touchstone()
    dataset.s11 = [
        Datapoint(freq, value.real, value.imag)
        for freq, value in zip(frequencies, values, strict=True)
    ]
    dataset.simulated = True
    dataset.simulated_scenario = vna.scenario_key
    return dataset


class TestTouchstoneMarking:
    @staticmethod
    def test_simulated_data_is_marked_in_the_file():
        text = _simulated_touchstone().saves(1)
        markers = [
            line for line in text.splitlines() if SIMULATED_MARKER in line
        ]
        assert len(markers) == 1
        assert markers[0].startswith("!")
        assert "not a physical measurement" in markers[0]

    @staticmethod
    def test_the_marker_names_the_scenario():
        text = _simulated_touchstone("filter").saves(1)
        assert "scenario filter" in text

    @staticmethod
    def test_the_marker_survives_a_round_trip():
        text = _simulated_touchstone().saves(1)
        reloaded = Touchstone()
        reloaded.loads(text)
        assert reloaded.simulated is True
        assert reloaded.simulated_scenario == "antenna"
        assert len(reloaded.s11) == 101

    @staticmethod
    def test_re_export_does_not_stack_markers(tmp_path):
        source = _simulated_touchstone()
        first = source.saves(1)
        reloaded = Touchstone()
        reloaded.loads(first)
        second = reloaded.saves(1)
        assert second.count(SIMULATED_MARKER) == 1

    @staticmethod
    def test_physical_data_carries_no_simulated_marker():
        dataset = Touchstone()
        dataset.s11 = [Datapoint(1_000_000, 0.1, 0.2)]
        text = dataset.saves(1)
        assert "SIMULATED" not in text

    @staticmethod
    def test_saved_file_keeps_the_marker_on_disk(tmp_path):
        dataset = _simulated_touchstone()
        dataset.filename = str(tmp_path / "demo.s1p")
        dataset.save(1)
        text = (tmp_path / "demo.s1p").read_text(encoding="utf-8")
        assert SIMULATED_MARKER in text
        again = Touchstone()
        again.loads(text)
        assert again.simulated


class TestSnapshotMarking:
    @staticmethod
    def test_snapshot_records_simulation_and_scenario():
        vna = make_simulated_vna("cable")
        vna.setSweep(1_000_000, 100_000_000)
        frequencies = vna.read_frequencies()
        points = [
            Datapoint(freq, value.real, value.imag)
            for freq, value in zip(
                frequencies, vna.readValues("data 0"), strict=True
            )
        ]
        snapshot = snapshot_from_datapoints(
            points,
            None,
            simulated=True,
            simulator_scenario=vna.scenario_key,
            source="simulated",
        )
        assert snapshot.simulated is True
        assert snapshot.provenance.simulator_scenario == "cable"

    @staticmethod
    def test_provenance_round_trips_through_a_dict():
        snapshot = snapshot_from_datapoints(
            [Datapoint(1_000_000, 0.1, 0.2)],
            None,
            simulated=True,
            simulator_scenario="filter",
        )
        from NanoVNASaver.Measurement import Provenance

        restored = Provenance.from_dict(snapshot.provenance.to_dict())
        assert restored.simulated is True
        assert restored.simulator_scenario == "filter"


class TestScenarioIdentity:
    @staticmethod
    def test_every_scenario_exposes_a_stable_key():
        for name in ("antenna", "filter", "cable", "attenuator"):
            vna = make_simulated_vna(name)
            assert vna.scenario_key == name

    @staticmethod
    def test_unknown_scenarios_are_rejected():
        with pytest.raises(ValueError, match="unknown scenario"):
            make_simulated_vna("nonexistent")

    @staticmethod
    def test_simulated_device_reports_itself():
        vna = make_simulated_vna("antenna")
        assert vna.simulated is True
        assert "simulat" in vna.name.lower()
        assert "scenario" in vna.readFirmware().lower()
