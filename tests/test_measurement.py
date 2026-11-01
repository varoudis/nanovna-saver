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
"""CORE-01: snapshot immutability, provenance and channel semantics."""

import math

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Measurement import (
    S11,
    S21,
    CalibrationIdentity,
    CompletionState,
    CorrectionAlreadyAppliedError,
    CorrectionSource,
    CorrectionState,
    MeasurementSnapshot,
    Provenance,
    SweepInfo,
    compute_calibration_identity,
    missing_parameters,
    snapshot_from_datapoints,
    snapshot_from_touchstone,
    snapshot_to_touchstone,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.Touchstone import Touchstone


def _points(freqs=(100, 200, 300), value=(0.5, 0.25)):
    return [Datapoint(freq, value[0], value[1]) for freq in freqs]


class TestSnapshotBasics:
    @staticmethod
    def test_available_channels_are_canonical_order():
        snapshot = snapshot_from_datapoints(_points(), _points(), name="both")
        assert snapshot.available == (S11, S21)

    @staticmethod
    def test_absent_channel_is_none_not_zeros():
        snapshot = snapshot_from_datapoints(_points(), None, name="one-port")
        assert snapshot.available == (S11,)
        assert snapshot.channel(S21) is None
        assert not snapshot.has(S21)
        assert S21 in missing_parameters(snapshot)

    @staticmethod
    def test_measured_zero_is_a_real_value():
        zeros = [Datapoint(100, 0.0, 0.0)]
        snapshot = snapshot_from_datapoints(zeros, None)
        assert snapshot.has(S11)
        assert snapshot.channel(S11)[0].z == 0

    @staticmethod
    def test_snapshot_is_independent_of_source_buffers():
        mutable = _points()
        snapshot = snapshot_from_datapoints(mutable, None)
        mutable[0] = Datapoint(999, 9.0, 9.0)
        mutable.append(Datapoint(400, 1.0, 1.0))
        assert len(snapshot.channel(S11)) == 3
        assert snapshot.channel(S11)[0].freq == 100

    @staticmethod
    def test_channel_mapping_cannot_be_mutated_from_outside():
        snapshot = snapshot_from_datapoints(_points(), None)
        with pytest.raises(TypeError):
            snapshot.corrected[S21] = ()  # type: ignore[index]

    @staticmethod
    def test_frequencies_are_derived_when_not_given():
        snapshot = snapshot_from_datapoints(_points(), None)
        assert snapshot.frequencies == (100, 200, 300)

    @staticmethod
    def test_name_change_returns_a_new_snapshot():
        snapshot = snapshot_from_datapoints(_points(), None, name="a")
        renamed = snapshot.with_name("b")
        assert renamed.name == "b"
        assert snapshot.name == "a"
        assert renamed.id == snapshot.id


class TestProvenance:
    @staticmethod
    def test_simulated_measurement_is_marked():
        snapshot = snapshot_from_datapoints(
            _points(),
            None,
            simulated=True,
            simulator_scenario="antenna",
            device="Simulated VNA",
        )
        assert snapshot.simulated
        assert snapshot.provenance.simulator_scenario == "antenna"
        assert snapshot.provenance.device == "Simulated VNA"

    @staticmethod
    def test_unknown_metadata_is_explicit_not_blank_success():
        snapshot = snapshot_from_datapoints(_points(), None)
        assert snapshot.provenance.source == "unknown"
        assert snapshot.provenance.device == "unknown"
        assert snapshot.provenance.calibration is None

    @staticmethod
    def test_sweep_and_calibration_metadata_is_captured():
        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        sweep = Sweep(start=27_000_000, end=30_000_000, points=101)
        snapshot = snapshot_from_datapoints(
            _points(), None, sweep=sweep, calibration=calibration
        )
        assert snapshot.provenance.sweep.start == 27_000_000
        assert snapshot.provenance.sweep.points == 101
        assert snapshot.provenance.calibration.one_port
        assert not snapshot.provenance.calibration.two_port
        assert snapshot.provenance.calibration.coverage_spans == (
            (27_000_000, 30_000_000),
        )

    @staticmethod
    def test_calibration_identity_is_stable_for_same_content():
        first = Calibration()
        first.load("tests/data/sol_27_30.cal")
        second = Calibration()
        second.load("tests/data/sol_27_30.cal")
        assert compute_calibration_identity(
            first
        ) == compute_calibration_identity(second)

    @staticmethod
    def test_calibration_identity_changes_with_content():
        first = Calibration()
        first.load("tests/data/sol_27_30.cal")
        digest = compute_calibration_identity(first)
        first.dataset.data[28_000_000].short = complex(0.5, 0.5)
        assert compute_calibration_identity(first) != digest

    @staticmethod
    def test_completion_state_round_trips():
        snapshot = snapshot_from_datapoints(
            _points(), None, completion=CompletionState.PARTIAL
        )
        assert (
            MeasurementSnapshot.from_dict(
                snapshot.to_dict()
            ).provenance.completion
            is CompletionState.PARTIAL
        )


class TestCorrectionIdentity:
    @staticmethod
    def test_corrected_channels_carry_the_correction_identity():
        snapshot = snapshot_from_datapoints(_points(), None)
        corrected = snapshot.mark_corrected(
            "cal-1", {S11: _points(value=(0.1, 0.1))}
        )
        assert corrected.correction.applied
        assert corrected.correction.identity == "cal-1"
        assert corrected.correction.source is CorrectionSource.APPLICATION
        assert corrected.channel(S11)[0].z == complex(0.1, 0.1)
        # Raw samples are untouched.
        assert corrected.channel(S11, corrected=False)[0].z == complex(
            0.5, 0.25
        )

    @staticmethod
    def test_same_correction_cannot_be_applied_twice():
        snapshot = snapshot_from_datapoints(_points(), None)
        once = snapshot.mark_corrected("cal-1", {S11: _points()})
        with pytest.raises(CorrectionAlreadyAppliedError):
            once.mark_corrected("cal-1", {S11: _points()})

    @staticmethod
    def test_a_different_correction_may_be_applied():
        snapshot = snapshot_from_datapoints(_points(), None)
        corrected = snapshot.mark_corrected("cal-2", {S11: _points()})
        assert corrected.correction.identity == "cal-2"

    @staticmethod
    def test_uncorrected_state_is_explicit():
        state = CorrectionState()
        assert not state.applied
        assert state.source is CorrectionSource.NONE


class TestTouchstoneAdapters:
    @staticmethod
    def test_s1p_import_keeps_only_s11():
        touchstone = Touchstone("./tests/data/valid.s1p")
        touchstone.load()
        snapshot = snapshot_from_touchstone(touchstone)
        snapshot.validate()
        assert snapshot.available == (S11,)
        assert S21 in missing_parameters(snapshot)

    @staticmethod
    def test_s2p_import_keeps_all_present_channels():
        touchstone = Touchstone("./tests/data/valid.s2p")
        touchstone.load()
        snapshot = snapshot_from_touchstone(touchstone)
        assert snapshot.available == (S11, S21, "S12", "S22")
        assert len(snapshot.channel(S21)) == len(touchstone.s11)

    @staticmethod
    def test_import_preserves_reference_impedance():
        touchstone = Touchstone("./tests/data/valid.s2p")
        touchstone.load()
        snapshot = snapshot_from_touchstone(touchstone)
        assert snapshot.reference_impedance == touchstone.opts.resistance

    @staticmethod
    def test_round_trip_through_touchstone_adapters():
        touchstone = Touchstone("./tests/data/valid.s2p")
        touchstone.load()
        snapshot = snapshot_from_touchstone(touchstone)
        exported = snapshot_to_touchstone(snapshot)
        assert exported.s11[0].z == touchstone.s11[0].z
        assert exported.s22[-1].z == touchstone.s22[-1].z

    @staticmethod
    def test_missing_channels_are_not_invented_on_export():
        touchstone = Touchstone("./tests/data/valid.s1p")
        touchstone.load()
        exported = snapshot_to_touchstone(snapshot_from_touchstone(touchstone))
        assert exported.s11
        assert not exported.s21
        assert not exported.s12
        assert not exported.s22


class TestSerialization:
    @staticmethod
    def test_dict_round_trip_is_exact():
        snapshot = snapshot_from_datapoints(
            _points(), _points(), name="rt", simulated=True
        )
        restored = MeasurementSnapshot.from_dict(snapshot.to_dict())
        assert restored.available == snapshot.available
        assert restored.name == snapshot.name
        assert restored.simulated
        for parameter in snapshot.available:
            assert restored.channel(parameter) == snapshot.channel(parameter)

    @staticmethod
    def test_non_finite_values_survive_round_trip():
        points = [
            Datapoint(100, math.nan, math.inf),
            Datapoint(200, 1.0, -math.inf),
        ]
        snapshot = snapshot_from_datapoints(points, None)
        restored = MeasurementSnapshot.from_dict(snapshot.to_dict())
        first = restored.channel(S11)[0]
        assert math.isnan(first.re)
        assert math.isinf(first.im)
        assert restored.channel(S11)[1].im == -math.inf

    @staticmethod
    def test_timestamp_keeps_timezone():
        snapshot = snapshot_from_datapoints(_points(), None)
        restored = MeasurementSnapshot.from_dict(snapshot.to_dict())
        assert restored.timestamp.tzinfo is not None
        assert restored.timestamp == snapshot.timestamp

    @staticmethod
    def test_future_format_version_is_rejected():
        payload = snapshot_from_datapoints(_points(), None).to_dict()
        payload["format_version"] = 99
        with pytest.raises(ValueError, match="newer than supported"):
            MeasurementSnapshot.from_dict(payload)


class TestValidation:
    @staticmethod
    def test_empty_channel_is_rejected():
        snapshot = MeasurementSnapshot(
            id="x",
            name="x",
            timestamp=snapshot_from_datapoints(_points(), None).timestamp,
            reference_impedance=50.0,
            raw={S11: ()},
            corrected={},
            provenance=Provenance(),
        )
        with pytest.raises(ValueError, match="empty channel"):
            snapshot.validate()

    @staticmethod
    def test_unsupported_parameter_is_rejected():
        snapshot = MeasurementSnapshot(
            id="x",
            name="x",
            timestamp=snapshot_from_datapoints(_points(), None).timestamp,
            reference_impedance=50.0,
            raw={"S33": (Datapoint(1, 0.0, 0.0),)},
            corrected={},
            provenance=Provenance(),
        )
        with pytest.raises(ValueError, match="unsupported parameter"):
            snapshot.validate()

    @staticmethod
    def test_unordered_channel_is_rejected():
        snapshot = MeasurementSnapshot(
            id="x",
            name="x",
            timestamp=snapshot_from_datapoints(_points(), None).timestamp,
            reference_impedance=50.0,
            raw={S11: _points(freqs=(300, 200, 100))},
            corrected={},
            provenance=Provenance(),
        )
        with pytest.raises(ValueError, match="not frequency ordered"):
            snapshot.validate()

    @staticmethod
    def test_non_positive_reference_impedance_is_rejected():
        snapshot = MeasurementSnapshot(
            id="x",
            name="x",
            timestamp=snapshot_from_datapoints(_points(), None).timestamp,
            reference_impedance=0.0,
            raw={S11: tuple(_points())},
            corrected={},
            provenance=Provenance(),
        )
        with pytest.raises(ValueError, match="reference impedance"):
            snapshot.validate()


class TestSupportTypes:
    @staticmethod
    def test_sweep_info_from_avg_sweep():
        from NanoVNASaver.Settings.Sweep import Properties, SweepMode

        sweep = Sweep(
            points=11,
            properties=Properties(mode=SweepMode.AVERAGE, averages=(5, 2)),
        )
        info = SweepInfo.from_sweep(sweep)
        assert info.mode == "AVERAGE"
        assert info.averages == 5
        assert info.truncates == 2

    @staticmethod
    def test_calibration_identity_available_flag():
        assert not CalibrationIdentity().available
        assert CalibrationIdentity(one_port=True).available

    @staticmethod
    def test_provenance_defaults_to_complete():
        assert Provenance().completion is CompletionState.COMPLETE
