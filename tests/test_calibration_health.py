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
"""CAL-02: calibration age, metadata and verification.

The verification tests build their raw standards from the three-term error
model written out here, not from anything in the application, so a pass means
the correction really undoes an independent error box.
"""

from datetime import datetime, timedelta, timezone

import pytest
from PySide6 import QtCore

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.CalibrationHealth import (
    METADATA_PREFIX,
    AgeState,
    CalibrationMetadata,
    VerificationError,
    calibration_age,
    parse_created,
    split_metadata,
    verify_standard,
)
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import (
    CalibrationIdentity,
    snapshot_from_datapoints,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    build_archive,
    make_document,
    parse_session,
)

NOW = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
FREQUENCIES = tuple(range(10_000_000, 20_000_001, 1_000_000))
#: Directivity, source match and tracking of an imperfect test set.
E00, E11, TRACKING = 0.03 + 0.02j, 0.05 - 0.01j, 0.92 + 0.11j


def measured(gamma: complex) -> complex:
    """What the error box reports for a true reflection coefficient."""
    return E00 + TRACKING * gamma / (1 - E11 * gamma)


def raw_trace(gamma: complex, frequencies=FREQUENCIES) -> list[Datapoint]:
    value = measured(gamma)
    return [Datapoint(f, value.real, value.imag) for f in frequencies]


def calibration() -> Calibration:
    cal = Calibration()
    for name, gamma in (("short", -1), ("open", 1), ("load", 0)):
        cal.insert(name, raw_trace(gamma))
    cal.calc_corrections()
    return cal


class TestMetadataFile:
    @staticmethod
    def test_measuring_a_standard_stamps_the_time():
        cal = calibration()
        assert parse_created(cal.metadata.created) is not None
        assert cal.measured_here

    @staticmethod
    def test_it_round_trips_through_a_file_without_joining_the_notes(tmp_path):
        cal = calibration()
        cal.notes = ["my note"]
        cal.metadata = CalibrationMetadata(
            created="2026-11-01T10:00:00Z",
            device="NanoVNA-H",
            kit="SMA",
            connector="SMA male",
            operator="Sam",
        )
        path = tmp_path / "a.cal"
        cal.save(str(path))
        assert METADATA_PREFIX in path.read_text()

        loaded = Calibration()
        loaded.load(str(path))
        assert loaded.metadata == cal.metadata
        assert loaded.notes == ["my note"]
        assert not loaded.measured_here

    @staticmethod
    def test_an_older_file_has_an_unknown_age_not_a_guess(tmp_path):
        cal = calibration()
        cal.metadata = CalibrationMetadata()
        path = tmp_path / "old.cal"
        cal.save(str(path))
        assert METADATA_PREFIX not in path.read_text()

        loaded = Calibration()
        loaded.load(str(path))
        assert loaded.metadata == CalibrationMetadata()
        assert (
            calibration_age(loaded.metadata.created).state is AgeState.UNKNOWN
        )

    @staticmethod
    def test_metadata_does_not_change_the_calibration_identity():
        from NanoVNASaver.Measurement import compute_calibration_identity

        cal = calibration()
        before = compute_calibration_identity(cal)
        cal.metadata = CalibrationMetadata(operator="Sam")
        assert compute_calibration_identity(cal) == before

    @staticmethod
    @pytest.mark.parametrize(
        "line",
        [METADATA_PREFIX + "not json", METADATA_PREFIX + "[1, 2]"],
    )
    def test_an_unreadable_line_means_unknown(line):
        metadata, notes = split_metadata([line, "kept"])
        assert metadata == CalibrationMetadata()
        assert notes == ["kept"]

    @staticmethod
    def test_unknown_keys_and_wrong_types_are_ignored():
        line = METADATA_PREFIX + '{"created": 5, "device": "A", "later": "x"}'
        assert CalibrationMetadata.from_note(line) == CalibrationMetadata(
            device="A"
        )

    @staticmethod
    def test_a_session_keeps_it(tmp_path):
        cal = calibration()
        cal.metadata = CalibrationMetadata(created="2026-11-01T10:00:00Z")
        snapshot = snapshot_from_datapoints(raw_trace(0.1), None)
        document = make_document(
            [snapshot], calibration_files={"loaded.cal": cal.to_text()}
        )
        parsed = parse_session(build_archive(document))
        restored = Calibration()
        restored.from_text(parsed.calibration_files["loaded.cal"])
        assert restored.metadata.created == "2026-11-01T10:00:00Z"


class TestAge:
    @staticmethod
    def test_no_reminder_by_default_however_old():
        created = (NOW - timedelta(days=400)).isoformat()
        age = calibration_age(created, now=NOW)
        assert age.state is AgeState.CURRENT
        assert not age.warns

    @staticmethod
    def test_the_reminder_fires_at_the_chosen_interval():
        created = (NOW - timedelta(hours=3)).isoformat()
        assert (
            calibration_age(created, reminder_hours=3.0, now=NOW).state
            is AgeState.REMINDER
        )
        assert (
            calibration_age(created, reminder_hours=3.5, now=NOW).state
            is AgeState.CURRENT
        )

    @staticmethod
    def test_timezones_are_honoured():
        # 14:00 at +02:00 is 12:00 UTC, so no time has passed.
        age = calibration_age("2026-11-02T14:00:00+02:00", now=NOW)
        assert age.seconds == 0

    @staticmethod
    def test_a_utc_suffix_is_understood():
        age = calibration_age("2026-11-02T11:00:00Z", now=NOW)
        assert age.seconds == 3600

    @staticmethod
    @pytest.mark.parametrize(
        "created", ["", "yesterday", "2026-11-02T11:00:00"]
    )
    def test_missing_unparsable_and_naive_times_are_unknown(created):
        age = calibration_age(created, now=NOW)
        assert age.state is AgeState.UNKNOWN
        assert age.seconds is None

    @staticmethod
    def test_a_future_stamp_blames_the_clock_rather_than_a_negative_age():
        age = calibration_age("2026-11-02T13:00:00Z", now=NOW)
        assert age.state is AgeState.FUTURE
        assert age.warns
        assert "clock" in age.describe()


class TestVerification:
    @staticmethod
    @pytest.mark.parametrize(
        "standard, gamma", [("short", -1), ("open", 1), ("load", 0)]
    )
    def test_the_right_standard_passes_through_an_independent_box(
        standard, gamma
    ):
        result = verify_standard(calibration(), standard, raw_trace(gamma))
        assert result.passed
        assert result.worst_residual < 1e-9
        assert result.points == len(FREQUENCIES)

    @staticmethod
    def test_the_wrong_standard_fails_with_the_true_residual():
        result = verify_standard(calibration(), "short", raw_trace(1))
        assert not result.passed
        # An open read against a short's model: |(+1) - (-1)| = 2.
        assert result.worst_residual == pytest.approx(2.0)
        assert "OUTSIDE" in result.describe()

    @staticmethod
    def test_the_residual_is_the_known_offset_and_the_tolerance_decides():
        trace = raw_trace(0.04)
        assert verify_standard(
            calibration(), "load", trace, tolerance=0.05
        ).passed
        result = verify_standard(calibration(), "load", trace, tolerance=0.03)
        assert not result.passed
        assert result.worst_residual == pytest.approx(0.04)

    @staticmethod
    def test_the_worst_point_is_located():
        trace = raw_trace(0)
        value = measured(0.2)
        trace[4] = Datapoint(trace[4].freq, value.real, value.imag)
        result = verify_standard(calibration(), "load", trace)
        assert result.worst_frequency == trace[4].freq
        assert not result.passed

    @staticmethod
    def test_points_outside_the_span_are_counted_not_judged():
        beyond = (*FREQUENCIES, 30_000_000)
        trace = raw_trace(0, beyond)
        worst = measured(1)
        trace[-1] = Datapoint(30_000_000, worst.real, worst.imag)
        result = verify_standard(calibration(), "load", trace)
        assert result.passed
        assert result.skipped_points == 1
        assert "outside the calibrated span" in result.describe()

    @staticmethod
    def test_no_overlap_is_refused():
        with pytest.raises(VerificationError, match="overlap"):
            verify_standard(
                calibration(), "load", raw_trace(0, (50_000_000, 60_000_000))
            )

    @staticmethod
    def test_refusals_report_nothing_rather_than_a_pass():
        with pytest.raises(VerificationError, match="apply a calibration"):
            verify_standard(Calibration(), "load", raw_trace(0))
        with pytest.raises(VerificationError, match="no S11"):
            verify_standard(calibration(), "load", [])
        with pytest.raises(VerificationError, match="cannot be verified"):
            verify_standard(calibration(), "through", raw_trace(0))
        with pytest.raises(VerificationError, match="tolerance"):
            verify_standard(calibration(), "load", raw_trace(0), tolerance=0)
        with pytest.raises(VerificationError, match="finite"):
            verify_standard(
                calibration(),
                "load",
                [Datapoint(FREQUENCIES[0], float("nan"), 0.0)],
            )

    @staticmethod
    def test_a_simulated_result_says_so():
        result = verify_standard(
            calibration(), "load", raw_trace(0), simulated=True
        )
        assert "not hardware evidence" in result.describe()

    @staticmethod
    def test_a_new_measurement_clears_old_results():
        cal = calibration()
        cal.verifications["load"] = verify_standard(cal, "load", raw_trace(0))
        cal.insert("load", raw_trace(0))
        assert cal.verifications == {}


class TestRunProvenance:
    @staticmethod
    def test_a_snapshot_keeps_its_calibrations_time():
        cal = calibration()
        cal.metadata = CalibrationMetadata(created="2026-11-01T10:00:00Z")
        snapshot = snapshot_from_datapoints(
            raw_trace(0.1), None, calibration=cal
        )
        assert snapshot.provenance.calibration.created == cal.metadata.created
        cal.metadata = CalibrationMetadata(created="2026-11-02T10:00:00Z")
        assert snapshot.provenance.calibration.created == "2026-11-01T10:00:00Z"

    @staticmethod
    def test_it_survives_serialisation_and_old_records_stay_unknown():
        identity = CalibrationIdentity(created="2026-11-01T10:00:00Z")
        assert (
            CalibrationIdentity.from_dict(identity.to_dict()).created
            == identity.created
        )
        assert CalibrationIdentity.from_dict({}).created == ""

    @staticmethod
    def test_a_report_keeps_age_apart_from_coverage():
        from NanoVNASaver.Reports import build_report, render_html
        from NanoVNASaver.Traces import Trace

        cal = calibration()
        created = datetime.now(timezone.utc) - timedelta(hours=5)
        cal.metadata = CalibrationMetadata(created=created.isoformat())
        snapshot = snapshot_from_datapoints(
            raw_trace(0.1), None, calibration=cal
        )
        trace = Trace(id="t", name="t", snapshot=snapshot)
        text = render_html(build_report("r", [trace]))
        assert "Calibration when swept: Age 5.0 h" in text

    @staticmethod
    def test_a_report_on_an_older_calibration_says_unknown():
        from NanoVNASaver.Reports import build_report, render_html
        from NanoVNASaver.Traces import Trace

        cal = calibration()
        cal.metadata = CalibrationMetadata()
        snapshot = snapshot_from_datapoints(
            raw_trace(0.1), None, calibration=cal
        )
        trace = Trace(id="t", name="t", snapshot=snapshot)
        text = render_html(build_report("r", [trace]))
        assert "Age unknown" in text


class _StandardDevice:
    """Makes a simulator report the error box's view of one standard."""

    def __init__(self, gamma):
        self.vna = make_simulated_vna("antenna", seed=1, noise_floor=0.0)
        value = measured(gamma)
        self.vna.readValues = lambda name: (
            [value] * self.vna.datapoints if name == "data 0" else []
        )


@pytest.fixture
def window(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    app = NanoVNASaver()
    app.session.dirty = False
    try:
        yield app
    finally:
        app.session.dirty = False
        app.worker.quit()
        app.worker.wait(5000)
        app.deleteLater()


def _wait_for_verification(window) -> None:
    timer = QtCore.QElapsedTimer()
    timer.start()
    panel = window.windows["calibration"]
    while panel._verification_pending is not None and timer.elapsed() < 15000:
        QtCore.QCoreApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50
        )
    assert panel._verification_pending is None


def _prepare(window, gamma):
    window.vna = _StandardDevice(gamma).vna
    window.sweep.update(10_000_000, 20_000_000, 1, 11)
    window.calibration = calibration()
    return window.windows["calibration"]


class TestWindow:
    @staticmethod
    def _select(panel, standard):
        panel.verify_standard_input.setCurrentIndex(
            panel.verify_standard_input.findData(standard)
        )

    def test_a_real_sweep_verifies_the_connected_standard(self, window):
        panel = _prepare(window, -1)
        self._select(panel, "short")
        panel.verifyConnectedStandard()
        _wait_for_verification(window)

        result = window.calibration.verifications["short"]
        assert result.passed
        assert result.simulated
        text = panel.health_verification_label.text()
        assert "Short" in text and "not hardware evidence" in text

    def test_the_wrong_standard_connected_is_reported(self, window):
        panel = _prepare(window, 1)
        self._select(panel, "short")
        panel.verifyConnectedStandard()
        _wait_for_verification(window)
        assert not window.calibration.verifications["short"].passed
        assert "OUTSIDE" in panel.health_verification_label.text()

    def test_an_unrelated_run_finishing_verifies_nothing(self, window):
        panel = _prepare(window, -1)
        panel._verification_pending = ("short", object(), 0.05)
        window.worker.signals.run_request_finished.connect(panel._verified)
        panel._verified(None, object())
        assert panel._verification_pending is not None
        assert window.calibration.verifications == {}

    def test_it_explains_what_it_needs(self, window):
        panel = window.windows["calibration"]
        panel.verifyConnectedStandard()
        assert "Apply a calibration" in panel.health_verification_label.text()
        window.calibration = calibration()
        panel.verifyConnectedStandard()
        assert "Connect a device" in panel.health_verification_label.text()

    def test_the_reminder_setting_reaches_the_sweep_panel(self, window):
        panel = _prepare(window, -1)
        # The panel only states a calibration status for data on screen.
        window.data.s11 = raw_trace(0.1)
        created = datetime.now(timezone.utc) - timedelta(hours=5)
        window.calibration.metadata = CalibrationMetadata(
            created=created.isoformat()
        )
        panel.reminder_input.setValue(0.0)
        assert not window.calibration_age_now().warns
        panel.reminder_input.setValue(2.0)
        assert window.calibration_age_now().state is AgeState.REMINDER
        assert "2 h reminder" in panel.health_age_label.text()
        assert "2 h reminder" in window.sweep_control.label_calibration.text()

    def test_coverage_and_age_stay_separate_statements(self, window):
        window.sweep_control.show_calibration_status(
            None, "application", calibration_age("", now=NOW)
        )
        assert "Age" not in window.sweep_control.label_calibration.text()

    def test_a_loaded_file_does_not_inherit_the_connected_device(
        self, window, tmp_path
    ):
        panel = _prepare(window, -1)
        cal = window.calibration
        cal.metadata = CalibrationMetadata(created="2026-11-01T10:00:00Z")
        path = tmp_path / "file.cal"
        cal.save(str(path))
        loaded = Calibration()
        loaded.load(str(path))
        window.calibration = loaded
        panel.recordMetadata()
        assert window.calibration.metadata.device == ""

    def test_a_calibration_measured_here_records_the_device(self, window):
        panel = _prepare(window, -1)
        panel.recordMetadata()
        assert window.calibration.metadata.device == window.vna.name


class TestResultsBelongToOneCalculation:
    @staticmethod
    def test_recalculating_clears_earlier_results():
        cal = calibration()
        cal.verifications["load"] = verify_standard(cal, "load", raw_trace(0))
        cal.calc_corrections()
        assert cal.verifications == {}

    def test_a_calibration_replaced_during_the_sweep_earns_nothing(
        self, window
    ):
        panel = _prepare(window, -1)
        panel.verify_standard_input.setCurrentIndex(0)
        panel.verifyConnectedStandard()
        replacement = calibration()
        window.calibration = replacement
        _wait_for_verification(window)
        assert replacement.verifications == {}
        assert "changed during" in panel.health_verification_label.text()

    def test_recalculating_during_the_sweep_earns_nothing(self, window):
        panel = _prepare(window, -1)
        panel.verifyConnectedStandard()
        window.calibration.calc_corrections()
        _wait_for_verification(window)
        assert window.calibration.verifications == {}
