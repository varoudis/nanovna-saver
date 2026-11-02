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
"""RF-04: combining a forward and a reverse sweep into one S-matrix.

The expectations here are the *physical* statement being tested: a DUT
measured from the other side has its ports exchanged, so the reverse
sweep's S11 is the DUT's S22.  No transform is applied, so every assertion
is an identity on the samples — which is exactly what makes a test able to
catch a re-labelling error.
"""

import os

import pytest

from NanoVNASaver.Calibration import CalData, Calibration
from NanoVNASaver.Measurement import (
    S11,
    S12,
    S21,
    S22,
    CompletionState,
    snapshot_from_datapoints,
)
from NanoVNASaver.Reversal import (
    CaptureStage,
    ReversalCompatibility,
    ReversalError,
    check_compatibility,
    combine_reversal,
    describe_provenance,
    own_channels,
    reversed_channels,
)
from NanoVNASaver.RFTools import Datapoint

FREQUENCIES = [14_000_000, 14_100_000, 14_200_000]

_APP = None


@pytest.fixture(scope="session")
def qt_app():
    global _APP  # noqa: PLW0603 - one application per interpreter
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _points(values):
    return [
        Datapoint(freq, value.real, value.imag)
        for freq, value in zip(FREQUENCIES, values, strict=True)
    ]


def _snapshot(s11, s21, *, name="sweep", **kwargs):
    return snapshot_from_datapoints(
        _points(s11), _points(s21), name=name, **kwargs
    )


#: An asymmetric, non-reciprocal DUT: S11 != S22 and S21 != S12.
FORWARD_S11 = [0.10 + 0.05j, 0.11 + 0.06j, 0.12 + 0.07j]
FORWARD_S21 = [0.80 - 0.10j, 0.79 - 0.11j, 0.78 - 0.12j]
REVERSE_S11 = [0.30 - 0.20j, 0.31 - 0.21j, 0.32 - 0.22j]
REVERSE_S21 = [0.60 + 0.40j, 0.59 + 0.41j, 0.58 + 0.42j]


class TestRelabelling:
    @staticmethod
    def test_the_reverse_s11_becomes_s22_and_its_s21_becomes_s12():
        """Ports are exchanged by turning the DUT around; nothing rotates."""
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        channels = reversed_channels(reverse)
        assert set(channels) == {S22, S12}
        assert [p.z for p in channels[S22]] == pytest.approx(REVERSE_S11)
        assert [p.z for p in channels[S12]] == pytest.approx(REVERSE_S21)

    @staticmethod
    def test_relabelling_keeps_every_sample_identical():
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        channels = reversed_channels(reverse)
        assert list(channels[S22]) == list(reverse.corrected[S11])
        assert list(channels[S12]) == list(reverse.corrected[S21])


class TestCompatibility:
    @staticmethod
    def test_two_orientations_of_one_dut_are_compatible():
        forward = _snapshot(FORWARD_S11, FORWARD_S21)
        reverse = _snapshot(REVERSE_S11, REVERSE_S21)
        assert (
            check_compatibility(forward, reverse)
            is ReversalCompatibility.SAME_DUT
        )

    @staticmethod
    def test_a_different_grid_is_refused_with_both_spans_named():
        forward = _snapshot(FORWARD_S11, FORWARD_S21)
        other = snapshot_from_datapoints(
            [Datapoint(20_000_000 + i * 100_000, 0.1, 0.1) for i in range(3)],
            [],
        )
        with pytest.raises(ReversalError, match="different frequency grids"):
            check_compatibility(forward, other)

    @staticmethod
    def test_a_different_reference_impedance_is_refused():
        forward = _snapshot(FORWARD_S11, FORWARD_S21)
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, reference_impedance=75.0)
        assert (
            check_compatibility(forward, reverse)
            is ReversalCompatibility.DIFFERENT_REFERENCE
        )
        with pytest.raises(ReversalError, match="50 ohm and.*75 ohm"):
            combine_reversal(forward, reverse)

    @staticmethod
    def test_a_different_calibration_is_refused():
        forward = _snapshot(
            FORWARD_S11,
            FORWARD_S21,
            calibration=_calibration_with(0.1),
        )
        reverse = _snapshot(
            REVERSE_S11,
            REVERSE_S21,
            calibration=_calibration_with(0.2),
        )
        assert (
            check_compatibility(forward, reverse)
            is ReversalCompatibility.DIFFERENT_CALIBRATION
        )
        with pytest.raises(ReversalError, match="different calibrations"):
            combine_reversal(forward, reverse)

    @staticmethod
    def test_a_different_source_is_refused():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, source="device")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, source="file")
        assert (
            check_compatibility(forward, reverse)
            is ReversalCompatibility.DIFFERENT_SOURCE
        )

    @staticmethod
    def test_mixing_simulated_and_measured_data_is_refused():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, simulated=True)
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, simulated=False)
        assert (
            check_compatibility(forward, reverse)
            is ReversalCompatibility.MIXED_SIMULATION
        )


def _calibration_with(value: float) -> Calibration:
    """A calibration whose standards differ by ``value``."""
    calibration = Calibration()
    for freq in FREQUENCIES:
        calibration.dataset[freq] = CalData(
            short=value + 0j, open=value + 0.5 + 0j, load=0.01j
        )
    return calibration


class TestCombination:
    @staticmethod
    def test_all_four_channels_come_from_the_right_sweep():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        combined = combine_reversal(forward, reverse)

        assert [p.z for p in combined.corrected[S11]] == pytest.approx(
            FORWARD_S11
        )
        assert [p.z for p in combined.corrected[S21]] == pytest.approx(
            FORWARD_S21
        )
        assert [p.z for p in combined.corrected[S22]] == pytest.approx(
            REVERSE_S11
        )
        assert [p.z for p in combined.corrected[S12]] == pytest.approx(
            REVERSE_S21
        )

    @staticmethod
    def test_the_asymmetry_survives_the_combination():
        """If the two sweeps were swapped the matrix would look different."""
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        combined = combine_reversal(forward, reverse)
        assert combined.corrected[S11] != combined.corrected[S22]
        assert combined.corrected[S21] != combined.corrected[S12]

    @staticmethod
    def test_the_combination_names_both_sources():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        combined = combine_reversal(forward, reverse)
        assert combined.provenance.derived_from == (
            f"{forward.id}+{reverse.id}"
        )
        lines = describe_provenance(combined)
        assert lines == [
            f"S11/S21 from forward sweep {forward.id}",
            f"S22/S12 from reverse sweep {reverse.id}",
        ]

    @staticmethod
    def test_the_combination_has_no_raw_samples():
        """A combined matrix is computed; it is not a measurement."""
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        combined = combine_reversal(forward, reverse)
        assert combined.raw == {}

    @staticmethod
    def test_both_source_snapshots_are_untouched():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        before_forward = list(forward.corrected[S11])
        before_reverse = list(reverse.corrected[S11])
        combine_reversal(forward, reverse)
        assert list(forward.corrected[S11]) == before_forward
        assert list(reverse.corrected[S11]) == before_reverse

    @staticmethod
    def test_a_partial_sweep_is_refused():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(
            REVERSE_S11,
            REVERSE_S21,
            name="reverse",
            completion=CompletionState.PARTIAL,
        )
        with pytest.raises(ReversalError, match="partial sweep"):
            combine_reversal(forward, reverse)

    @staticmethod
    def test_a_missing_reverse_channel_is_named_not_invented():
        """A reverse sweep with only S11 leaves S12 genuinely absent."""
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = snapshot_from_datapoints(
            _points(REVERSE_S11), [], name="reverse"
        )
        combined = combine_reversal(forward, reverse)
        assert S12 not in combined.corrected
        assert S22 in combined.corrected
        assert "S12" in combined.provenance.notes
        assert own_channels(combined) == (S11, S21, S22)


class TestHonesty:
    @staticmethod
    def test_a_forward_only_snapshot_reports_only_what_it_measured():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        assert own_channels(forward) == (S11, S21)

    @staticmethod
    def test_an_absent_channel_is_never_a_zero_sample():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = snapshot_from_datapoints(
            _points(REVERSE_S11), [], name="reverse"
        )
        combined = combine_reversal(forward, reverse)
        # Absent, not S12 = 0.
        assert S12 not in combined.corrected
        assert all(name != S12 for name in own_channels(combined))

    @staticmethod
    def test_describe_provenance_is_empty_for_a_plain_measurement():
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        assert describe_provenance(forward) == []


class TestCaptureWorkflow:
    @staticmethod
    def _capture():
        from NanoVNASaver.Reversal import ReversalCapture

        return ReversalCapture()

    @staticmethod
    def test_the_forward_sweep_alone_does_not_combine():
        capture = TestCaptureWorkflow._capture()
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        assert capture.accept(forward, direction="forward") is None
        assert capture.stage is CaptureStage.AWAITING_REVERSE
        assert capture.forward is forward
        assert capture.reverse is None

    @staticmethod
    def test_the_second_sweep_completes_the_capture():
        capture = TestCaptureWorkflow._capture()
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        capture.accept(forward, direction="forward")
        combined = capture.accept(reverse, direction="reverse")
        assert capture.complete
        assert combined is capture.combined
        assert combined.corrected[S22] == capture.combined.corrected[S22]
        assert "complete" in capture.describe()

    @staticmethod
    def test_a_reverse_sweep_before_the_forward_is_refused():
        """The DUT has not been turned around yet; nothing to combine."""
        capture = TestCaptureWorkflow._capture()
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        with pytest.raises(ReversalError, match="before the forward"):
            capture.accept(reverse, direction="reverse")
        assert capture.stage is CaptureStage.AWAITING_FORWARD

    @staticmethod
    def test_a_second_forward_sweep_is_refused():
        capture = TestCaptureWorkflow._capture()
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        capture.accept(forward, direction="forward")
        with pytest.raises(ReversalError, match="already been captured"):
            capture.accept(forward, direction="forward")

    @staticmethod
    def test_no_sweep_is_accepted_after_completion():
        capture = TestCaptureWorkflow._capture()
        capture.accept(
            _snapshot(FORWARD_S11, FORWARD_S21, name="f"), direction="forward"
        )
        capture.accept(
            _snapshot(REVERSE_S11, REVERSE_S21, name="r"), direction="reverse"
        )
        with pytest.raises(ReversalError, match="start a new one"):
            capture.accept(
                _snapshot(REVERSE_S11, REVERSE_S21, name="r2"),
                direction="reverse",
            )

    @staticmethod
    def test_an_incompatible_pair_fails_with_a_reason_and_keeps_both():
        capture = TestCaptureWorkflow._capture()
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        mismatched = _snapshot(
            REVERSE_S11, REVERSE_S21, name="reverse", reference_impedance=75
        )
        capture.accept(forward, direction="forward")
        with pytest.raises(ReversalError):
            capture.accept(mismatched, direction="reverse")
        assert capture.stage is CaptureStage.FAILED
        assert "75 ohm" in capture.reason
        # Both are kept: the operator needs the original sweeps to diagnose.
        assert capture.forward is forward
        assert capture.reverse is mismatched
        assert capture.combined is None
        assert capture.describe().startswith("failed:")

    @staticmethod
    def test_cancelling_keeps_what_was_captured():
        capture = TestCaptureWorkflow._capture()
        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        capture.accept(forward, direction="forward")
        capture.cancel()
        assert capture.stage is CaptureStage.FAILED
        assert capture.reason == "cancelled"
        assert capture.forward is forward

    @staticmethod
    def test_an_unknown_direction_is_an_error_not_a_default():
        capture = TestCaptureWorkflow._capture()
        with pytest.raises(ReversalError, match="unknown direction"):
            capture.accept(
                _snapshot(FORWARD_S11, FORWARD_S21), direction="sideways"
            )


class TestWindowWorkflow:
    """RF-04 through the application's real sweep-completion hook."""

    @staticmethod
    def _window(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        app = NanoVNASaver()
        app.session.dirty = False
        return app

    @staticmethod
    def _live(app, s11, s21):
        app._live_snapshot = _snapshot(s11, s21)

    def test_two_sweeps_produce_a_matrix_and_guide_the_operator(self, qt_app):
        app = self._window(qt_app)
        try:
            app.start_reversal_capture()
            assert app.reversal_direction == "forward"
            assert "Forward" in app.reversal_label.text()

            self._live(app, FORWARD_S11, FORWARD_S21)
            assert app._capture_reversal_step() is True
            assert app.reversal_direction == "reverse"
            assert "Reverse" in app.reversal_label.text()
            assert app.s_matrix is None

            self._live(app, REVERSE_S11, REVERSE_S21)
            assert app._capture_reversal_step() is True
            assert app.reversal_direction == ""
            assert app.s_matrix is not None
            assert [p.z for p in app.s_matrix.corrected[S22]] == pytest.approx(
                REVERSE_S11
            )
            assert "complete" in app.reversal_label.text()
        finally:
            app.session.dirty = False
            app.worker.quit()
            app.worker.wait(5000)
            app.deleteLater()

    def test_an_incompatible_second_sweep_is_reported_not_stored(self, qt_app):
        app = self._window(qt_app)
        try:
            app.start_reversal_capture()
            self._live(app, FORWARD_S11, FORWARD_S21)
            app._capture_reversal_step()
            # A reverse sweep at another reference cannot be combined.
            app._live_snapshot = _snapshot(
                REVERSE_S11, REVERSE_S21, reference_impedance=75
            )
            assert app._capture_reversal_step() is True
            assert app.s_matrix is None
            assert "Reversal failed" in app.reversal_label.text()
            assert app.reversal_direction == ""
        finally:
            app.session.dirty = False
            app.worker.quit()
            app.worker.wait(5000)
            app.deleteLater()

    def test_a_normal_sweep_is_not_consumed_by_an_idle_capture(self, qt_app):
        app = self._window(qt_app)
        try:
            self._live(app, FORWARD_S11, FORWARD_S21)
            assert app._capture_reversal_step() is False
            assert app.s_matrix is None
        finally:
            app.session.dirty = False
            app.worker.quit()
            app.worker.wait(5000)
            app.deleteLater()

    def test_cancelling_between_the_sweeps_stops_the_capture(self, qt_app):
        app = self._window(qt_app)
        try:
            app.start_reversal_capture()
            self._live(app, FORWARD_S11, FORWARD_S21)
            app._capture_reversal_step()
            app.cancel_reversal_capture()
            self._live(app, REVERSE_S11, REVERSE_S21)
            assert app._capture_reversal_step() is False
            assert app.s_matrix is None
        finally:
            app.session.dirty = False
            app.worker.quit()
            app.worker.wait(5000)
            app.deleteLater()


class TestExport:
    @staticmethod
    def test_the_combined_matrix_exports_four_columns(tmp_path):
        """A combined matrix must be exportable as a real .s2p."""
        from NanoVNASaver.Touchstone import Touchstone

        forward = _snapshot(FORWARD_S11, FORWARD_S21, name="forward")
        reverse = _snapshot(REVERSE_S11, REVERSE_S21, name="reverse")
        combined = combine_reversal(forward, reverse, name="matrix")

        touchstone = Touchstone("")
        for name in (S11, S21, S12, S22):
            setattr(touchstone, name.lower(), list(combined.corrected[name]))
        touchstone.filename = str(tmp_path / "matrix.s2p")
        touchstone.save(4)

        text = (tmp_path / "matrix.s2p").read_text(encoding="utf-8")
        rows = [
            line
            for line in text.splitlines()
            if line.strip() and not line.startswith(("!", "#"))
        ]
        assert len(rows) == len(FREQUENCIES)
        # S11, S21, S12, S22 in that order, two reals each.
        first = rows[0].split()
        assert len(first) == 9
        assert complex(float(first[1]), float(first[2])) == pytest.approx(
            FORWARD_S11[0]
        )
        assert complex(float(first[3]), float(first[4])) == pytest.approx(
            FORWARD_S21[0]
        )
        assert complex(float(first[5]), float(first[6])) == pytest.approx(
            REVERSE_S21[0]
        )
        assert complex(float(first[7]), float(first[8])) == pytest.approx(
            REVERSE_S11[0]
        )
