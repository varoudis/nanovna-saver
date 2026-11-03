#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
"""FIX-02: measured versus synthesized Touchstone parameters."""

import math
import os

import pytest

from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Touchstone import (
    EXPORT_NOTE_PREFIX,
    ChannelLengthMismatchError,
    FrequencyMismatchError,
    MissingParameterError,
    NonAscendingFrequencyError,
    NonFiniteValueError,
    Touchstone,
    TouchstoneExportError,
    UnsupportedParameterCountError,
)

VALID_S2P = "./tests/data/valid.s2p"
VALID_S1P = "./tests/data/valid.s1p"


def _data_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line and not line.startswith("!") and not line.startswith("#")
    ]


def _comment_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("!")]


def _option_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("#"))


def _s11_s21(points: int = 3) -> Touchstone:
    """A forward-only dataset: S11 and S21 measured, S12/S22 absent."""
    touchstone = Touchstone()
    touchstone.s11 = [
        Datapoint(1e6 * (i + 1), 0.1 * (i + 1), 0.0) for i in range(points)
    ]
    touchstone.s21 = [
        Datapoint(1e6 * (i + 1), 0.0, -0.5) for i in range(points)
    ]
    return touchstone


def _reload(text: str, tmp_path, suffix: str = ".s2p") -> Touchstone:
    path = tmp_path / f"reload{suffix}"
    path.write_text(text, encoding="utf-8")
    touchstone = Touchstone(str(path))
    touchstone.load()
    return touchstone


class TestS1PRoundTrip:
    def test_s1p_round_trip(self, tmp_path):
        original = Touchstone(VALID_S1P)
        original.load()
        text = original.saves(1)

        # Reflection-only export stays available, states what it holds, and
        # never claims a two-port characterization.
        assert _option_line(text) == "# HZ S RI R 50"
        notes = [
            ln
            for ln in _comment_lines(text)
            if ln.startswith(EXPORT_NOTE_PREFIX)
        ]
        assert any("1-port" in ln for ln in notes)
        assert any("reflection-only" in ln for ln in notes)
        assert not any("S21" in line for line in _data_lines(text))
        assert not any("NOT measured" in ln for ln in notes)

        reloaded = _reload(text, tmp_path, ".s1p")
        assert len(reloaded.s11) == len(original.s11)
        assert not reloaded.s21
        for first, second in zip(original.s11, reloaded.s11, strict=True):
            assert first == second

    def test_s1p_writes_only_s11_columns(self):
        touchstone = _s11_s21()
        for line in _data_lines(touchstone.saves(1)):
            assert len(line.split()) == 3


class TestForwardOnlyS2P:
    def test_placeholders_and_comments(self, tmp_path):
        touchstone = _s11_s21()
        assert touchstone.s11 and touchstone.s21
        assert not touchstone.s12 and not touchstone.s22
        text = touchstone.saves(4)

        notes = [
            ln
            for ln in _comment_lines(text)
            if ln.startswith(EXPORT_NOTE_PREFIX)
        ]
        assert any(
            "measured (present in source data): S11, S21" in ln for ln in notes
        )
        assert any(
            "zero placeholders, NOT measured: S12, S22" in ln for ln in notes
        )
        assert any("incomplete two-port characterization" in ln for ln in notes)
        assert any("may ignore comment lines" in ln for ln in notes)

        rows = _data_lines(text)
        assert len(rows) == 3
        for row, source in zip(rows, touchstone.s11, strict=True):
            fields = row.split()
            assert len(fields) == 9
            assert fields[0] == str(source.freq)
            # Genuine S11 and S21 are written unchanged ...
            assert fields[1] == str(source.re)
            assert fields[2] == str(source.im)
            # ... and S12/S22 are explicit zeros.
            assert fields[5:9] == ["0.0", "0.0", "0.0", "0.0"]

    def test_forward_only_reload(self, tmp_path):
        touchstone = _s11_s21()
        reloaded = _reload(touchstone.saves(4), tmp_path)
        assert len(reloaded.s11) == len(touchstone.s11)
        assert len(reloaded.s21) == len(touchstone.s21)
        assert [dp.z for dp in reloaded.s11] == [dp.z for dp in touchstone.s11]
        assert [dp.z for dp in reloaded.s21] == [dp.z for dp in touchstone.s21]
        assert all(dp.z == 0 for dp in reloaded.s12)
        assert all(dp.z == 0 for dp in reloaded.s22)

    def test_placeholder_detection_matches_export(self):
        touchstone = _s11_s21()
        assert touchstone.placeholder_parameters(4) == ("S12", "S22")
        assert touchstone.placeholder_parameters(1) == ()
        assert touchstone.present_parameters() == ("S11", "S21")
        touchstone.s12 = list(touchstone.s11)
        assert touchstone.placeholder_parameters(4) == ("S22",)


class TestFullS2P:
    def test_genuine_channels_preserved(self, tmp_path):
        original = Touchstone(VALID_S2P)
        original.load()
        text = original.saves(4)

        notes = [
            ln
            for ln in _comment_lines(text)
            if ln.startswith(EXPORT_NOTE_PREFIX)
        ]
        assert any(
            "measured (present in source data): S11, S21, S12, S22" in ln
            for ln in notes
        )
        assert not any("NOT measured" in ln for ln in notes)
        assert not any("incomplete two-port" in ln for ln in notes)

        reloaded = _reload(text, tmp_path)
        for parameter in ("11", "21", "12", "22"):
            assert len(reloaded.s(parameter)) == len(original.s(parameter))
            for first, second in zip(
                original.s(parameter), reloaded.s(parameter), strict=True
            ):
                assert first.freq == second.freq
                assert first.re == pytest.approx(second.re)
                assert first.im == pytest.approx(second.im)

        # Spot-check the first row against the source file itself.
        assert reloaded.s11[0] == pytest.approx(
            Datapoint(500000, -3.33238e-001, 1.80018e-004)
        )
        assert reloaded.s12[0].re == pytest.approx(6.75290e-001)
        assert reloaded.s12[0].im == pytest.approx(-8.20129e-007)
        assert reloaded.s22[0].re == pytest.approx(-3.33238e-001)
        assert reloaded.s22[0].im == pytest.approx(3.08078e-004)

    def test_comments_serialized_and_deduplicated(self, tmp_path):
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(1, 0.1, 0.0)]
        touchstone.s21 = [Datapoint(1, 0.2, 0.0)]
        touchstone.comments = ["! measured on a test bench", "second note"]

        first = touchstone.saves(1)
        assert "! measured on a test bench" in _comment_lines(first)
        assert "! second note" in _comment_lines(first)

        # Re-exporting the same object must not stack the generated notes.
        reloaded = _reload(first, tmp_path, ".s1p")
        assert reloaded.comments == [
            "! measured on a test bench",
            "! second note",
            *[
                ln
                for ln in _comment_lines(first)
                if ln.startswith(EXPORT_NOTE_PREFIX)
            ],
        ]
        assert reloaded.saves(1).count(EXPORT_NOTE_PREFIX) == first.count(
            EXPORT_NOTE_PREFIX
        )


class TestReferenceImpedance:
    def test_non_default_impedance_written_and_reloaded(self, tmp_path):
        touchstone = _s11_s21()
        touchstone.opts.resistance = 75
        text = touchstone.saves(4)
        assert _option_line(text) == "# HZ S RI R 75"

        reloaded = _reload(text, tmp_path)
        assert reloaded.opts.resistance == 75
        assert str(reloaded.opts) == "# HZ S RI R 75"

    def test_reference_impedance_comes_from_loaded_file(self, tmp_path):
        source = tmp_path / "seventyfive.s1p"
        source.write_text(
            "# HZ S RI R 75\n1000000 0.25 0.0\n", encoding="utf-8"
        )
        touchstone = Touchstone(str(source))
        touchstone.load()
        assert _option_line(touchstone.saves(1)) == "# HZ S RI R 75"

    def test_invalid_impedance_rejected(self):
        touchstone = _s11_s21()
        touchstone.opts.resistance = 0
        with pytest.raises(TouchstoneExportError, match="positive finite"):
            touchstone.saves(4)

    def test_fractional_impedance_round_trips(self, tmp_path):
        touchstone = _s11_s21()
        touchstone.opts.resistance = 75.5
        text = touchstone.saves(4)
        assert _option_line(text) == "# HZ S RI R 75.5"
        assert _reload(text, tmp_path).opts.resistance == 75.5


class TestExportValidation:
    def test_unsupported_parameter_count(self):
        touchstone = _s11_s21()
        for count in (0, 2, 3, 5, -1):
            with pytest.raises(UnsupportedParameterCountError):
                touchstone.saves(count)
        assert issubclass(UnsupportedParameterCountError, ValueError)

    def test_missing_s11(self):
        touchstone = Touchstone()
        touchstone.s21 = [Datapoint(1, 0.5, 0.0)]
        with pytest.raises(MissingParameterError, match="S11"):
            touchstone.saves(1)
        with pytest.raises(MissingParameterError, match="S11"):
            touchstone.saves(4)

    def test_missing_s21(self):
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(1, 0.5, 0.0)]
        with pytest.raises(MissingParameterError, match="S21"):
            touchstone.saves(4)
        # The one-port export does not need S21.
        assert touchstone.saves(1)

    def test_mismatched_frequency_grids(self):
        touchstone = _s11_s21()
        touchstone.s21[1] = Datapoint(12345, 0.0, -0.5)
        with pytest.raises(FrequencyMismatchError, match="not correlated"):
            touchstone.saves(4)

    def test_frequency_mismatch_stays_a_lookup_error(self):
        """Backwards compatibility for callers of the old exception."""
        touchstone = _s11_s21()
        touchstone.s21[0] = Datapoint(99, 0.0, -0.5)
        with pytest.raises(
            LookupError, match="Frequencies of sdata not correlated"
        ):
            touchstone.saves(4)

    def test_mismatched_optional_channel_grid(self):
        touchstone = _s11_s21()
        touchstone.s12 = [Datapoint(1e6 * (i + 1), 0.0, 0.0) for i in range(3)]
        touchstone.s22 = [Datapoint(1e6 * (i + 1), 0.0, 0.0) for i in range(3)]
        touchstone.s22[2] = Datapoint(7, 0.0, 0.0)
        with pytest.raises(FrequencyMismatchError):
            touchstone.saves(4)

    def test_channel_length_mismatch(self):
        touchstone = _s11_s21()
        touchstone.s21 = touchstone.s21[:2]
        with pytest.raises(ChannelLengthMismatchError, match="samples"):
            touchstone.saves(4)

    @pytest.mark.parametrize(
        "value",
        [math.nan, math.inf, -math.inf],
        ids=["nan", "inf", "-inf"],
    )
    def test_non_finite_s11_value(self, value):
        touchstone = _s11_s21()
        touchstone.s11[1] = Datapoint(touchstone.s11[1].freq, value, 0.0)
        with pytest.raises(NonFiniteValueError, match="not finite"):
            touchstone.saves(4)

    def test_non_finite_optional_value(self):
        touchstone = _s11_s21()
        touchstone.s12 = list(touchstone.s11)
        touchstone.s22 = list(touchstone.s11)
        touchstone.s22[0] = Datapoint(touchstone.s22[0].freq, 0.0, math.nan)
        with pytest.raises(NonFiniteValueError, match="S22"):
            touchstone.saves(4)

    def test_errors_share_a_catchable_base(self):
        for error in (
            UnsupportedParameterCountError,
            MissingParameterError,
            ChannelLengthMismatchError,
            FrequencyMismatchError,
            NonAscendingFrequencyError,
            NonFiniteValueError,
        ):
            assert issubclass(error, TouchstoneExportError)


class TestValidationBeforeWriting:
    def test_rejected_export_leaves_file_untouched(self, tmp_path):
        target = tmp_path / "output.s2p"
        target.write_text("sentinel\n", encoding="utf-8")

        touchstone = _s11_s21()
        touchstone.filename = str(target)
        touchstone.s21 = touchstone.s21[:1]
        with pytest.raises(ChannelLengthMismatchError):
            touchstone.save(4)
        assert target.read_text(encoding="utf-8") == "sentinel\n"

    def test_rejected_export_creates_no_file(self, tmp_path):
        target = tmp_path / "never.s2p"
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(1, math.nan, 0.0)]
        touchstone.filename = str(target)
        with pytest.raises(NonFiniteValueError):
            touchstone.save(1)
        assert not os.path.exists(target)

    def test_successful_save_matches_saves(self, tmp_path):
        target = tmp_path / "written.s2p"
        touchstone = _s11_s21()
        touchstone.comments = ["! provenance"]
        touchstone.filename = str(target)
        touchstone.save(4)
        assert target.read_text(encoding="utf-8") == touchstone.saves(4)


class TestAscendingFrequency:
    """Bug 5: a file whose samples decrease is not Touchstone.

    Touchstone 1.1 expects the data lines in non-descending frequency
    order, so a descending pair produces a file a conforming reader may
    reject or silently reorder, and a NaN cannot be ordered at all.
    ``saves``/``save`` must refuse both like every other inconsistency,
    before any file is opened.  A *repeat* is deliberately allowed: a
    segmented logarithmic sweep is built with each segment ending on the
    next one's start (``Sweep.get_index_range``), so refusing it made every
    such sweep unexportable.
    """

    @staticmethod
    @pytest.mark.parametrize(
        "frequencies,expected",
        [
            ([3_000_000, 2_000_000, 1_000_000], "never descending"),
            ([1_000_000, 3_000_000, 2_000_000], "never descending"),
            ([1_000_000, 2_000_000, math.nan], "not a number"),
            ([math.nan, 2_000_000, 3_000_000], "not a number"),
        ],
        ids=["descending", "out-of-order", "nan-last", "nan-first"],
    )
    def test_a_decreasing_or_unordered_frequency_is_refused(
        frequencies, expected
    ):
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(freq, 0.1, 0.2) for freq in frequencies]
        with pytest.raises(NonAscendingFrequencyError, match=expected):
            touchstone.saves(1)

    @staticmethod
    def test_the_error_names_the_offending_index_and_frequency():
        touchstone = Touchstone()
        touchstone.s11 = [
            Datapoint(1_000_000, 0.1, 0.2),
            Datapoint(2_000_000, 0.1, 0.2),
            Datapoint(1_500_000, 0.1, 0.2),
        ]
        with pytest.raises(NonAscendingFrequencyError) as info:
            touchstone.saves(1)
        message = str(info.value)
        assert "index 2" in message
        assert "1500000" in message
        assert "2000000" in message

    @staticmethod
    def test_the_rejection_is_a_touchstone_export_error():
        assert issubclass(NonAscendingFrequencyError, TouchstoneExportError)
        assert issubclass(NonAscendingFrequencyError, ValueError)

    @staticmethod
    def test_a_two_port_export_is_checked_too():
        touchstone = Touchstone()
        touchstone.s11 = [
            Datapoint(3e6, 0.1, 0.2),
            Datapoint(2e6, 0.1, 0.2),
            Datapoint(1e6, 0.1, 0.2),
        ]
        touchstone.s21 = list(touchstone.s11)
        with pytest.raises(NonAscendingFrequencyError, match="S11"):
            touchstone.saves(4)

    @staticmethod
    def test_a_single_sample_is_ascending():
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(1_000_000, 0.1, 0.2)]
        assert touchstone.saves(1).splitlines()[-1] == "1000000 0.1 0.2"

    @staticmethod
    def test_save_refuses_before_opening_the_file(tmp_path):
        target = tmp_path / "descending.s1p"
        touchstone = Touchstone()
        touchstone.s11 = [
            Datapoint(3_000_000, 0.1, 0.2),
            Datapoint(1_000_000, 0.1, 0.2),
        ]
        touchstone.filename = str(target)
        with pytest.raises(NonAscendingFrequencyError):
            touchstone.save(1)
        assert not target.exists()

    @staticmethod
    def test_save_leaves_an_existing_file_untouched(tmp_path):
        target = tmp_path / "existing.s1p"
        target.write_text("sentinel\n", encoding="utf-8")
        touchstone = Touchstone()
        touchstone.s11 = [
            Datapoint(3_000_000, 0.1, 0.2),
            Datapoint(1_000_000, 0.1, 0.2),
        ]
        touchstone.filename = str(target)
        with pytest.raises(NonAscendingFrequencyError):
            touchstone.save(1)
        assert target.read_text(encoding="utf-8") == "sentinel\n"

    @staticmethod
    def test_an_ascending_dataset_still_round_trips(tmp_path):
        target = tmp_path / "ascending.s2p"
        touchstone = _s11_s21(points=3)
        touchstone.filename = str(target)
        touchstone.save(4)

        reloaded = Touchstone(str(target))
        reloaded.load()
        assert [dp.freq for dp in reloaded.s11] == [
            1_000_000,
            2_000_000,
            3_000_000,
        ]
        assert [dp.freq for dp in reloaded.s21] == [
            1_000_000,
            2_000_000,
            3_000_000,
        ]

    @staticmethod
    def test_a_repeated_frequency_is_accepted():
        """A log sweep's segment boundary repeats; that is not an error."""
        touchstone = Touchstone()
        touchstone.s11 = [
            Datapoint(1_000_000, 0.1, 0.2),
            Datapoint(2_000_000, 0.2, 0.2),
            Datapoint(2_000_000, 0.3, 0.2),
            Datapoint(3_000_000, 0.4, 0.2),
        ]
        rows = _data_rows(touchstone.saves(1))
        assert [row.split()[0] for row in rows] == [
            "1000000",
            "2000000",
            "2000000",
            "3000000",
        ]


def _log_sweep_grid(segments: int) -> list[int]:
    """Frequencies of a real segmented logarithmic ``Sweep``."""
    from NanoVNASaver.Settings.Sweep import Properties, Sweep

    sweep = Sweep(
        14_000_000,
        14_300_000,
        101,
        segments,
        Properties(logarithmic=True),
    )
    return list(sweep.get_frequencies())


def _data_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line[:1].isdigit()]


class TestLogarithmicSweepExport:
    """A log-mode sweep exports every sample, each frequency once."""

    @staticmethod
    @pytest.mark.parametrize("segments", [2, 3])
    def test_a_log_sweep_is_exported(segments):
        frequencies = _log_sweep_grid(segments)
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(f, 0.5, 0.1) for f in frequencies]

        rows = _data_rows(touchstone.saves(1))

        assert [int(row.split()[0]) for row in rows] == frequencies

    @staticmethod
    @pytest.mark.parametrize("segments", [2, 3])
    def test_no_segment_boundary_is_written_twice(segments):
        """The grid itself no longer repeats a boundary frequency.

        Each segment used to end on its successor's start, so the file held
        that frequency twice and only an equal-neighbour allowance let the
        export through.  ``Sweep.get_index_range`` now leaves a one-hertz gap
        between segments, so the allowance is no longer exercised.
        """
        frequencies = _log_sweep_grid(segments)
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(f, 0.5, 0.1) for f in frequencies]

        rows = _data_rows(touchstone.saves(1))
        written = [int(row.split()[0]) for row in rows]

        assert written == frequencies
        assert len(set(written)) == len(written)

    @staticmethod
    def test_one_log_sweep_round_trips(tmp_path):
        frequencies = _log_sweep_grid(2)
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(f, 0.5, 0.1) for f in frequencies]
        target = tmp_path / "log.s1p"
        touchstone.filename = str(target)
        touchstone.save(1)

        reloaded = Touchstone(str(target))
        reloaded.load()

        assert [dp.freq for dp in reloaded.s11] == frequencies

    @staticmethod
    def test_a_log_sweep_that_decreases_is_still_refused():
        """Allowing a repeat must not allow a reversal."""
        frequencies = list(reversed(_log_sweep_grid(2)))
        touchstone = Touchstone()
        touchstone.s11 = [Datapoint(f, 0.5, 0.1) for f in frequencies]

        with pytest.raises(NonAscendingFrequencyError, match="never"):
            touchstone.saves(1)


class TestSimulatedProvenance:
    @staticmethod
    def test_simulated_flag_is_serialized_and_restored():
        from NanoVNASaver.Touchstone import SIMULATED_MARKER

        dataset = Touchstone()
        dataset.s11 = [Datapoint(1_000_000, 0.1, 0.2)]
        dataset.simulated = True
        dataset.simulated_scenario = "attenuator"
        text = dataset.saves(1)

        assert text.count(SIMULATED_MARKER) == 1
        reloaded = Touchstone()
        reloaded.loads(text)
        assert reloaded.simulated
        assert reloaded.simulated_scenario == "attenuator"

    @staticmethod
    def test_physical_exports_never_claim_simulation():
        dataset = Touchstone()
        dataset.s11 = [Datapoint(1_000_000, 0.1, 0.2)]
        assert "SIMULATED" not in dataset.saves(1)
