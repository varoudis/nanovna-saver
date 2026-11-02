"""PERF-02: the optimised read paths, checked against hand-built frames.

Every expected value below is computed by hand from the record layout, not
by calling the code under test: a decoder test that recomputes the
implementation's own arithmetic proves nothing.
"""

from __future__ import annotations

from struct import pack

import pytest

from NanoVNASaver.Hardware.CombinedScan import (
    BINARY_RECORD_SIZE,
    PATH_BINARY,
    PATH_COMBINED,
    PATH_TRUSTED,
    CombinedSweep,
    MalformedDataError,
    OptimisedReadError,
    ShortDataError,
    decode_scan_bin,
    one_shot_read,
    parse_combined_reply,
    preferences_from,
)
from NanoVNASaver.Hardware.NanoVNA_V2 import NanoVNA_V2
from NanoVNASaver.Hardware.ScanCapability import ScanPreferences


def record(fwd, rev0, rev1, index):
    """One 32-byte FIFO record; ``fwd``/``rev0``/``rev1`` are I/Q pairs."""
    return pack(
        "<iiiiiihxxxxxx",
        fwd[0],
        fwd[1],
        rev0[0],
        rev0[1],
        rev1[0],
        rev1[1],
        index,
    )


#: Three points whose ratios are exact in binary floating point:
#: S11 = 500/1000 = 0.5, S21 = 250/1000 = 0.25, and so on.
FRAME = b"".join(
    [
        record((1000, 0), (500, 0), (250, 0), 0),
        record((2000, 0), (0, 1000), (0, 500), 1),
        record((4000, 0), (-1000, 0), (-2000, 0), 2),
    ]
)


class TestBinaryDecode:
    @staticmethod
    def test_the_frame_decodes_to_the_hand_computed_ratios():
        sweep = decode_scan_bin(FRAME, points=3)

        assert sweep.frequencies == (0, 1, 2)
        assert sweep.s11 == (
            complex(0.5, 0),
            complex(0, 0.5),
            complex(-0.25, 0),
        )
        assert sweep.s21 == (
            complex(0.25, 0),
            complex(0, 0.25),
            complex(-0.5, 0),
        )
        assert sweep.path == PATH_BINARY

    @staticmethod
    def test_a_record_is_thirty_two_bytes_of_little_endian():
        assert BINARY_RECORD_SIZE == 32
        assert len(record((1, 2), (3, 4), (5, 6), 7)) == 32
        assert record((1, 0), (0, 0), (0, 0), 0)[:4] == b"\x01\x00\x00\x00"

    @staticmethod
    def test_records_may_arrive_out_of_order():
        shuffled = FRAME[64:] + FRAME[:64]
        sweep = decode_scan_bin(shuffled, points=3)
        assert sweep.s11[0] == complex(0.5, 0)
        assert sweep.s11[2] == complex(-0.25, 0)

    @staticmethod
    def test_a_one_port_request_never_invents_s21():
        sweep = decode_scan_bin(FRAME, points=3, channels=("data 0",))
        assert sweep.s21 == ()
        assert sweep.channels == ("data 0",)
        with pytest.raises(MalformedDataError):
            sweep.channel("data 1")

    @staticmethod
    def test_a_short_buffer_is_rejected_not_padded():
        with pytest.raises(ShortDataError):
            decode_scan_bin(FRAME[:-1], points=3)

    @staticmethod
    def test_a_long_buffer_is_rejected_not_trimmed():
        with pytest.raises(ShortDataError):
            decode_scan_bin(FRAME + FRAME[:32], points=3)

    @staticmethod
    def test_a_duplicate_index_is_rejected():
        payload = (
            record((1000, 0), (500, 0), (250, 0), 0)
            + record((1000, 0), (500, 0), (250, 0), 0)
            + record((1000, 0), (500, 0), (250, 0), 2)
        )
        with pytest.raises(MalformedDataError, match="duplicate"):
            decode_scan_bin(payload, points=3)

    @staticmethod
    def test_an_index_outside_the_sweep_is_rejected():
        payload = (
            record((1000, 0), (500, 0), (250, 0), 0)
            + record((1000, 0), (500, 0), (250, 0), 1)
            + record((1000, 0), (500, 0), (250, 0), 3)
        )
        with pytest.raises(MalformedDataError, match="outside"):
            decode_scan_bin(payload, points=3)

    @staticmethod
    def test_a_zero_forward_sample_is_rejected():
        payload = record((0, 0), (500, 0), (250, 0), 0)
        with pytest.raises(MalformedDataError, match="forward"):
            decode_scan_bin(payload, points=1)

    @staticmethod
    def test_an_impossible_magnitude_is_rejected():
        # 100000/1000 = 100, far beyond any real S parameter.
        payload = record((1000, 0), (100_000, 0), (0, 0), 0)
        with pytest.raises(MalformedDataError, match="exceeds"):
            decode_scan_bin(payload, points=1)

    @staticmethod
    def test_no_points_asked_for_is_short_data():
        with pytest.raises(ShortDataError):
            decode_scan_bin(b"", points=0)


class TestCombinedTextDecode:
    @staticmethod
    def test_two_channels_decode_into_the_requested_grid():
        sweep = parse_combined_reply(
            ["0.5 0 0.25 0", "0 0.5 0 0.25"],
            points=2,
            frequencies=(1_000_000, 2_000_000),
            start=1_000_000,
            stop=2_000_000,
        )
        assert sweep.frequencies == (1_000_000, 2_000_000)
        assert sweep.s11 == (complex(0.5, 0), complex(0, 0.5))
        assert sweep.s21 == (complex(0.25, 0), complex(0, 0.25))
        assert sweep.path == PATH_COMBINED

    @staticmethod
    def test_one_channel_needs_two_fields():
        sweep = parse_combined_reply(
            ["0.5 0", "0 0.5"], points=2, channels=("data 0",)
        )
        assert sweep.s21 == ()

    @staticmethod
    def test_too_few_points_is_short_data():
        with pytest.raises(ShortDataError):
            parse_combined_reply(["0.5 0 0.25 0"], points=2)

    @staticmethod
    def test_too_many_points_is_rejected():
        with pytest.raises(ShortDataError):
            parse_combined_reply(["0.5 0 0.25 0"] * 3, points=2)

    @staticmethod
    def test_a_wrong_field_count_is_malformed():
        with pytest.raises(MalformedDataError, match="values"):
            parse_combined_reply(["0.5 0 0.25"], points=1)

    @staticmethod
    def test_garbage_fields_are_malformed():
        with pytest.raises(MalformedDataError):
            parse_combined_reply(["0.5 x 0.25 0"], points=1)

    @staticmethod
    def test_a_non_finite_value_is_malformed():
        with pytest.raises(MalformedDataError, match="non-finite"):
            parse_combined_reply(["nan 0 0.25 0"], points=1)

    @staticmethod
    def test_a_grid_that_misses_the_requested_stop_is_malformed():
        with pytest.raises(MalformedDataError, match="grid"):
            parse_combined_reply(
                ["0.5 0 0.25 0", "0.5 0 0.25 0"],
                points=2,
                frequencies=(1, 2),
                start=1,
                stop=3,
            )


class TestFallback:
    """A rejected optimised read always becomes a complete trusted sweep."""

    @staticmethod
    def _trusted() -> CombinedSweep:
        return CombinedSweep(
            frequencies=(1, 2), s11=(1 + 0j, 2 + 0j), s21=(3 + 0j, 4 + 0j)
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (ShortDataError("only one point"), "short"),
            (MalformedDataError("garbage"), "malformed"),
            (OptimisedReadError("nonsense"), "rejected"),
            (IOError("port vanished"), "error"),
        ],
    )
    def test_every_failure_kind_falls_back_and_is_reported(error, kind):
        outcome = one_shot_read(
            lambda: (_ for _ in ()).throw(error), TestFallback._trusted
        )

        assert outcome.path == PATH_TRUSTED
        assert outcome.fell_back
        assert not outcome.optimised
        assert outcome.failure is not None
        assert outcome.failure.kind == kind
        assert kind in outcome.failure.describe()
        assert outcome.sweep == TestFallback._trusted()

    @staticmethod
    def test_an_accepted_read_is_never_mixed_with_trusted_data():
        def trusted():  # pragma: no cover - must not be called
            raise AssertionError("trusted path used for a valid read")

        optimised = CombinedSweep(
            frequencies=(1, 2), s11=(0.5 + 0j, 0.25 + 0j), s21=(0 + 0j, 0 + 0j)
        )
        outcome = one_shot_read(lambda: optimised, trusted)

        assert outcome.path == PATH_COMBINED
        assert outcome.sweep is optimised
        assert not outcome.fell_back

    @staticmethod
    def test_a_failing_trusted_read_propagates():
        with pytest.raises(IOError):
            one_shot_read(
                lambda: (_ for _ in ()).throw(ShortDataError("short")),
                lambda: (_ for _ in ()).throw(IOError("device gone")),
            )


class TestPreferencesHelper:
    @staticmethod
    def test_the_helper_defaults_to_the_trusted_path():
        assert preferences_from() == ScanPreferences()
        assert not preferences_from().enabled
        assert preferences_from(binary_scan=True).enabled


class TestExistingCombinedReads:
    """The drivers that already read both channels in one go, unchanged."""

    @staticmethod
    def test_the_v2_fifo_decode_divides_by_the_forward_sample():
        # Built without __init__: this is the decode the driver has always
        # done, and it must keep matching the hand-computed ratios.
        vna = object.__new__(NanoVNA_V2)
        vna._sweepdata = [(complex(), complex())] * 3

        vna._read_pointstoread(3, FRAME)

        assert vna._sweepdata[0] == (complex(0.5, 0), complex(0.25, 0))
        assert vna._sweepdata[1] == (complex(0, 0.5), complex(0, 0.25))
        assert vna._sweepdata[2] == (complex(-0.25, 0), complex(-0.5, 0))
