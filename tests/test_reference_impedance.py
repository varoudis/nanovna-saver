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
"""RF-02: reference-impedance renormalization, propagated end to end.

The textbook transformation is written out here from first principles and is
never taken from the module under test:

    Z     = Z0 * (1 + G) / (1 - G)        (one-port load impedance)
    G     = (Z - Z0) / (Z + Z0)
    a     = (Z0_old - Z0_new) / (Z0_old + Z0_new)
    G_new = (G + a) / (1 + a * G)

The real check is *invariance of the physical load*: ``Z`` computed against
the old reference before renormalization must equal ``Z`` computed against
the new reference afterwards.  G is a ratio and legitimately changes; the
load may not.
"""

import math
from pathlib import Path

import pytest

from NanoVNASaver.Measurement import snapshot_from_touchstone
from NanoVNASaver.Processing import (
    PORT1_CHANNELS,
    ProcessingError,
    apply_processing,
    apply_renormalize,
    renormalize_gamma,
    renormalize_stage,
)
from NanoVNASaver.ReferenceImpedance import (
    AUDIT,
    AUDIT_BY_LOCATION,
    AUDIT_TABLE,
    HARD_CODED_REFUSED,
    HARD_CODED_REMAINING,
    PROPAGATED,
    STATUSES,
    ReferenceImpedanceError,
    format_reference_impedance,
    remaining_call_sites,
    renormalization_blocker,
    require_reference_impedance,
)
from NanoVNASaver.RFTools import (
    Datapoint,
    gamma_to_impedance,
    impedance_to_gamma,
)
from NanoVNASaver.Touchstone import (
    InvalidReferenceImpedanceError,
    Options,
    Touchstone,
    TouchstoneExportError,
)

# ----------------------------------------------------------------------
# Independent textbook formulas (the reference the module is judged by)
# ----------------------------------------------------------------------


def textbook_gamma(z: complex, z0: float) -> complex:
    """G = (Z - Z0) / (Z + Z0)."""
    return (z - z0) / (z + z0)


def textbook_impedance(gamma: complex, z0: float) -> complex:
    """Z = Z0 * (1 + G) / (1 - G)."""
    return z0 * (1 + gamma) / (1 - gamma)


def textbook_renormalize(
    gamma: complex, z0_old: float, z0_new: float
) -> complex:
    """G' = (G + a) / (1 + a*G), a = (Z0_old - Z0_new)/(Z0_old + Z0_new)."""
    a = (z0_old - z0_new) / (z0_old + z0_new)
    return (gamma + a) / (1 + a * gamma)


#: Loads spanning the interesting cases: matched, resistive above and below
#: the reference, strongly reactive, near-open and near-short.
LOADS: tuple[tuple[str, complex], ...] = (
    ("matched 50 ohm", complex(50.0, 0.0)),
    ("resistive 100 ohm", complex(100.0, 0.0)),
    ("resistive 25 ohm", complex(25.0, 0.0)),
    ("capacitive 10 - j25", complex(10.0, -25.0)),
    ("inductive 10 + j25", complex(10.0, 25.0)),
    ("near-open 1e6 ohm", complex(1.0e6, 0.0)),
    ("near-short 1e-3 ohm", complex(1.0e-3, 0.0)),
    ("near-open reactive 1e5 + j1e5", complex(1.0e5, 1.0e5)),
    ("near-short reactive 0.01 + j1e-4", complex(0.01, 1.0e-4)),
)


class TestOnePortMathAgainstTextbook:
    """``gamma_to_impedance``/``impedance_to_gamma`` versus the formulas."""

    @pytest.mark.parametrize("z0", [50.0, 75.0])
    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_gamma_matches_textbook(self, z0, label, load):
        assert impedance_to_gamma(load, z0) == pytest.approx(
            textbook_gamma(load, z0), rel=1e-15, abs=1e-15
        ), label

    @pytest.mark.parametrize("z0", [50.0, 75.0])
    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_impedance_matches_textbook(self, z0, label, load):
        gamma = textbook_gamma(load, z0)
        assert gamma_to_impedance(gamma, z0) == pytest.approx(
            textbook_impedance(gamma, z0), rel=1e-9, abs=1e-12
        ), label

    @pytest.mark.parametrize("z0", [50.0, 75.0])
    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_gamma_and_impedance_are_inverses(self, z0, label, load):
        # Rounding through gamma costs relative precision on loads far from
        # the reference, so the tolerance is relative, not absolute.
        gamma = impedance_to_gamma(load, z0)
        assert gamma_to_impedance(gamma, z0) == pytest.approx(load, rel=1e-9), (
            label
        )

    def test_matched_load_gives_zero_reflection(self):
        for z0 in (50.0, 75.0, 93.0):
            assert impedance_to_gamma(complex(z0, 0.0), z0) == 0

    @pytest.mark.parametrize("z0", [50.0, 75.0])
    def test_open_and_short_approach_the_unit_circle(self, z0):
        # The exact limits, not 1.0: a 1e12 ohm load against 75 ohm is
        # (1e12 - 75)/(1e12 + 75), which is 1 - 1.5e-10 off unity.
        wide_open = complex(1.0e12, 0.0)
        dead_short = complex(1.0e-12, 0.0)
        assert impedance_to_gamma(wide_open, z0).real == pytest.approx(
            textbook_gamma(wide_open, z0).real, rel=1e-15
        )
        assert impedance_to_gamma(wide_open, z0).real == pytest.approx(
            1.0, abs=1e-6
        )
        assert impedance_to_gamma(dead_short, z0).real == pytest.approx(
            -1.0, abs=1e-6
        )

    def test_invalid_reference_impedance_refused(self):
        for bad in (0.0, -50.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="positive finite"):
                impedance_to_gamma(complex(50.0, 0.0), bad)

    def test_non_finite_impedance_refused(self):
        with pytest.raises(ValueError, match="must be finite"):
            impedance_to_gamma(complex(math.inf, 0.0), 50.0)


class TestRenormalizeRoundTrip:
    """50 -> 75 -> 50 is exact, and the physical load never moves."""

    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_round_trip_returns_the_same_gamma(self, label, load):
        gamma = impedance_to_gamma(load, 50.0)
        there = renormalize_gamma(50.0, 75.0, gamma)
        back = renormalize_gamma(75.0, 50.0, there)
        assert back == pytest.approx(gamma, rel=1e-12, abs=1e-12), label

    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_load_impedance_is_invariant(self, label, load):
        """The check that matters: Z is the same before and after.

        The gamma round trip above is bounded at 1e-12.  Turning an extreme
        gamma (a 1 mohm load against 50 ohm is gamma = -0.99996) back into
        an impedance through ``(1 + G)/(1 - G)`` costs a few ulp of
        *relative* precision, so the invariance is asserted relatively.
        """
        gamma_50 = impedance_to_gamma(load, 50.0)
        gamma_75 = renormalize_gamma(50.0, 75.0, gamma_50)
        assert textbook_impedance(gamma_50, 50.0) == pytest.approx(
            load, rel=1e-10
        ), label
        assert textbook_impedance(gamma_75, 75.0) == pytest.approx(
            load, rel=1e-10
        ), label
        # Both sides agree with each other to the same precision: the load
        # did not move, only the number that describes it.
        assert textbook_impedance(gamma_75, 75.0) == pytest.approx(
            textbook_impedance(gamma_50, 50.0), rel=1e-10
        ), label

    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_matches_the_textbook_transform(self, label, load):
        gamma = textbook_gamma(load, 50.0)
        assert renormalize_gamma(50.0, 75.0, gamma) == pytest.approx(
            textbook_renormalize(gamma, 50.0, 75.0), rel=1e-15, abs=1e-15
        ), label

    @pytest.mark.parametrize(
        "gamma",
        [
            complex(0.999999, 0.0),
            complex(-0.999999, 0.0),
            complex(0.9999, 0.001),
            complex(-0.9999, -0.001),
            complex(0.0, 0.0),
            complex(0.0, 1.0),
        ],
        ids=[
            "near-open",
            "near-short",
            "near-open reactive",
            "near-short reactive",
            "matched",
            "unit reactive",
        ],
    )
    def test_round_trip_near_the_unit_circle(self, gamma):
        there = renormalize_gamma(50.0, 75.0, gamma)
        back = renormalize_gamma(75.0, 50.0, there)
        assert back == pytest.approx(gamma, rel=1e-12, abs=1e-12)

    def test_load_equal_to_the_new_reference_maps_to_zero(self):
        # Z0_old = 50, Z0_new = 75, so the 75 ohm load must become G = 0.
        gamma = impedance_to_gamma(complex(75.0, 0.0), 50.0)
        assert renormalize_gamma(50.0, 75.0, gamma) == pytest.approx(
            0.0, abs=1e-15
        )
        # And symmetrically for a 50 ohm load against a 50 ohm reference.
        assert renormalize_gamma(75.0, 50.0, impedance_to_gamma(50j, 75.0)) != 0
        assert renormalize_gamma(
            75.0, 50.0, impedance_to_gamma(complex(50.0, 0.0), 75.0)
        ) == pytest.approx(0.0, abs=1e-15)

    def test_same_reference_is_the_identity(self):
        gamma = complex(0.3, -0.4)
        assert renormalize_gamma(50.0, 50.0, gamma) == gamma

    def test_singular_refused(self):
        # a = (50 - 75)/(50 + 75) = -0.2, so G = 5 makes 1 + a*G zero.
        a = (50.0 - 75.0) / (50.0 + 75.0)
        assert a == pytest.approx(-0.2)
        with pytest.raises(ProcessingError, match="singular"):
            renormalize_gamma(50.0, 75.0, complex(-1.0 / a, 0.0))

    def test_non_positive_and_non_finite_reference_refused(self):
        gamma = complex(0.1, 0.0)
        for bad in (0.0, -75.0, math.inf, math.nan):
            with pytest.raises(ProcessingError, match="positive|finite"):
                renormalize_gamma(50.0, bad, gamma)
            with pytest.raises(ProcessingError, match="positive|finite"):
                renormalize_gamma(bad, 75.0, gamma)


class TestApplyRenormalize:
    """The channel-level entry point, including what it refuses."""

    @staticmethod
    def _channel(load: complex, z0: float, freq: int = 1_000_000):
        gamma = impedance_to_gamma(load, z0)
        return (Datapoint(freq, gamma.real, gamma.imag),)

    @pytest.mark.parametrize(("label", "load"), LOADS)
    def test_reflection_channel_renormalized(self, label, load):
        stage = renormalize_stage(75.0)
        source = {"S11": self._channel(load, 50.0)}
        result = apply_renormalize(source, stage, current_reference=50.0)
        gamma = result["S11"][0].z
        assert textbook_impedance(gamma, 75.0) == pytest.approx(
            load, rel=1e-9
        ), label
        # The frequency grid is untouched.
        assert result["S11"][0].freq == 1_000_000

    def test_s22_is_renormalized_and_other_names_pass_through(self):
        stage = renormalize_stage(75.0)
        source = {
            "S11": self._channel(complex(100.0, 0.0), 50.0),
            "S22": self._channel(complex(25.0, 0.0), 50.0),
            "DC": self._channel(complex(60.0, 0.0), 50.0),
        }
        result = apply_renormalize(source, stage, current_reference=50.0)
        assert textbook_impedance(result["S11"][0].z, 75.0) == pytest.approx(
            complex(100.0, 0.0), rel=1e-12
        )
        assert textbook_impedance(result["S22"][0].z, 75.0) == pytest.approx(
            complex(25.0, 0.0), rel=1e-12
        )
        # An unknown channel is not a reflection coefficient: unchanged.
        assert result["DC"] == source["DC"]

    def test_equal_reference_is_a_copy(self):
        stage = renormalize_stage(50.0)
        source = {"S11": self._channel(complex(100.0, 0.0), 50.0)}
        result = apply_renormalize(source, stage, current_reference=50.0)
        assert result["S11"] == source["S11"]

    @pytest.mark.parametrize("channel", ["S21", "S12"])
    def test_transmission_channel_refused_with_reason(self, channel):
        stage = renormalize_stage(75.0)
        source = {
            "S11": self._channel(complex(100.0, 0.0), 50.0),
            channel: (Datapoint(1_000_000, 0.5, 0.1),),
        }
        with pytest.raises(ProcessingError) as caught:
            apply_renormalize(source, stage, current_reference=50.0)
        message = str(caught.value)
        # The refusal has to name what is refused and why, or a pipeline
        # that silently drops a stage would be indistinguishable from one
        # that applied it.
        assert "transmission channel" in message
        assert "not supported" in message
        assert "reflection data" in message
        # The pre-flight blocker in this module names the offending channel
        # and the missing transform, so a control can refuse before the run
        # starts instead of failing after the first segment.
        blocker = renormalization_blocker(source)
        assert blocker is not None
        assert channel in blocker
        assert "wave-cascading" in blocker
        assert "two-port" in blocker
        assert "not implemented" in blocker

    def test_no_blocker_for_reflection_only_data(self):
        assert renormalization_blocker({"S11": (), "S22": ()}) is None
        assert renormalization_blocker(PORT1_CHANNELS) is None

    def test_equal_references_are_never_blocked(self):
        """A no-op renormalization cannot be blocked: it changes nothing.

        ``apply_renormalize`` returns a copy, transmission channels
        included, when the target already equals the current reference.
        The pre-flight must agree, or a caller would refuse a run that
        would have succeeded.
        """
        stage = renormalize_stage(50.0)
        source = {
            "S11": self._channel(complex(100.0, 0.0), 50.0),
            "S21": (Datapoint(1_000_000, 0.5, 0.1),),
        }
        channels = {"S11": (), "S21": (), "S12": ()}
        assert renormalization_blocker(channels) is not None
        assert renormalization_blocker(channels, equal_references=True) is None
        # The claim above is not an assumption: at the same reference the
        # stage really does return the transmission channel untouched.
        result = apply_renormalize(source, stage, current_reference=50.0)
        assert result["S21"] == source["S21"]
        assert result["S11"] == source["S11"]
        # The guard is keyword-only, so an existing positional call cannot
        # silently start meaning something else.
        with pytest.raises(TypeError):
            renormalization_blocker(channels, True)

    def test_invalid_target_refused(self):
        stage = renormalize_stage(75.0)
        source = {"S11": self._channel(complex(100.0, 0.0), 50.0)}
        for bad in (-75.0, 0.0, math.inf, math.nan):
            broken = type(stage).build(
                "renormalize", {"reference_impedance": bad}
            )
            with pytest.raises(
                ProcessingError, match="positive finite|must be finite"
            ):
                apply_renormalize(source, broken, current_reference=50.0)

    def test_stage_builder_refuses_non_positive(self):
        with pytest.raises(ProcessingError, match="positive"):
            renormalize_stage(0)


class TestPipelinePropagatesTheReference:
    """``apply_processing`` carries the new Z0 out with the data."""

    def test_pipeline_reports_the_renormalized_reference(self):
        from NanoVNASaver.Processing import ProcessingConfig

        config = ProcessingConfig.build(renormalize_stage(75.0))
        load = complex(100.0, -20.0)
        gamma = impedance_to_gamma(load, 50.0)
        result = apply_processing(
            {"S11": (Datapoint(1_000_000, gamma.real, gamma.imag),)},
            config,
            reference_impedance=50.0,
        )
        assert result.reference_impedance == 75.0
        assert textbook_impedance(result.channels["S11"][0].z, 75.0) == (
            pytest.approx(load, rel=1e-12)
        )
        assert "75 ohm" in config.describe()

    def test_identity_pipeline_keeps_the_reference(self):
        from NanoVNASaver.Processing import ProcessingConfig

        result = apply_processing(
            {"S11": (Datapoint(1_000_000, 0.1, 0.0),)},
            ProcessingConfig(),
            reference_impedance=75.0,
        )
        assert result.reference_impedance == 75.0


# ----------------------------------------------------------------------
# Touchstone: the R keyword is read, kept and written back
# ----------------------------------------------------------------------


def _reload(text: str, tmp_path, suffix: str = ".s1p") -> Touchstone:
    path = tmp_path / f"reloaded{suffix}"
    path.write_text(text, encoding="utf-8")
    dataset = Touchstone(str(path))
    dataset.load()
    return dataset


def _option_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("#"))


class TestTouchstoneReferenceImpedance:
    def test_seventy_five_is_read_from_the_header(self):
        dataset = Touchstone()
        dataset.loads_strict(
            "! a 75 ohm measurement\n"
            "# HZ S RI R 75\n"
            "1000000 0.2 0.0\n"
            "2000000 0.1 0.05\n"
        )
        assert dataset.opts.resistance == 75
        assert dataset.r == 75

    def test_fifty_is_read_from_the_header(self):
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI R 50\n1000000 0.2 0.0\n")
        assert dataset.opts.resistance == 50
        assert str(dataset.opts) == "# HZ S RI R 50"

    def test_missing_r_keeps_the_default_and_says_so(self):
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI\n1000000 0.2 0.0\n")
        assert dataset.opts.resistance == 50

    @pytest.mark.parametrize(
        ("line", "unit", "t_format"),
        [
            ("# HZ S RI R 75", "hz", "ri"),
            ("# R 75 HZ S RI", "hz", "ri"),
            ("# HZ R 75 S RI", "hz", "ri"),
            ("# ri r 75 s hz", "hz", "ri"),
            ("# R 75 ri s hz", "hz", "ri"),
        ],
    )
    def test_r_keyword_position_is_irrelevant(self, line, unit, t_format):
        dataset = Touchstone()
        dataset.loads_strict(f"{line}\n1000000 0.2 0.0\n")
        assert dataset.opts.resistance == 75, line
        assert dataset.opts.unit == unit
        assert dataset.opts.format == t_format

    @pytest.mark.parametrize("value", ["0", "-75", "nan", "inf", "-inf", "abc"])
    def test_invalid_r_rejected_by_a_strict_load(self, value):
        dataset = Touchstone()
        with pytest.raises(InvalidReferenceImpedanceError):
            dataset.loads_strict(f"# HZ S RI R {value}\n1000000 0.2 0.0\n")

    def test_r_without_a_value_rejected(self):
        dataset = Touchstone()
        with pytest.raises(InvalidReferenceImpedanceError, match="no value"):
            dataset.loads_strict("# HZ S RI R\n1000000 0.2 0.0\n")

    def test_invalid_r_is_a_type_error_so_a_lenient_load_drops_the_file(
        self, caplog
    ):
        dataset = Touchstone()
        dataset.loads("# HZ S RI R 0\n1000000 0.2 0.0\n")
        assert dataset.s11 == []
        assert "R keyword" in caplog.text

    def test_written_back_for_a_seventy_five_ohm_file(self, tmp_path):
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI R 75\n1000000 0.2 0.0\n")
        text = dataset.saves(1)
        assert _option_line(text) == "# HZ S RI R 75"
        assert _reload(text, tmp_path).opts.resistance == 75

    def test_written_back_for_a_fifty_ohm_file(self, tmp_path):
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI R 50\n1000000 0.2 0.0\n")
        text = dataset.saves(1)
        assert _option_line(text) == "# HZ S RI R 50"
        assert _reload(text, tmp_path).opts.resistance == 50

    def test_written_back_for_a_fractional_reference(self, tmp_path):
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI R 75.5\n1000000 0.2 0.0\n")
        text = dataset.saves(1)
        assert _option_line(text) == "# HZ S RI R 75.5"
        assert _reload(text, tmp_path).opts.resistance == 75.5

    def test_writer_refuses_an_unusable_reference(self):
        dataset = Touchstone()
        dataset.s11 = [Datapoint(1_000_000, 0.2, 0.0)]
        for bad in (0, -50, math.inf, math.nan):
            dataset.opts.resistance = bad
            with pytest.raises(TouchstoneExportError, match="positive finite"):
                dataset.saves(1)

    def test_reference_survives_into_the_snapshot(self):
        dataset = Touchstone()
        dataset.loads_strict(
            "# HZ S RI R 75\n1000000 0.2 0.0\n2000000 0.1 0.0\n"
        )
        snapshot = snapshot_from_touchstone(dataset, name="seventyfive")
        assert snapshot.reference_impedance == 75.0
        # The retained reference is the one used to turn gamma into ohms.
        point = snapshot.channel("S11")[0]
        assert point.impedance(snapshot.reference_impedance) == pytest.approx(
            complex(75.0, 0.0) * (1 + 0.2) / (1 - 0.2), rel=1e-12
        )

    def test_snapshot_of_a_fifty_ohm_file(self):
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI R 50\n1000000 0.2 0.0\n")
        assert snapshot_from_touchstone(dataset).reference_impedance == 50.0

    def test_write_after_renormalizing_a_one_port_file(self, tmp_path):
        """End to end: read 50, renormalize to 75, write 75."""
        dataset = Touchstone()
        dataset.loads_strict("# HZ S RI R 50\n1000000 0.2 0.0\n")
        snapshot = snapshot_from_touchstone(dataset)
        load = snapshot.channel("S11")[0].impedance(50.0)
        stage = renormalize_stage(75.0)
        source = {"S11": snapshot.channel("S11")}
        renormalized = apply_renormalize(source, stage, current_reference=50.0)
        dataset.opts.resistance = 75.0
        dataset.s11 = list(renormalized["S11"])
        text = dataset.saves(1)
        assert _option_line(text) == "# HZ S RI R 75"
        reloaded = _reload(text, tmp_path)
        assert reloaded.opts.resistance == 75
        assert reloaded.s11[0].impedance(75.0) == pytest.approx(load, rel=1e-12)


class TestReferenceImpedanceHelpers:
    def test_require_accepts_positive_finite(self):
        assert require_reference_impedance(75) == 75.0
        assert require_reference_impedance("75.5") == 75.5

    @pytest.mark.parametrize(
        "bad", [0, -1, math.inf, -math.inf, math.nan, None, "abc"]
    )
    def test_require_refuses_unusable(self, bad):
        with pytest.raises(ReferenceImpedanceError, match="positive finite"):
            require_reference_impedance(bad)

    @pytest.mark.parametrize("bad", [True, False])
    def test_require_refuses_a_boolean(self, bad):
        """``True`` is an ``int`` in Python, but it is not an impedance."""
        with pytest.raises(ReferenceImpedanceError, match="must be a number"):
            require_reference_impedance(bad)

    def test_format_matches_conventional_spelling(self):
        assert format_reference_impedance(50) == "50"
        assert format_reference_impedance(50.0) == "50"
        assert format_reference_impedance(75) == "75"
        assert format_reference_impedance(75.5) == "75.5"
        with pytest.raises(ReferenceImpedanceError):
            format_reference_impedance(0)


class TestAudit:
    def test_every_status_is_used(self):
        used = {entry.status for entry in AUDIT_TABLE}
        assert used == set(STATUSES)

    def test_audit_mapping_agrees_with_the_table(self):
        assert dict(AUDIT) == {e.location: e.status for e in AUDIT_TABLE}
        assert set(AUDIT_BY_LOCATION) == set(AUDIT)

    def test_locations_are_unique_and_file_colon_line(self):
        assert len(AUDIT_TABLE) == len(AUDIT)
        for location in AUDIT:
            path, _, line = location.rpartition(":")
            assert path.endswith(".py"), location
            assert line.isdigit() and int(line) > 0, location

    def test_remaining_sites_are_listed_and_all_pending(self):
        remaining = remaining_call_sites()
        assert remaining
        assert all(entry.status == HARD_CODED_REMAINING for entry in remaining)
        assert all(entry.pending for entry in remaining)

    def test_known_sites_have_the_expected_status(self):
        def statuses_of(path):
            return {
                entry.status
                for location, entry in AUDIT_BY_LOCATION.items()
                if location.rpartition(":")[0] == path
            }

        # Looked up by file rather than by line number: the audit must not
        # fail merely because unrelated code above a site moved.
        assert PROPAGATED in statuses_of("src/NanoVNASaver/Measurement.py")
        assert statuses_of("src/NanoVNASaver/CalibrationKits.py") == {
            HARD_CODED_REFUSED
        }
        # The charts now carry the reference from the data they draw.
        assert statuses_of("src/NanoVNASaver/Charts/RIZ.py") == {PROPAGATED}
        assert statuses_of("src/NanoVNASaver/Charts/Chart.py") == {PROPAGATED}

    def test_every_location_names_an_existing_file_and_line(self):
        """The audit cannot rot into pointers at code that moved away."""
        for entry in AUDIT_TABLE:
            path, _, line = entry.location.partition(":")
            source = Path(path)
            assert source.is_file(), entry.location
            line_count = len(source.read_text(encoding="utf-8").splitlines())
            assert 1 <= int(line) <= line_count, entry.location

    def test_audit_rows_survive_into_the_public_api(self):
        from NanoVNASaver import ReferenceImpedance as module

        assert module.audit_with_status(PROPAGATED)
        with pytest.raises(ValueError, match="unknown audit status"):
            module.audit_with_status("nonsense")

    def test_audit_entry_validation(self):
        from NanoVNASaver.ReferenceImpedance import AuditEntry

        with pytest.raises(ValueError, match="unknown audit status"):
            AuditEntry("a.py:1", "maybe", "note")
        with pytest.raises(ValueError, match="location and a note"):
            AuditEntry("", PROPAGATED, "note")


class TestOptionsKeywordHandling:
    """The reader still rejects what it always rejected."""

    def test_illegal_option_line(self):
        options = Options()
        with pytest.raises(TypeError, match="Illegal option line"):
            options.parse("# ILLEGAL")
        with pytest.raises(TypeError, match="Not an option line"):
            options.parse("")

    def test_defaults(self):
        options = Options()
        assert str(options) == "# GHZ S MA R 50"
        assert options.resistance == 50

    def test_constructor_refuses_an_unusable_reference(self):
        """The object that writes option lines cannot hold nonsense."""
        for bad in (0, -75, math.nan, math.inf, True):
            with pytest.raises(InvalidReferenceImpedanceError):
                Options(resistance=bad)

    def test_constructor_accepts_and_keeps_a_fractional_reference(self):
        options = Options(resistance=75.5)
        assert options.resistance == 75.5
        assert str(options) == "# GHZ S MA R 75.5"
