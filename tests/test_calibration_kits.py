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
"""CAL-01: a versioned calibration kit library.

The kit file is the user-facing artefact, so these tests pin its JSON
schema, its validation messages, its atomic storage and - most importantly -
that applying a kit produces exactly the standard description the hand
entered coefficient fields produce.  No correction coefficient is computed
here: the kit layer only describes standards, ``Calibration`` still does all
of the maths.
"""

import json
import math
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from NanoVNASaver.Calibration import CalElement, Calibration
from NanoVNASaver.CalibrationKits import (
    BUILTIN_NAMES,
    COEFFICIENT_FIELDS,
    IDEAL_KIT,
    KIT_FORMAT,
    KIT_VERSION,
    LIBRARY_FORMAT,
    LIBRARY_VERSION,
    SMA_MALE_KIT,
    CalibrationKit,
    CalibrationKitError,
    CalibrationKitNameError,
    CalibrationKitValidationError,
    KitConnector,
    KitLibrary,
    KitStandard,
    KitValidSpan,
    builtin_kits,
    default_library_directory,
    displayed_value,
    find_builtin,
    kit_from_document,
    kit_from_legacy_values,
    kits_from_document,
    legacy_values,
    stored_value,
)

FREQUENCIES = sorted(
    {1_000_000 * 3**n for n in range(0, 12)}
    | {10_000_000, 25_000_000, 100_000_000, 433_000_000, 1_000_000_000}
)


def _coefficient_kit(name: str = "Test kit", **overrides) -> CalibrationKit:
    """A kit built from awkward numbers the window could have been given.

    Values are entered the way the UI enters them - a mantissa in the
    displayed unit, divided by the field's divisor - so the kit holds
    exactly what a hand-typed calibration would hold.
    """
    shown = {
        ("short", "l0"): 5.7,
        ("short", "l1"): -0.896,
        ("short", "l2"): -0.011,
        ("short", "l3"): -0.000412,
        ("short", "length"): -34.2,
        ("open", "c0"): 2.1,
        ("open", "c1"): 0.0567,
        ("open", "c2"): -0.000239,
        ("open", "c3"): 0.0000002,
        ("open", "length"): 1.5,
        ("load", "r"): 50.2,
        ("load", "l"): 1230.0,
        ("load", "c"): 81.0,
        ("load", "length"): 2.5,
        ("through", "length"): 4.5,
    }
    divisors = {
        (standard, key): divisor
        for standard, key, _legacy, divisor, _default in COEFFICIENT_FIELDS
    }
    standards = {}
    for standard in ("short", "open", "load", "through"):
        standards[standard] = KitStandard(
            model="coeff",
            coefficients={
                key: value / divisors[(name_, key)]
                for (name_, key), value in shown.items()
                if name_ == standard
            },
        )
    kwargs = {
        "name": name,
        "description": "Fixture kit",
        "connector": KitConnector(type="SMA", sex="female"),
        "valid_span": KitValidSpan(1_000_000, 1_000_000_000),
    }
    kwargs.update(overrides)
    return CalibrationKit(standards=standards, **kwargs)


def _field_texts(kit: CalibrationKit) -> dict[str, str]:
    """The text the calibration window would show for ``kit``."""
    return {
        legacy_key: f"{value:.15g}"
        for legacy_key, value in legacy_values(kit).items()
    }


def _element_from_texts(texts: dict[str, str]) -> CalElement:
    """What ``calculate()`` builds from the displayed coefficient fields."""
    element = CalElement()
    for standard, key, legacy_key, divisor, _default in COEFFICIENT_FIELDS:
        setattr(element, f"{standard}_state", "COEFF")
        setattr(
            element,
            f"{standard}_{key}",
            float(texts[legacy_key]) / divisor,
        )
    element.through_is_ideal = False
    return element


def _gamma(element: CalElement) -> list[complex]:
    calibration = Calibration()
    calibration.cal_element = element
    values: list[complex] = []
    for frequency in FREQUENCIES:
        values.extend(
            (
                calibration.gamma_short(frequency),
                calibration.gamma_open(frequency),
                calibration.gamma_load(frequency),
                calibration.gamma_through(frequency),
            )
        )
    return values


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
class TestKitDocument:
    @staticmethod
    def test_document_round_trip_is_lossless():
        kit = _coefficient_kit()
        restored = CalibrationKit.from_dict(
            json.loads(json.dumps(kit.to_dict()))
        )
        assert restored == kit
        assert restored.calibration_identity() == kit.calibration_identity()

    @staticmethod
    def test_document_shape_is_documented_schema():
        document = _coefficient_kit().to_dict()
        assert document["format"] == KIT_FORMAT
        assert document["version"] == KIT_VERSION
        assert set(document["standards"]) == {
            "short",
            "open",
            "load",
            "through",
        }
        assert document["standards"]["short"]["coefficients"]["l0"] == 5.7e-12
        assert document["reference_impedance"] == 50.0
        assert document["valid_span"] == {
            "start_hz": 1_000_000,
            "stop_hz": 1_000_000_000,
        }
        # Written with style: stable key order and no NaN/Infinity tokens.
        assert json.dumps(document, sort_keys=True) == json.dumps(
            json.loads(json.dumps(document, sort_keys=True)), sort_keys=True
        )

    @staticmethod
    def test_identity_ignores_name_description_and_timestamps():
        kit = _coefficient_kit()
        twin = _coefficient_kit(name="Same numbers, other name")
        renamed = kit.renamed("Renamed kit")
        assert kit.calibration_identity() == twin.calibration_identity()
        assert kit.calibration_identity() == renamed.calibration_identity()
        assert (
            kit.calibration_identity()
            == _coefficient_kit().calibration_identity()
        )

    @staticmethod
    def test_identity_changes_with_electrical_content():
        kit = _coefficient_kit()
        other = CalibrationKit.from_dict(
            kit.to_dict() | {"reference_impedance": 75.0}
        )
        assert kit.calibration_identity() != other.calibration_identity()

    @staticmethod
    def test_identity_is_a_digest_not_the_content():
        identity = _coefficient_kit().calibration_identity()
        assert len(identity) == 32
        assert all(character in "0123456789abcdef" for character in identity)

    @staticmethod
    def test_builtins_are_frozen_and_named():
        assert [kit.name for kit in builtin_kits()] == list(BUILTIN_NAMES)
        with pytest.raises(FrozenInstanceError):
            IDEAL_KIT.name = "Changed"
        assert find_builtin(IDEAL_KIT.name) is IDEAL_KIT
        assert find_builtin("not a kit") is None

    @staticmethod
    def test_builtin_sma_kit_matches_historical_defaults():
        element = SMA_MALE_KIT.to_cal_element()
        assert element.short_state == "COEFF"
        assert element.short_l0 == 5.7e-12
        assert element.short_length == -34.2e-12
        assert element.open_c0 == 2.1e-14
        assert element.load_r == 50.0
        assert element.through_is_ideal is True

    @staticmethod
    def test_ideal_kit_maps_to_ideal_standards():
        element = IDEAL_KIT.to_cal_element()
        assert element.short_state == "IDEAL"
        assert element.open_state == "IDEAL"
        assert element.load_state == "IDEAL"
        assert element.through_is_ideal is True

    @staticmethod
    def test_unsupported_reference_impedance_is_refused():
        kit = CalibrationKit.from_dict(
            _coefficient_kit().to_dict() | {"reference_impedance": 75.0}
        )
        with pytest.raises(CalibrationKitError, match="75 ohm"):
            kit.to_cal_element()
        with pytest.raises(CalibrationKitError, match="75 ohm"):
            kit.apply_to(Calibration())


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestKitValidation:
    @staticmethod
    def _document(**changes) -> dict:
        document = _coefficient_kit().to_dict()
        document.update(changes)
        return document

    @staticmethod
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"format": "something-else"}, "unexpected kit format"),
            ({"version": KIT_VERSION + 1}, "newer than supported"),
            ({"version": 0}, "not a valid version"),
            ({"name": "   "}, "must not be empty"),
            ({"name": None}, "name must be a string"),
            ({"reference_impedance": -50.0}, "must be positive"),
            ({"reference_impedance": "fifty"}, "must be a number"),
            ({"reference_impedance": math.nan}, "must be finite"),
            ({"created": "yesterday"}, "not an ISO-8601 timestamp"),
            ({"unexpected": 1}, "unknown key"),
            (
                {"valid_span": {"start_hz": 0, "stop_hz": 10}},
                "must be positive",
            ),
            (
                {"valid_span": {"start_hz": 10, "stop_hz": 10}},
                "must be above start_hz",
            ),
            ({"valid_span": {"start_hz": 1.5, "stop_hz": 10}}, "integer"),
            ({"standards": {}}, "missing standard"),
        ],
    )
    def test_malformed_documents_are_rejected(changes, message):
        with pytest.raises(CalibrationKitValidationError, match=message):
            CalibrationKit.from_dict(TestKitValidation._document(**changes))

    @staticmethod
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"model": "guess"}, "unknown"),
            ({"coefficients": {"unknown": 1.0}}, "unknown coefficient"),
            (
                {"model": "coeff", "coefficients": {"l0": math.inf}},
                "must be finite",
            ),
            (
                {"model": "file", "file_ref": "x.s1p"},
                "cannot carry coefficients",
            ),
            ({"model": "coeff", "coefficients": {}}, "needs coefficients"),
        ],
    )
    def test_malformed_standards_are_rejected(changes, message):
        document = _coefficient_kit().to_dict()
        entry = {"model": "coeff", "coefficients": {"l0": 1e-12}}
        entry.update(changes)
        document["standards"]["short"] = entry
        with pytest.raises(CalibrationKitValidationError, match=message):
            CalibrationKit.from_dict(document)

    @staticmethod
    def test_through_cannot_use_a_file():
        document = _coefficient_kit().to_dict()
        document["standards"]["through"] = {
            "model": "file",
            "file_ref": "through.s2p",
        }
        with pytest.raises(
            CalibrationKitValidationError, match="cannot use a file"
        ):
            CalibrationKit.from_dict(document)

    @staticmethod
    def test_ideal_standard_cannot_carry_coefficients():
        with pytest.raises(
            CalibrationKitValidationError, match="ideal standard"
        ):
            KitStandard(model="ideal", coefficients={"l0": 1.0})

    @staticmethod
    def test_connector_and_span_reject_empty_and_inverted_values():
        with pytest.raises(CalibrationKitValidationError, match="connector"):
            KitConnector(type="   ")
        with pytest.raises(CalibrationKitValidationError, match="stop_hz"):
            KitValidSpan(start_hz=100, stop_hz=50)


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
class TestKitLibraryStorage:
    @staticmethod
    def test_round_trip_through_the_library_directory(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit())
        library.save()
        assert os.path.isfile(library.filename)
        reopened = KitLibrary(directory=str(tmp_path)).load()
        assert [kit.name for kit in reopened.user_kits()] == ["Test kit"]
        assert (
            reopened.get("Test kit").calibration_identity()
            == _coefficient_kit().calibration_identity()
        )

    @staticmethod
    def test_library_document_is_versioned(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit())
        library.save()
        with open(library.filename, encoding="utf-8") as source:
            document = json.load(source)
        assert document["format"] == LIBRARY_FORMAT
        assert document["version"] == LIBRARY_VERSION
        assert len(document["kits"]) == 1

    @staticmethod
    def test_missing_library_is_empty_not_an_error(tmp_path):
        library = KitLibrary(directory=str(tmp_path)).load()
        assert library.user_kits() == []
        assert [kit.name for kit in library.all_kits()] == list(BUILTIN_NAMES)

    @staticmethod
    def test_two_directories_do_not_share_kits(tmp_path):
        first = KitLibrary(directory=str(tmp_path / "one"))
        second = KitLibrary(directory=str(tmp_path / "two"))
        first.add(_coefficient_kit(name="Only in one"))
        first.save()
        second.save()
        assert (
            KitLibrary(directory=str(tmp_path / "two"))
            .load()
            .get("Only in one")
            is None
        )
        assert (
            KitLibrary(directory=str(tmp_path / "one"))
            .load()
            .get("Only in one")
            is not None
        )

    @staticmethod
    def test_malformed_library_leaves_memory_and_disk_untouched(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit())
        library.save()
        before_memory = [kit.name for kit in library.user_kits()]
        with open(library.filename, "rb") as source:
            before_bytes = source.read()
        with open(library.filename, "w", encoding="utf-8") as target:
            target.write('{"format": "nanovna-saver-calkit-library",')
        with pytest.raises(CalibrationKitValidationError):
            library.load()
        assert [kit.name for kit in library.user_kits()] == before_memory
        with open(library.filename, "rb") as source:
            assert source.read() != before_bytes

    @staticmethod
    def test_save_is_atomic_and_leaves_no_temporary_file(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit())
        library.save()
        library.save()
        assert os.listdir(tmp_path) == ["calkits.json"]

    @staticmethod
    def test_default_library_directory_is_lazy():
        library = KitLibrary()
        assert library._directory is None
        location = library.directory
        assert location
        assert library._directory == location
        assert default_library_directory() == location


# ----------------------------------------------------------------------
# Names, collisions and built-ins
# ----------------------------------------------------------------------
class TestKitLibraryNames:
    @staticmethod
    def test_collision_is_refused_then_allowed_with_replace(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit())
        with pytest.raises(CalibrationKitNameError, match="already exists"):
            library.add(_coefficient_kit())
        library.add(_coefficient_kit(description="Second"), replace=True)
        assert len(library.user_kits()) == 1
        assert library.get("Test kit").description == "Second"

    @staticmethod
    def test_builtin_names_cannot_be_stored_renamed_or_deleted(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        with pytest.raises(CalibrationKitNameError, match="built-in"):
            library.add(_coefficient_kit(name=BUILTIN_NAMES[0]))
        with pytest.raises(CalibrationKitNameError, match="built-in"):
            library.rename(BUILTIN_NAMES[0], "Free name")
        with pytest.raises(CalibrationKitNameError, match="built-in"):
            library.delete(BUILTIN_NAMES[0])
        assert find_builtin(BUILTIN_NAMES[0]) is not None

    @staticmethod
    def test_rename_and_duplicate_round_trip(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit())
        copy = library.duplicate("Test kit", "Copy")
        assert copy.name == "Copy"
        assert (
            copy.calibration_identity()
            == library.get("Test kit").calibration_identity()
        )
        renamed = library.rename("Copy", "Renamed")
        assert renamed.name == "Renamed"
        assert library.get("Copy") is None
        assert library.delete("Renamed") is True
        assert library.delete("Renamed") is False

    @staticmethod
    def test_stored_kit_shadowing_a_builtin_is_dropped_on_load(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.save()
        document = {
            "format": LIBRARY_FORMAT,
            "version": LIBRARY_VERSION,
            "kits": [_coefficient_kit(name=BUILTIN_NAMES[0]).to_dict()],
        }
        with open(library.filename, "w", encoding="utf-8") as target:
            json.dump(document, target)
        assert KitLibrary(directory=str(tmp_path)).load().user_kits() == []


# ----------------------------------------------------------------------
# Import and export
# ----------------------------------------------------------------------
class TestKitTransfer:
    @staticmethod
    def test_export_then_import_into_another_library(tmp_path):
        source = KitLibrary(directory=str(tmp_path / "source"))
        source.add(_coefficient_kit())
        exported = source.export_kit("Test kit", str(tmp_path / "kit.json"))
        assert os.path.isfile(exported)

        target = KitLibrary(directory=str(tmp_path / "target"))
        imported = target.import_kit(exported)
        assert imported.calibration_identity() == (
            source.get("Test kit").calibration_identity()
        )

    @staticmethod
    def test_import_collision_rejects_the_whole_file(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.add(_coefficient_kit(name="Existing"))
        library.save()
        with open(library.filename, "rb") as source:
            before = source.read()

        document = {
            "format": LIBRARY_FORMAT,
            "version": LIBRARY_VERSION,
            "kits": [
                _coefficient_kit(name="Fresh").to_dict(),
                _coefficient_kit(name="Existing").to_dict(),
            ],
        }
        filename = tmp_path / "incoming.json"
        filename.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CalibrationKitNameError, match="already exists"):
            library.import_file(str(filename))
        assert [kit.name for kit in library.user_kits()] == ["Existing"]
        with open(library.filename, "rb") as source:
            assert source.read() == before

    @staticmethod
    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ("not json at all", "not valid JSON"),
            ("[]", "must be a JSON object"),
            ('{"format": "other"}', "unexpected kit format"),
            (
                '{"format": "nanovna-saver-calkit", "version": 99}',
                "newer than supported",
            ),
            (
                '{"format": "nanovna-saver-calkit", "name": "x",'
                ' "standards": {}, "bogus": 1}',
                "unknown key",
            ),
            (
                '{"format": "nanovna-saver-calkit", "name": "x",'
                ' "reference_impedance": NaN, "standards": {}}',
                "must be finite",
            ),
        ],
    )
    def test_malformed_imports_are_rejected(payload, message, tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        filename = tmp_path / "bad.json"
        filename.write_text(payload, encoding="utf-8")
        with pytest.raises(CalibrationKitError, match=message):
            library.import_file(str(filename))
        assert library.user_kits() == []
        assert not os.path.exists(library.filename)

    @staticmethod
    def test_kit_and_library_documents_both_parse():
        kit = _coefficient_kit()
        assert (
            kits_from_document(kit.to_dict())[0].calibration_identity()
            == kit.calibration_identity()
        )
        library_document = {
            "format": LIBRARY_FORMAT,
            "version": LIBRARY_VERSION,
            "kits": [kit.to_dict()],
        }
        assert (
            kits_from_document(library_document)[0].calibration_identity()
            == kit.calibration_identity()
        )
        assert kit_from_document(kit.to_dict()).name == kit.name


# ----------------------------------------------------------------------
# Applying a kit must equal entering the numbers by hand
# ----------------------------------------------------------------------
class TestKitApplication:
    @staticmethod
    def test_apply_matches_hand_entered_fields():
        kit = _coefficient_kit()
        via_kit = kit.to_cal_element()
        via_fields = _element_from_texts(_field_texts(kit))
        for standard, key, _legacy, _divisor, _default in COEFFICIENT_FIELDS:
            assert getattr(via_kit, f"{standard}_{key}") == getattr(
                via_fields, f"{standard}_{key}"
            ), (standard, key)
        assert via_kit.short_state == via_fields.short_state == "COEFF"
        assert via_kit.through_is_ideal == via_fields.through_is_ideal
        assert _gamma(via_kit) == _gamma(via_fields)

    @staticmethod
    def test_apply_installs_the_element_and_records_provenance():
        calibration = Calibration()
        kit = _coefficient_kit()
        element = kit.apply_to(calibration)
        assert calibration.cal_element is element
        assert calibration.kit_name == kit.name
        assert calibration.kit_identity == kit.calibration_identity()

    @staticmethod
    def test_apply_never_touches_measured_data_or_the_calculation_flag():
        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        dataset = calibration.dataset
        calculated = calibration.isCalculated
        SMA_MALE_KIT.apply_to(calibration)
        assert calibration.dataset is dataset
        assert calibration.isCalculated is calculated
        assert calibration.kit_name == SMA_MALE_KIT.name

    @staticmethod
    def test_ideal_kit_matches_ideal_gammas():
        element = IDEAL_KIT.to_cal_element()
        assert _gamma(element) == [
            value
            for _ in FREQUENCIES
            for value in (-1 + 0j, 1 + 0j, 0j, 1 + 0j)
        ]

    @staticmethod
    def test_legacy_defaults_are_the_defaults_the_window_starts_with():
        legacy = kit_from_legacy_values(
            "Legacy defaults",
            {legacy_key: "" for _, _, legacy_key, _, _ in COEFFICIENT_FIELDS},
        )
        element = legacy.to_cal_element()
        for standard, key, _legacy, _divisor, default in COEFFICIENT_FIELDS:
            if default:
                continue
            assert getattr(element, f"{standard}_{key}") == 0.0, (
                standard,
                key,
            )
        # The two non-zero defaults are the ones the window ships with.
        assert element.open_c0 == 50.0 / 1.0e15
        assert element.load_r == 50.0
        assert element.through_is_ideal is False

    @staticmethod
    def test_unsupported_reference_impedance_never_silently_applies():
        kit = CalibrationKit.from_dict(
            _coefficient_kit().to_dict() | {"reference_impedance": 75.0}
        )
        calibration = Calibration()
        original = calibration.cal_element
        with pytest.raises(CalibrationKitError):
            kit.apply_to(calibration)
        assert calibration.cal_element is original
        assert calibration.kit_name == ""


# ----------------------------------------------------------------------
# Units and legacy settings
# ----------------------------------------------------------------------
class TestKitUnits:
    @staticmethod
    def test_display_and_storage_use_the_field_divisors():
        for standard, key, _legacy, divisor, _default in COEFFICIENT_FIELDS:
            assert stored_value(standard, key, 5.7) == 5.7 / divisor
            assert displayed_value(standard, key, 5.7e-12) == 5.7e-12 * divisor

    @staticmethod
    def test_legacy_values_are_the_values_the_window_shows():
        values = legacy_values(SMA_MALE_KIT)
        assert values["ShortL0"] == 5.7
        assert values["ShortL2"] == -1.1e-29 * 1.0e33
        assert values["ShortDelay"] == -34.2
        assert values["OpenC0"] == 21.0
        assert values["LoadR"] == 50.0

    @staticmethod
    def test_legacy_round_trip_keeps_the_same_si_values():
        kit = _coefficient_kit()
        values = legacy_values(kit)
        rebuilt = kit_from_legacy_values("Rebuilt", values)
        for standard, key, _legacy, _divisor, _default in COEFFICIENT_FIELDS:
            assert getattr(
                rebuilt.to_cal_element(), f"{standard}_{key}"
            ) == getattr(kit.to_cal_element(), f"{standard}_{key}"), (
                standard,
                key,
            )

    @staticmethod
    def test_legacy_text_defaults_match_the_window_defaults():
        kit = kit_from_legacy_values(
            "Defaults",
            {"ShortL0": "", "OpenC0": None, "LoadR": "50"},
        )
        assert kit.standards["short"].coefficients["l0"] == 0.0
        assert kit.standards["open"].coefficients["c0"] == 50.0 / 1.0e15
        assert kit.standards["load"].coefficients["r"] == 50.0
        assert kit.source == "legacy-settings"

    @staticmethod
    def test_legacy_non_numeric_value_is_reported_with_its_key():
        with pytest.raises(
            CalibrationKitValidationError, match="ShortL0.*not a number"
        ):
            kit_from_legacy_values("Broken", {"ShortL0": "five"})

    @staticmethod
    def test_transient_kits_are_listed_but_never_saved(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        legacy = kit_from_legacy_values("Old saved standard", {"LoadR": "49.9"})
        library.register_transient(legacy)
        library.add(_coefficient_kit())
        library.save()
        assert library.get("Old saved standard") is legacy
        assert legacy in library.all_kits()
        with open(library.filename, encoding="utf-8") as source:
            stored = json.load(source)
        assert [kit["name"] for kit in stored["kits"]] == ["Test kit"]

    @staticmethod
    def test_registering_the_same_legacy_name_twice_keeps_the_newest(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        library.register_transient(
            kit_from_legacy_values("Old", {"LoadR": "1"})
        )
        newest = kit_from_legacy_values("Old", {"LoadR": "2"})
        library.register_transient(newest)
        assert library.transient_kits() == [newest]
        library.clear_transient()
        assert library.transient_kits() == []


# ----------------------------------------------------------------------
# Window integration
# ----------------------------------------------------------------------
@pytest.fixture
def kit_window(qt_app, tmp_path, monkeypatch):
    """A calibration window whose settings and kit library are temporary."""
    from PySide6 import QtCore

    from NanoVNASaver.Windows import CalibrationSettings

    library_directory = tmp_path / "kits"
    monkeypatch.setattr(
        CalibrationSettings,
        "KitLibrary",
        lambda: KitLibrary(directory=str(library_directory)),
    )
    settings = QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.Format.IniFormat
    )

    class _SweepControl:
        class btn_stop:
            @staticmethod
            def isEnabled():
                return False

        @staticmethod
        def update_text():
            return None

    class _App:
        calibration = Calibration()
        sweep_control = _SweepControl()

        def __init__(self):
            self.errors = []
            self.settings = settings

        def showError(self, message):
            self.errors.append(message)

    window = CalibrationSettings.CalibrationWindow(_App())
    yield window
    window.deleteLater()


class TestCalibrationWindowKits:
    @staticmethod
    def test_selector_lists_builtin_kits(kit_window):
        names = [
            kit_window.cal_kit_selector.itemText(index)
            for index in range(kit_window.cal_kit_selector.count())
        ]
        assert names == list(BUILTIN_NAMES)

    @staticmethod
    def test_selector_shows_legacy_saved_standards(kit_window):
        settings = kit_window.app.settings
        settings.beginWriteArray("CalibrationStandards", 1)
        settings.setArrayIndex(0)
        settings.setValue("Name", "My saved standard")
        settings.setValue("LoadR", "49.5")
        settings.endArray()
        settings.sync()

        kit_window.reloadLegacyKits()
        names = [
            kit_window.cal_kit_selector.itemText(index)
            for index in range(kit_window.cal_kit_selector.count())
        ]
        assert "My saved standard" in names
        assert kit_window.kit_library.get("My saved standard") is not None
        # The legacy array is read, never rewritten.
        assert settings.beginReadArray("CalibrationStandards") == 1
        settings.setArrayIndex(0)
        assert settings.value("LoadR") == "49.5"
        settings.endArray()

    @staticmethod
    def test_applying_a_kit_sets_the_element_and_provenance(kit_window):
        kit = _coefficient_kit(name="Window kit")
        kit_window.kit_library.add(kit)
        kit_window.refreshKitSelector()
        kit_window.cal_kit_selector.setCurrentText("Window kit")

        kit_window.applyCalibrationKit()

        # calculate() refuses without measured data, so nothing is corrected;
        # the kit itself is installed and the fields show its numbers.
        assert kit_window.app.calibration.kit_name == "Window kit"
        assert (
            kit_window.app.calibration.kit_identity
            == kit.calibration_identity()
        )
        assert kit_window.short_l0_input.text() == "5.7"
        assert kit_window.load_resistance.text() == "50.2"
        assert kit_window.use_coefficients.isChecked()

    @staticmethod
    def test_applied_kit_keeps_exact_coefficients(kit_window):
        kit = _coefficient_kit(name="Window kit")
        kit_window.kit_library.add(kit)
        kit_window.refreshKitSelector()
        kit_window.cal_kit_selector.setCurrentText("Window kit")
        kit_window.applyCalibrationKit()

        pending = kit_window.pendingCalElement()
        assert pending is not None
        expected = kit.to_cal_element()
        for standard, key, _legacy, _divisor, _default in COEFFICIENT_FIELDS:
            assert getattr(pending, f"{standard}_{key}") == getattr(
                expected, f"{standard}_{key}"
            ), (standard, key)

    @staticmethod
    def test_pending_element_is_dropped_after_a_manual_edit(kit_window):
        kit = _coefficient_kit(name="Window kit")
        kit_window.kit_library.add(kit)
        kit_window.refreshKitSelector()
        kit_window.cal_kit_selector.setCurrentText("Window kit")
        kit_window.applyCalibrationKit()
        assert kit_window.pendingCalElement() is not None

        kit_window.short_l0_input.setText("6.1")
        assert kit_window.pendingCalElement() is None

    @staticmethod
    def test_selecting_a_different_kit_does_not_apply_it(kit_window):
        before = kit_window.app.calibration.kit_name
        index = kit_window.cal_kit_selector.findText(
            "SMA male (typical copper)"
        )
        kit_window.cal_kit_selector.setCurrentIndex(index)
        kit_window.kitSelected(index)
        assert kit_window.app.calibration.kit_name == before
        assert "SMA male" in kit_window.cal_kit_detail.text()

    @staticmethod
    def test_builtin_kits_cannot_be_renamed_or_deleted_from_the_window(
        kit_window,
    ):
        kit_window.cal_kit_selector.setCurrentText("Ideal standards")
        kit_window.kitSelected(0)
        assert not kit_window.kitRenameButton.isEnabled()
        assert not kit_window.kitDeleteButton.isEnabled()
        assert kit_window.kitApplyButton.isEnabled()

    @staticmethod
    def test_saving_a_kit_writes_the_library(kit_window, monkeypatch):
        from PySide6 import QtWidgets

        kit_window.cal_kit_selector.setCurrentText("SMA male (typical copper)")
        kit_window.kitSelected(0)
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getText",
            staticmethod(lambda *args, **kwargs: ("My new kit", True)),
        )
        kit_window.saveCalibrationKit()

        stored = kit_window.kit_library.get("My new kit")
        assert stored is not None
        assert os.path.isfile(kit_window.kit_library.filename)
        assert kit_window.cal_kit_selector.currentText() == "My new kit"


class TestKitProvenance:
    """A kit applied to a calibration must be named in the snapshot."""

    @staticmethod
    def test_the_kit_name_and_identity_reach_the_provenance():
        from NanoVNASaver.Measurement import snapshot_from_datapoints
        from NanoVNASaver.RFTools import Datapoint

        calibration = Calibration()
        kit = find_builtin("SMA male (typical copper)")
        kit.apply_to(calibration)
        snapshot = snapshot_from_datapoints(
            [Datapoint(14_000_000, 0.2, 0.1)],
            None,
            calibration=calibration,
        )
        assert snapshot.provenance.calibration.kit_name == kit.name
        assert (
            snapshot.provenance.calibration.kit_identity
            == kit.calibration_identity()
        )

    @staticmethod
    def test_a_hand_typed_calibration_names_no_kit():
        from NanoVNASaver.Measurement import snapshot_from_datapoints
        from NanoVNASaver.RFTools import Datapoint

        calibration = Calibration()
        snapshot = snapshot_from_datapoints(
            [Datapoint(14_000_000, 0.2, 0.1)],
            None,
            calibration=calibration,
        )
        assert snapshot.provenance.calibration.kit_name == ""
        assert snapshot.provenance.calibration.kit_identity == ""

    @staticmethod
    def test_an_older_snapshot_without_kit_fields_still_decodes():
        from NanoVNASaver.Measurement import CalibrationIdentity

        decoded = CalibrationIdentity.from_dict({"identity": "abc"})
        assert decoded.kit_name == ""
        assert decoded.kit_identity == ""


class TestAnUnreadableLibraryIsNotOverwritten:
    @staticmethod
    def _broken(tmp_path):
        library = KitLibrary(directory=str(tmp_path))
        Path(library.filename).write_text('{"format": "something else"}')
        with pytest.raises(CalibrationKitError):
            library.load()
        return library

    def test_saving_refuses_and_keeps_the_file(self, tmp_path):
        library = self._broken(tmp_path)
        before = Path(library.filename).read_text()
        with pytest.raises(CalibrationKitError, match="not overwriting"):
            library.save()
        assert Path(library.filename).read_text() == before

    def test_a_good_reload_allows_saving_again(self, tmp_path):
        library = self._broken(tmp_path)
        Path(library.filename).unlink()
        library.load()
        library.save()
        assert Path(library.filename).exists()
