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
"""PRESET-02/PRESET-03: built-in presets, validation and user storage."""

import json

import pytest

from NanoVNASaver.Analysis.Results import AnalysisKind
from NanoVNASaver.Presets import (
    ANTENNA_PRESET,
    BUILTIN_NAMES,
    CABLE_PRESET,
    FILTER_PRESET,
    PRESET_FORMAT,
    PRESET_VERSION,
    Preset,
    PresetError,
    PresetStore,
    PresetValidationError,
    SweepPreset,
    apply_preset,
    builtin_presets,
    default_store_filename,
    find_builtin,
    validate_sweep,
)


class TestBuiltinPresets:
    @staticmethod
    def test_three_builtins_are_available():
        presets = builtin_presets()
        assert [preset.name for preset in presets] == list(BUILTIN_NAMES)

    @staticmethod
    def test_antenna_preset_configures_reflection_workflow():
        assert "S11 VSWR" in ANTENNA_PRESET.charts
        assert "S11 Smith Chart" in ANTENNA_PRESET.charts
        assert AnalysisKind.RESONANCE.value in ANTENNA_PRESET.analyses
        assert AnalysisKind.VSWR.value in ANTENNA_PRESET.analyses
        assert ANTENNA_PRESET.parameters["vswr_limit"] == 2.0
        assert ANTENNA_PRESET.sweep.averages >= 1

    @staticmethod
    def test_filter_preset_configures_transmission_workflow():
        assert "S21 Gain" in FILTER_PRESET.charts
        assert "S21 Group Delay" in FILTER_PRESET.charts
        assert AnalysisKind.CUTOFF.value in FILTER_PRESET.analyses
        assert FILTER_PRESET.parameters["attenuation"] == 3.0

    @staticmethod
    def test_cable_preset_configures_tdr_and_velocity_factor():
        assert "TDR" in CABLE_PRESET.charts
        assert AnalysisKind.CABLE.value in CABLE_PRESET.analyses
        assert CABLE_PRESET.parameters["velocity_factor"] == 0.66

    @staticmethod
    def test_builtins_are_immutable():
        """Presets are frozen, so callers cannot corrupt the built-ins."""
        from dataclasses import FrozenInstanceError

        presets = builtin_presets()
        with pytest.raises(FrozenInstanceError):
            presets[0].name = "Mutated"
        assert builtin_presets()[0].name == "Antenna tuning"

    @staticmethod
    def test_find_builtin():
        assert find_builtin("Cable testing") is CABLE_PRESET
        assert find_builtin("nope") is None


class TestValidation:
    @staticmethod
    def test_valid_sweep_has_no_problems():
        assert (
            validate_sweep(SweepPreset(start=1e6, end=30e6, points=101)) == []
        )

    @staticmethod
    def test_device_maximum_is_enforced():
        problems = validate_sweep(
            SweepPreset(start=7e6, end=30e6, points=101),
            max_frequency=20e6,
        )
        assert any("exceeds the device maximum" in item for item in problems)

    @staticmethod
    def test_device_minimum_is_enforced():
        problems = validate_sweep(
            SweepPreset(start=100, end=30e6, points=101), min_frequency=50_000
        )
        assert any("below the device minimum" in item for item in problems)

    @staticmethod
    def test_unsupported_point_count_is_reported():
        problems = validate_sweep(
            SweepPreset(start=1e6, end=30e6, points=150),
            valid_points=(101, 401),
        )
        assert any("not supported by the device" in item for item in problems)

    @staticmethod
    def test_inverted_span_is_rejected():
        problems = validate_sweep(SweepPreset(start=30e6, end=1e6))
        assert any("stop frequency" in item for item in problems)


class TestApplyPreset:
    @staticmethod
    def test_apply_returns_one_coherent_configuration():
        application = apply_preset(
            FILTER_PRESET, max_frequency=6.3e9, valid_points=(101, 401)
        )
        assert application.preset is FILTER_PRESET
        assert application.sweep == FILTER_PRESET.sweep
        assert AnalysisKind.CUTOFF in application.analysis_kinds
        assert application.warnings == ()

    @staticmethod
    def test_applying_never_requires_a_connection():
        application = apply_preset(ANTENNA_PRESET, max_frequency=6.3e9)
        assert application.requires_connection is False

    @staticmethod
    def test_unsupported_device_is_rejected_not_partially_applied():
        with pytest.raises(PresetValidationError, match="cannot be applied"):
            apply_preset(
                Preset(
                    name="Too wide",
                    sweep=SweepPreset(start=1e6, end=10e9, points=101),
                ),
                max_frequency=300e6,
            )

    @staticmethod
    def test_unknown_analysis_is_a_warning_not_a_crash():
        application = apply_preset(
            Preset(name="Odd", analyses=("resonance", "nonsense")),
            max_frequency=6.3e9,
        )
        assert application.analysis_kinds == (AnalysisKind.RESONANCE,)
        assert application.warnings

    @staticmethod
    def test_invalid_vswr_limit_is_rejected():
        with pytest.raises(PresetValidationError, match="VSWR limit"):
            apply_preset(
                Preset(name="Bad", parameters={"vswr_limit": 1.0}),
                max_frequency=6.3e9,
            )

    @staticmethod
    def test_invalid_velocity_factor_is_rejected():
        with pytest.raises(PresetValidationError, match="velocity factor"):
            apply_preset(
                Preset(name="Bad", parameters={"velocity_factor": 1.5}),
                max_frequency=6.3e9,
            )


class TestPresetSerialization:
    @staticmethod
    def test_round_trip():
        preset = Preset(
            name="Mine",
            description="custom",
            sweep=SweepPreset(start=1e6, end=10e6, points=101, segments=2),
            charts=("S11 VSWR",),
            analyses=("vswr",),
            parameters={"vswr_limit": 1.8},
        )
        restored = Preset.from_dict(preset.to_dict())
        assert restored == preset

    @staticmethod
    def test_builtin_flag_is_never_trusted_from_a_file():
        payload = ANTENNA_PRESET.to_dict()
        assert payload["builtin"] is True
        assert Preset.from_dict(payload).builtin is False

    @staticmethod
    def test_newer_version_is_rejected():
        payload = {"name": "x", "version": PRESET_VERSION + 1}
        with pytest.raises(PresetError, match="newer than supported"):
            Preset.from_dict(payload)

    @staticmethod
    def test_missing_name_is_rejected():
        with pytest.raises(PresetError, match="no name"):
            Preset.from_dict({"format": PRESET_FORMAT})

    @staticmethod
    def test_wrong_format_is_rejected():
        with pytest.raises(PresetError, match="format"):
            Preset.from_dict({"format": "other", "name": "x"})


class TestPresetStore:
    @staticmethod
    def test_builtins_appear_without_being_stored(tmp_path):
        store = PresetStore(str(tmp_path / "presets.nvpreset"))
        assert len(store.presets) == 0
        assert [preset.name for preset in store.all_presets()] == list(
            BUILTIN_NAMES
        )

    @staticmethod
    def test_add_save_and_reload_custom_preset(tmp_path):
        filename = str(tmp_path / "presets.nvpreset")
        store = PresetStore(filename)
        store.add(Preset(name="Mine", sweep=SweepPreset(start=1e6, end=2e6)))
        store.save()
        reloaded = PresetStore.load(filename)
        assert [preset.name for preset in reloaded.presets] == ["Mine"]
        assert reloaded.find("Mine").sweep.end == 2e6

    @staticmethod
    def test_builtins_survive_store_corruption(tmp_path):
        filename = str(tmp_path / "presets.nvpreset")
        store = PresetStore(filename)
        store.add(Preset(name="Mine"))
        store.save()
        (tmp_path / "presets.nvpreset").write_text("{broken", encoding="utf-8")
        with pytest.raises(PresetError, match="could not read"):
            PresetStore.load(filename)
        # Built-ins do not depend on the file at all.
        assert builtin_presets()

    @staticmethod
    def test_duplicate_and_rename(tmp_path):
        store = PresetStore(str(tmp_path / "p.nvpreset"))
        store.add(Preset(name="Original"))
        copy = store.duplicate("Original", "Copy")
        assert copy.name == "Copy"
        store.rename("Copy", "Renamed")
        assert store.find("Renamed") is not None
        assert store.find("Copy") is None

    @staticmethod
    def test_builtins_cannot_be_renamed_or_deleted(tmp_path):
        store = PresetStore(str(tmp_path / "p.nvpreset"))
        with pytest.raises(PresetError, match="built-in"):
            store.delete("Antenna tuning")
        with pytest.raises(PresetError, match="built-in"):
            store.rename("Antenna tuning", "Mine")

    @staticmethod
    def test_duplicate_name_is_rejected(tmp_path):
        store = PresetStore(str(tmp_path / "p.nvpreset"))
        store.add(Preset(name="One"))
        with pytest.raises(PresetError, match="already exists"):
            store.duplicate("One", "One")

    @staticmethod
    def test_shadowing_a_builtin_is_reported_as_a_conflict(tmp_path):
        store = PresetStore(str(tmp_path / "p.nvpreset"))
        store.add(Preset(name="Antenna tuning", description="user override"))
        assert store.conflicts() == ["Antenna tuning"]

    @staticmethod
    def test_export_and_import_transfers_independently(tmp_path):
        store = PresetStore(str(tmp_path / "p.nvpreset"))
        store.add(Preset(name="Portable", description="travels"))
        exported = str(tmp_path / "portable.nvpreset")
        store.export_preset("Portable", exported)

        other = PresetStore(str(tmp_path / "other.nvpreset"))
        imported = other.import_preset(exported)
        assert imported.name == "Portable"
        assert imported.description == "travels"
        other.save()
        assert PresetStore.load(other.filename).find("Portable")

    @staticmethod
    def test_import_under_a_new_name_avoids_builtin_collision(tmp_path):
        exported = str(tmp_path / "antenna.nvpreset")
        with open(exported, "w", encoding="utf-8") as target:
            json.dump(ANTENNA_PRESET.to_dict(), target)

        store = PresetStore(str(tmp_path / "p.nvpreset"))
        with pytest.raises(PresetError, match="collides"):
            store.import_preset(exported)
        imported = store.import_preset(exported, new_name="My antenna")
        assert imported.name == "My antenna"

    @staticmethod
    def test_missing_file_is_an_empty_store(tmp_path):
        store = PresetStore.load(str(tmp_path / "absent.nvpreset"))
        assert store.presets == []

    @staticmethod
    def test_failed_save_leaves_no_temp_file(tmp_path, monkeypatch):
        filename = str(tmp_path / "p.nvpreset")
        store = PresetStore(filename)
        store.add(Preset(name="Mine"))

        import NanoVNASaver.Presets as presets_module

        def explode(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr(presets_module.os, "replace", explode)
        with pytest.raises(OSError, match="disk full"):
            store.save()
        monkeypatch.undo()
        assert not [
            name for name in tmp_path.iterdir() if name.suffix == ".tmp"
        ]

    @staticmethod
    def test_default_store_filename_lives_under_config_dir():
        path = default_store_filename("/tmp/cfg")
        assert path.endswith(".nvpreset")
        assert path.startswith("/tmp/cfg")
