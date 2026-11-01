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
"""Measurement presets (PRESET-02, PRESET-03).

A preset is a named, versioned bundle of settings:

* sweep configuration (span, points, segments, averaging)
* which charts are shown
* which analyses apply and their parameters
* scenario-specific extras such as cable velocity factor

Applying a preset never connects a device, starts a sweep, or otherwise
touches hardware.  It produces a validated :class:`PresetApplication` that
the UI applies as one coherent settings change and then lets the user start
the sweep explicitly.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from .Analysis.Results import AnalysisKind

PRESET_FORMAT = "nanovna-saver-preset"
PRESET_VERSION = 1
PRESET_SUFFIX = ".nvpreset"


class PresetError(Exception):
    """Base class for preset problems."""


class PresetValidationError(PresetError):
    """A preset is structurally valid but cannot be applied here."""


@dataclass(frozen=True)
class SweepPreset:
    """Sweep configuration portion of a preset."""

    start: int = 3_600_000
    end: int = 30_000_000
    points: int = 101
    segments: int = 1
    averages: int = 1
    truncates: int = 0
    mode: str = "SINGLE"
    logarithmic: bool = False

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "points": self.points,
            "segments": self.segments,
            "averages": self.averages,
            "truncates": self.truncates,
            "mode": self.mode,
            "logarithmic": self.logarithmic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepPreset":
        defaults = cls()
        return cls(
            start=int(data.get("start", defaults.start)),
            end=int(data.get("end", defaults.end)),
            points=int(data.get("points", defaults.points)),
            segments=int(data.get("segments", defaults.segments)),
            averages=int(data.get("averages", defaults.averages)),
            truncates=int(data.get("truncates", defaults.truncates)),
            mode=str(data.get("mode", defaults.mode)),
            logarithmic=bool(data.get("logarithmic", defaults.logarithmic)),
        )


@dataclass(frozen=True)
class Preset:
    """A complete, named measurement workflow."""

    name: str
    description: str = ""
    sweep: SweepPreset = field(default_factory=SweepPreset)
    charts: tuple[str, ...] = ()
    analyses: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    builtin: bool = False
    version: int = PRESET_VERSION

    def to_dict(self) -> dict:
        return {
            "format": PRESET_FORMAT,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "sweep": self.sweep.to_dict(),
            "charts": list(self.charts),
            "analyses": list(self.analyses),
            "parameters": dict(self.parameters),
            "builtin": self.builtin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Preset":
        if not isinstance(data, Mapping):
            raise PresetError("preset is not an object")
        if "format" in data and data["format"] != PRESET_FORMAT:
            raise PresetError(f"unexpected preset format {data['format']!r}")
        version = int(data.get("version", PRESET_VERSION))
        if version > PRESET_VERSION:
            raise PresetError(
                f"preset version {version} is newer than supported"
                f" version {PRESET_VERSION}"
            )
        name = str(data.get("name", "")).strip()
        if not name:
            raise PresetError("preset has no name")
        return cls(
            name=name,
            description=str(data.get("description", "")),
            sweep=SweepPreset.from_dict(data.get("sweep") or {}),
            charts=tuple(str(c) for c in data.get("charts") or ()),
            analyses=tuple(str(a) for a in data.get("analyses") or ()),
            parameters=dict(data.get("parameters") or {}),
            # Built-ins are never trusted from a file.
            builtin=False,
            version=version,
        )

    def renamed(self, name: str) -> "Preset":
        name = name.strip()
        if not name:
            raise PresetError("preset name must not be empty")
        return replace(self, name=name, builtin=False)


# ----------------------------------------------------------------------
# Built-in presets (PRESET-02)
# ----------------------------------------------------------------------
ANTENNA_PRESET = Preset(
    name="Antenna tuning",
    description=(
        "Single-port sweep tuned for finding resonance and the VSWR"
        " bandwidth of an antenna."
    ),
    sweep=SweepPreset(
        start=7_000_000,
        end=30_000_000,
        points=401,
        segments=1,
        averages=3,
    ),
    charts=("S11 VSWR", "S11 Smith Chart", "S11 Return Loss"),
    analyses=(AnalysisKind.RESONANCE.value, AnalysisKind.VSWR.value),
    parameters={
        "vswr_limit": 2.0,
        "vswr_bandwidth": 2.0,
        "marker_count": 3,
    },
    builtin=True,
)

FILTER_PRESET = Preset(
    name="Filter testing",
    description=(
        "Two-port sweep of gain, phase and group delay with cutoff"
        " measurement at the selected attenuation."
    ),
    sweep=SweepPreset(
        start=1_000_000,
        end=200_000_000,
        points=401,
        segments=1,
        averages=3,
    ),
    charts=("S21 Gain", "S21 Phase", "S21 Group Delay"),
    analyses=(
        AnalysisKind.INSERTION_LOSS.value,
        AnalysisKind.CUTOFF.value,
        AnalysisKind.GROUP_DELAY.value,
    ),
    parameters={"attenuation": 3.0, "marker_count": 2},
    builtin=True,
)

CABLE_PRESET = Preset(
    name="Cable testing",
    description=(
        "Reflection sweep for cable loss, impedance and TDR length"
        " estimation at a stated velocity factor."
    ),
    sweep=SweepPreset(
        start=1_000_000,
        end=100_000_000,
        points=401,
        segments=1,
        averages=3,
    ),
    charts=("S11 Return Loss", "S21 Gain", "TDR"),
    analyses=(AnalysisKind.CABLE.value, AnalysisKind.INSERTION_LOSS.value),
    parameters={
        "velocity_factor": 0.66,
        "cable_length": 10.0,
        "marker_count": 2,
    },
    builtin=True,
)

BUILTIN_PRESETS: tuple[Preset, ...] = (
    ANTENNA_PRESET,
    FILTER_PRESET,
    CABLE_PRESET,
)

BUILTIN_NAMES = tuple(preset.name for preset in BUILTIN_PRESETS)


def builtin_presets() -> list[Preset]:
    """Fresh copies, so callers cannot corrupt the built-ins."""
    return [preset.renamed(preset.name) for preset in BUILTIN_PRESETS]


def find_builtin(name: str) -> Preset | None:
    return next(
        (preset for preset in BUILTIN_PRESETS if preset.name == name), None
    )


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PresetApplication:
    """Result of validating a preset against device capabilities.

    The UI applies ``sweep`` and ``charts``/``analyses`` as one change and
    then waits for an explicit sweep start.
    """

    preset: Preset
    sweep: SweepPreset
    analysis_kinds: tuple[AnalysisKind, ...]
    warnings: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def requires_connection(self) -> bool:
        """A preset never connects by itself."""
        return False


def validate_sweep(
    sweep: SweepPreset,
    *,
    min_frequency: int = 1,
    max_frequency: int | None = None,
    valid_points: Iterable[int] = (),
    max_segments: int = 1024,
) -> list[str]:
    """Return human-readable problems, empty when the sweep is usable."""
    problems: list[str] = []
    if sweep.start < min_frequency:
        problems.append(
            f"start frequency {sweep.start} Hz is below the device minimum"
            f" {min_frequency} Hz"
        )
    if sweep.end <= sweep.start:
        problems.append("stop frequency must be greater than start")
    if max_frequency is not None and sweep.end > max_frequency:
        problems.append(
            f"stop frequency {sweep.end} Hz exceeds the device maximum"
            f" {max_frequency} Hz"
        )
    if sweep.points < 2:
        problems.append("point count must be at least 2")
    if valid_points and sweep.points not in set(valid_points):
        problems.append(
            f"point count {sweep.points} is not supported by the device"
            f" (choose from {sorted(valid_points)})"
        )
    if sweep.segments < 1:
        problems.append("segment count must be at least 1")
    if sweep.segments > max_segments:
        problems.append(f"segment count exceeds {max_segments}")
    if sweep.averages < 1:
        problems.append("average count must be at least 1")
    return problems


def apply_preset(
    preset: Preset,
    *,
    min_frequency: int = 1,
    max_frequency: int | None = None,
    valid_points: Iterable[int] = (),
) -> PresetApplication:
    """Validate ``preset`` and describe the resulting configuration.

    Raises :class:`PresetValidationError` when the preset cannot be applied,
    so the UI never silently applies a partially valid configuration.
    """
    problems = validate_sweep(
        preset.sweep,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        valid_points=valid_points,
    )
    if problems:
        raise PresetValidationError(
            f"preset {preset.name!r} cannot be applied: " + "; ".join(problems)
        )

    kinds: list[AnalysisKind] = []
    warnings: list[str] = []
    for name in preset.analyses:
        try:
            kinds.append(AnalysisKind(name))
        except ValueError:
            warnings.append(f"unknown analysis {name!r} ignored")

    parameters = dict(preset.parameters)
    if "velocity_factor" in parameters:
        factor = parameters["velocity_factor"]
        if not isinstance(factor, (int, float)) or not 0 < factor <= 1:
            raise PresetValidationError(
                "velocity factor must be a number in (0, 1]"
            )
    if "vswr_limit" in parameters:
        limit = parameters["vswr_limit"]
        if not isinstance(limit, (int, float)) or limit <= 1:
            raise PresetValidationError(
                "VSWR limit must be a number greater than 1"
            )

    return PresetApplication(
        preset=preset,
        sweep=preset.sweep,
        analysis_kinds=tuple(kinds),
        warnings=tuple(warnings),
        parameters=parameters,
    )


# ----------------------------------------------------------------------
# User preset storage (PRESET-03)
# ----------------------------------------------------------------------
@dataclass
class PresetStore:
    """User presets persisted as one versioned JSON document.

    Built-ins are never stored; they are always regenerated from code so a
    user cannot permanently lose or corrupt them.  A stored preset that
    shadows a built-in name is kept as a user preset and reported as a
    conflict instead of silently replacing the built-in.
    """

    filename: str
    presets: list[Preset] = field(default_factory=list)

    @property
    def builtins(self) -> list[Preset]:
        return builtin_presets()

    def all_presets(self) -> list[Preset]:
        return self.builtins + list(self.presets)

    def find(self, name: str) -> Preset | None:
        for preset in self.presets:
            if preset.name == name:
                return preset
        return find_builtin(name)

    def conflicts(self) -> list[str]:
        return [
            preset.name
            for preset in self.presets
            if preset.name in BUILTIN_NAMES
        ]

    def add(self, preset: Preset) -> Preset:
        """Add or replace a user preset by name."""
        if not preset.name.strip():
            raise PresetError("preset name must not be empty")
        stored = preset.renamed(preset.name)
        for index, existing in enumerate(self.presets):
            if existing.name == stored.name:
                self.presets[index] = stored
                return stored
        self.presets.append(stored)
        return stored

    def duplicate(self, name: str, new_name: str) -> Preset:
        source = self.find(name)
        if source is None:
            raise PresetError(f"no preset named {name!r}")
        if self.find(new_name) is not None:
            raise PresetError(f"a preset named {new_name!r} already exists")
        return self.add(source.renamed(new_name))

    def rename(self, name: str, new_name: str) -> Preset:
        preset = next(
            (item for item in self.presets if item.name == name), None
        )
        if preset is None:
            if find_builtin(name) is not None:
                raise PresetError(
                    f"{name!r} is a built-in preset and cannot be renamed;"
                    " duplicate it instead"
                )
            raise PresetError(f"no user preset named {name!r}")
        if self.find(new_name) is not None:
            raise PresetError(f"a preset named {new_name!r} already exists")
        renamed = preset.renamed(new_name)
        self.presets[self.presets.index(preset)] = renamed
        return renamed

    def delete(self, name: str) -> bool:
        for index, preset in enumerate(self.presets):
            if preset.name == name:
                del self.presets[index]
                return True
        if find_builtin(name) is not None:
            raise PresetError(
                f"{name!r} is a built-in preset and cannot be deleted"
            )
        return False

    # -- persistence ---------------------------------------------------
    def to_document(self) -> dict:
        return {
            "format": PRESET_FORMAT,
            "version": PRESET_VERSION,
            "presets": [preset.to_dict() for preset in self.presets],
        }

    def save(self) -> None:
        payload = json.dumps(
            self.to_document(), sort_keys=True, indent=2
        ).encode("utf-8")
        directory = os.path.dirname(os.path.abspath(self.filename)) or "."
        os.makedirs(directory, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=directory, prefix=".nvpreset-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.filename)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    @classmethod
    def load(cls, filename: str) -> "PresetStore":
        store = cls(filename=filename)
        if not os.path.exists(filename):
            return store
        try:
            with open(filename, encoding="utf-8") as source:
                document = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise PresetError(
                f"could not read presets from {filename}: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise PresetError("preset document is not an object")
        if document.get("format") != PRESET_FORMAT:
            raise PresetError(
                f"unexpected preset document format {document.get('format')!r}"
            )
        version = int(document.get("version", 0))
        if version > PRESET_VERSION:
            raise PresetError(
                f"preset document version {version} is newer than supported"
            )
        for entry in document.get("presets") or []:
            store.presets.append(Preset.from_dict(entry))
        return store

    # -- transfer ------------------------------------------------------
    def export_preset(self, name: str, filename: str) -> None:
        preset = self.find(name)
        if preset is None:
            raise PresetError(f"no preset named {name!r}")
        with open(filename, "w", encoding="utf-8") as target:
            json.dump(preset.to_dict(), target, sort_keys=True, indent=2)

    def import_preset(self, filename: str, *, new_name: str = "") -> Preset:
        try:
            with open(filename, encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise PresetError(f"could not read {filename}: {exc}") from exc
        preset = Preset.from_dict(data)
        if new_name:
            preset = preset.renamed(new_name)
        if find_builtin(preset.name) is not None and not new_name:
            raise PresetError(
                f"{preset.name!r} collides with a built-in preset;"
                " import it under a new name"
            )
        return self.add(preset)


def default_store_filename(config_dir: str | None = None) -> str:
    """Location of the user preset document."""
    if config_dir is None:
        config_dir = os.path.join(os.path.expanduser("~"), ".config")
        config_dir = os.path.join(config_dir, "NanoVNASaver")
    return os.path.join(config_dir, f"presets{PRESET_SUFFIX}")
