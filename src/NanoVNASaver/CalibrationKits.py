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
"""Versioned calibration kit library (CAL-01).

What a kit describes
--------------------
A calibration kit is the *physical* meaning of the short, open, load and
through standards: the coefficients that turn "short attached" into a
reflection coefficient at each frequency.  The correction equations in
:mod:`NanoVNASaver.Calibration` are not touched by this module; a kit only
supplies the numbers that the existing equations consume through
:class:`~NanoVNASaver.Calibration.CalElement`.

The four standards use exactly the models already implemented by
``Calibration.gamma_short/open/load/through``:

``ideal``
    The textbook values (short ``-1``, open ``+1``, load ``0``, through
    ``+1``).
``coeff``
    The polynomial/offset-delay model of the existing code.  The keys are
    ``l0``..``l3`` (short inductance), ``c0``..``c3`` (open capacitance),
    ``r``/``l``/``c`` (load resistance/inductance/capacitance) and
    ``length`` (offset delay).
``file``
    A measured Touchstone ``.s1p`` file, resolved relative to the kit file.
    A through standard cannot use ``file``: the existing through model is
    either ideal or an offset delay.

Units (stored values, not displayed values)
-------------------------------------------
Everything is stored in SI base units, exactly as ``CalElement`` holds it:

===============  =========  ==========================================
key              unit       GUI shows
===============  =========  ==========================================
``l0``..``l3``   H          "L0 (H(e-12))" = nH text / 1e12
``c0``..``c3``   F          "C0 (F(e-15))" = pF text / 1e15
``r``            ohm        ohm
``l``            H          nH text / 1e12
``c``            F          pF text / 1e15
``length``       s          ps text / 1e12
===============  =========  ==========================================

``reference_impedance`` is metadata: the correction equations in
``Calibration`` are written for 50 ohm, so a kit that declares another
reference impedance can be stored, validated and exported but is refused by
:meth:`CalibrationKit.apply_to` instead of being silently applied with the
wrong assumption.  ``valid_span`` likewise records where the kit's
coefficients were characterised; it never restricts measurement.

Document format
---------------
A kit is one JSON object (:data:`KIT_FORMAT`, :data:`KIT_VERSION`)::

    {
      "format": "nanovna-saver-calkit",
      "version": 1,
      "name": "SMA male (typical copper)",
      "description": "Factory defaults of this application",
      "connector": {"type": "SMA", "sex": "male"},
      "reference_impedance": 50.0,
      "valid_span": {"start_hz": 1000000, "stop_hz": 6000000000},
      "standards": {
        "short": {"model": "coeff",
                  "coefficients": {"l0": 5.7e-12, "l1": -8.96e-20,
                                   "l2": -1.1e-29, "l3": -4.12e-37,
                                   "length": -3.42e-11}},
        "open":  {"model": "coeff",
                  "coefficients": {"c0": 2.1e-14, "c1": 5.67e-23,
                                   "c2": -2.39e-31, "c3": 2.0e-40,
                                   "length": 0.0}},
        "load":  {"model": "coeff",
                  "coefficients": {"r": 50.0, "l": 0.0, "c": 0.0,
                                   "length": 0.0}},
        "through": {"model": "ideal"}
      },
      "created": "2026-11-01T00:00:00Z",
      "source": "builtin"
    }

The user library is one document holding a list of such kits
(:data:`LIBRARY_FORMAT`) and is written atomically (temporary file plus
``os.replace``).  Built-in kits are never stored: they are always rebuilt
from code, so a user can delete a copy but never corrupt the originals.

A kit *name* is a label.  Only :meth:`CalibrationKit.calibration_identity`
tells two geometrically different standards apart, and neither proves that
the numbers describe the hardware in hand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .Calibration import CalElement
from .Touchstone import Touchstone

logger = logging.getLogger(__name__)

#: Format tag of a single kit document.
KIT_FORMAT = "nanovna-saver-calkit"
#: Schema version written by this build.  Newer documents are refused.
KIT_VERSION = 1
#: Format tag of the user library document.
LIBRARY_FORMAT = "nanovna-saver-calkit-library"
LIBRARY_VERSION = 1
#: File name of the user library inside its directory.
LIBRARY_FILENAME = "calkits.json"

MODEL_IDEAL = "ideal"
MODEL_COEFF = "coeff"
MODEL_FILE = "file"
#: Model names understood by this schema version.
MODELS = (MODEL_IDEAL, MODEL_COEFF, MODEL_FILE)

STANDARD_NAMES = ("short", "open", "load", "through")

#: Every coefficient of the coefficient model, as
#: ``(standard, key, legacy QSettings key, display divisor, default)``.
#:
#: ``key`` is both the kit JSON key and the ``CalElement`` field suffix, so
#: ``("short", "l0", ...)`` is ``KitStandard.coefficients["l0"]`` and
#: ``CalElement.short_l0``.  The divisor is the exact one the calibration
#: window uses (``entered_text / divisor = CalElement value``), so a value
#: that round-trips through the UI text and back is bit-identical.  The
#: default is what an absent legacy key falls back to.  See the module
#: docstring for the unit table.
COEFFICIENT_FIELDS: tuple[tuple[str, str, str, float, float], ...] = (
    ("short", "l0", "ShortL0", 1.0e12, 0.0),
    ("short", "l1", "ShortL1", 1.0e24, 0.0),
    ("short", "l2", "ShortL2", 1.0e33, 0.0),
    ("short", "l3", "ShortL3", 1.0e42, 0.0),
    ("short", "length", "ShortDelay", 1.0e12, 0.0),
    ("open", "c0", "OpenC0", 1.0e15, 50.0),
    ("open", "c1", "OpenC1", 1.0e27, 0.0),
    ("open", "c2", "OpenC2", 1.0e36, 0.0),
    ("open", "c3", "OpenC3", 1.0e45, 0.0),
    ("open", "length", "OpenDelay", 1.0e12, 0.0),
    ("load", "r", "LoadR", 1.0, 50.0),
    ("load", "l", "LoadL", 1.0e12, 0.0),
    ("load", "c", "LoadC", 1.0e15, 0.0),
    ("load", "length", "LoadDelay", 1.0e12, 0.0),
    ("through", "length", "ThroughDelay", 1.0e12, 0.0),
)

#: Coefficients allowed per standard, in SI units (see module docstring).
STANDARD_COEFFICIENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        standard: tuple(
            key
            for standard_, key, _, _, _ in COEFFICIENT_FIELDS
            if standard_ == standard
        )
        for standard in STANDARD_NAMES
    }
)

#: ``(standard, key)`` -> ``(legacy key, display divisor, default)``.
COEFFICIENT_METADATA: Mapping[tuple[str, str], tuple[str, float, float]] = (
    MappingProxyType(
        {
            (standard, key): (legacy_key, divisor, default)
            for standard, key, legacy_key, divisor, default in (
                COEFFICIENT_FIELDS
            )
        }
    )
)

#: Reference impedance the correction equations are written for.
SUPPORTED_REFERENCE_IMPEDANCE = 50.0

#: Legacy ``QSettings`` key -> ``(standard, key, divisor)``.
LEGACY_UNITS: Mapping[str, tuple[str, str, float]] = MappingProxyType(
    {
        legacy_key: (standard, key, divisor)
        for standard, key, legacy_key, divisor, _ in COEFFICIENT_FIELDS
    }
)

#: Legacy ``QSettings`` key -> default used when the key is absent.
LEGACY_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {legacy_key: default for _, _, legacy_key, _, default in COEFFICIENT_FIELDS}
)

KIT_KEYS = frozenset(
    {
        "format",
        "version",
        "name",
        "description",
        "connector",
        "reference_impedance",
        "valid_span",
        "standards",
        "created",
        "source",
    }
)
CONNECTOR_KEYS = frozenset({"type", "sex"})
SPAN_KEYS = frozenset({"start_hz", "stop_hz"})
STANDARD_KEYS = frozenset({"model", "coefficients", "file_ref"})


class CalibrationKitError(ValueError):
    """A kit cannot be read, stored or applied."""


class CalibrationKitValidationError(CalibrationKitError):
    """A kit document is malformed; the message names the offending part."""


class CalibrationKitNameError(CalibrationKitError):
    """A name is taken by a built-in kit or by another stored kit."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mapping(value: Any, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise CalibrationKitValidationError(
            f"{where} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _check_keys(data: Mapping, allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CalibrationKitValidationError(
            f"{where} has unknown key(s): {', '.join(map(repr, unknown))}"
        )


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise CalibrationKitValidationError(
            f"{where} must be a string, got {type(value).__name__}"
        )
    return value


def _number(value: Any, where: str) -> float:
    """Return ``value`` as a finite float, rejecting bools and strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationKitValidationError(
            f"{where} must be a number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationKitValidationError(f"{where} must be finite")
    return result


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationKitValidationError(
            f"{where} must be an integer, got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class KitConnector:
    """Mechanical interface a kit's standards are built for."""

    type: str = "unknown"
    sex: str = ""

    def __post_init__(self) -> None:
        if not self.type.strip():
            raise CalibrationKitValidationError(
                "connector type must not be empty"
            )
        object.__setattr__(self, "type", self.type.strip())
        object.__setattr__(self, "sex", self.sex.strip())

    def to_dict(self) -> dict:
        return {"type": self.type, "sex": self.sex}

    @classmethod
    def from_dict(cls, data: Any, where: str = "connector") -> KitConnector:
        source = _mapping(data, where)
        _check_keys(source, CONNECTOR_KEYS, where)
        return cls(
            type=_text(source.get("type", "unknown"), f"{where}.type"),
            sex=_text(source.get("sex", ""), f"{where}.sex"),
        )

    def describe(self) -> str:
        if self.sex:
            return f"{self.type} {self.sex}"
        return self.type


@dataclass(frozen=True)
class KitValidSpan:
    """Frequency range over which the kit's coefficients were characterised."""

    start_hz: int
    stop_hz: int

    def __post_init__(self) -> None:
        if self.start_hz <= 0:
            raise CalibrationKitValidationError(
                f"valid_span.start_hz must be positive, got {self.start_hz}"
            )
        if self.stop_hz <= self.start_hz:
            raise CalibrationKitValidationError(
                "valid_span.stop_hz must be above start_hz"
                f" ({self.start_hz} - {self.stop_hz} Hz)"
            )

    def to_dict(self) -> dict:
        return {"start_hz": self.start_hz, "stop_hz": self.stop_hz}

    @classmethod
    def from_dict(cls, data: Any, where: str = "valid_span") -> KitValidSpan:
        source = _mapping(data, where)
        _check_keys(source, SPAN_KEYS, where)
        for key in ("start_hz", "stop_hz"):
            if key not in source:
                raise CalibrationKitValidationError(
                    f"{where}.{key} is required"
                )
        return cls(
            start_hz=_integer(source["start_hz"], f"{where}.start_hz"),
            stop_hz=_integer(source["stop_hz"], f"{where}.stop_hz"),
        )

    def describe(self) -> str:
        return f"{self.start_hz} - {self.stop_hz} Hz"


@dataclass(frozen=True)
class KitStandard:
    """One standard of a kit: ideal, polynomial/offset model, or file."""

    model: str = MODEL_IDEAL
    coefficients: Mapping[str, float] = field(default_factory=dict)
    file_ref: str = ""

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise CalibrationKitValidationError(
                f"unknown standard model {self.model!r};"
                f" expected one of {', '.join(MODELS)}"
            )
        coefficients = {
            str(key): _number(value, f"coefficient {key!r}")
            for key, value in self.coefficients.items()
        }
        if self.model == MODEL_IDEAL:
            if coefficients or self.file_ref:
                raise CalibrationKitValidationError(
                    "an ideal standard cannot carry coefficients or a file"
                )
        elif self.model == MODEL_COEFF:
            if not coefficients:
                raise CalibrationKitValidationError(
                    "a coeff standard needs coefficients"
                )
            if self.file_ref:
                raise CalibrationKitValidationError(
                    "a coeff standard cannot reference a file"
                )
        else:  # MODEL_FILE
            if coefficients:
                raise CalibrationKitValidationError(
                    "a file standard cannot carry coefficients"
                )
            if not self.file_ref.strip():
                raise CalibrationKitValidationError(
                    "a file standard needs a file_ref"
                )
        object.__setattr__(self, "coefficients", MappingProxyType(coefficients))

    def validate_for(self, standard: str) -> None:
        """Reject coefficients that do not belong to ``standard``."""
        allowed = STANDARD_COEFFICIENTS[standard]
        unknown = sorted(set(self.coefficients) - set(allowed))
        if unknown:
            raise CalibrationKitValidationError(
                f"{standard} standard has unknown coefficient(s):"
                f" {', '.join(map(repr, unknown))}"
            )

    def to_dict(self) -> dict:
        if self.model == MODEL_IDEAL:
            return {"model": MODEL_IDEAL}
        if self.model == MODEL_FILE:
            return {"model": MODEL_FILE, "file_ref": self.file_ref}
        return {
            "model": MODEL_COEFF,
            "coefficients": dict(self.coefficients),
        }

    @classmethod
    def from_dict(cls, data: Any, where: str = "standard") -> KitStandard:
        source = _mapping(data, where)
        _check_keys(source, STANDARD_KEYS, where)
        if "model" not in source:
            raise CalibrationKitValidationError(f"{where}.model is required")
        model = _text(source["model"], f"{where}.model")
        if model not in MODELS:
            raise CalibrationKitValidationError(
                f"{where}.model {model!r} is unknown;"
                f" expected one of {', '.join(MODELS)}"
            )
        coefficients: dict[str, float] = {}
        if "coefficients" in source:
            raw = _mapping(source["coefficients"], f"{where}.coefficients")
            coefficients = {
                str(key): _number(value, f"{where}.coefficients.{key}")
                for key, value in raw.items()
            }
        return cls(
            model=model,
            coefficients=coefficients,
            file_ref=_text(source.get("file_ref", ""), f"{where}.file_ref"),
        )


@dataclass(frozen=True)
class CalibrationKit:
    """An immutable, versioned description of a set of standards."""

    name: str
    description: str = ""
    connector: KitConnector = field(default_factory=KitConnector)
    reference_impedance: float = SUPPORTED_REFERENCE_IMPEDANCE
    valid_span: KitValidSpan | None = None
    standards: Mapping[str, KitStandard] = field(default_factory=dict)
    created: str = ""
    source: str = ""
    version: int = KIT_VERSION

    def __post_init__(self) -> None:
        name = _text(self.name, "name").strip()
        if not name:
            raise CalibrationKitValidationError("kit name must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "description", _text(self.description, "description")
        )
        object.__setattr__(self, "created", _text(self.created, "created"))
        object.__setattr__(self, "source", _text(self.source, "source"))
        impedance = _number(self.reference_impedance, "reference_impedance")
        if impedance <= 0:
            raise CalibrationKitValidationError(
                f"reference_impedance must be positive, got {impedance:g} ohm"
            )
        object.__setattr__(self, "reference_impedance", impedance)
        if self.version > KIT_VERSION:
            raise CalibrationKitValidationError(
                f"kit version {self.version} is newer than supported"
                f" ({KIT_VERSION})"
            )
        if self.version < 1:
            raise CalibrationKitValidationError(
                f"kit version {self.version} is not a valid version"
            )
        if self.created:
            try:
                datetime.fromisoformat(self.created.replace("Z", "+00:00"))
            except ValueError as exc:
                raise CalibrationKitValidationError(
                    f"created {self.created!r} is not an ISO-8601 timestamp"
                ) from exc
        standards = dict(self.standards)
        missing = [name_ for name_ in STANDARD_NAMES if name_ not in standards]
        if missing:
            raise CalibrationKitValidationError(
                f"kit is missing standard(s): {', '.join(missing)}"
            )
        unknown = sorted(set(standards) - set(STANDARD_NAMES))
        if unknown:
            raise CalibrationKitValidationError(
                f"kit has unknown standard(s): {', '.join(map(repr, unknown))}"
            )
        frozen: dict[str, KitStandard] = {}
        for standard_name in STANDARD_NAMES:
            standard = standards[standard_name]
            if not isinstance(standard, KitStandard):
                raise CalibrationKitValidationError(
                    f"standard {standard_name!r} must be a KitStandard"
                )
            standard.validate_for(standard_name)
            if standard_name == "through" and standard.model == MODEL_FILE:
                raise CalibrationKitValidationError(
                    "the through standard cannot use a file; it is either"
                    " ideal or an offset delay"
                )
            frozen[standard_name] = standard
        object.__setattr__(self, "standards", MappingProxyType(frozen))

    # -- identity ------------------------------------------------------
    def calibration_identity(self) -> str:
        """Digest of the electrically relevant content of the kit.

        The name, description, creation time, source and schema version are
        deliberately excluded: two kits with the same standards at the same
        reference plane describe the same correction even if one is called
        "shop kit" and the other "spare kit".  They are excluded the other
        way round too, so renaming a kit never changes its identity.
        """
        payload = json.dumps(
            {
                "reference_impedance": self.reference_impedance,
                "connector": self.connector.to_dict(),
                "valid_span": (
                    self.valid_span.to_dict() if self.valid_span else None
                ),
                "standards": {
                    name: self.standards[name].to_dict()
                    for name in STANDARD_NAMES
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def describe(self) -> str:
        span = self.valid_span.describe() if self.valid_span else "unspecified"
        return (
            f"{self.name}: {self.connector.describe()},"
            f" Z0 {self.reference_impedance:g} ohm, {span},"
            f" identity {self.calibration_identity()[:12]}"
        )

    def renamed(self, new_name: str) -> CalibrationKit:
        return replace(self, name=new_name)

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": KIT_FORMAT,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "connector": self.connector.to_dict(),
            "reference_impedance": self.reference_impedance,
            "valid_span": self.valid_span.to_dict()
            if self.valid_span
            else None,
            "standards": {
                name: self.standards[name].to_dict() for name in STANDARD_NAMES
            },
            "created": self.created,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CalibrationKit:
        """Validate and build a kit from a parsed JSON object."""
        source = _mapping(data, "kit")
        _check_keys(source, KIT_KEYS, "kit")
        if source.get("format") != KIT_FORMAT:
            raise CalibrationKitValidationError(
                f"unexpected kit format {source.get('format')!r};"
                f" expected {KIT_FORMAT!r}"
            )
        version = _integer(source.get("version", KIT_VERSION), "kit.version")
        if version > KIT_VERSION:
            raise CalibrationKitValidationError(
                f"kit version {version} is newer than supported ({KIT_VERSION})"
            )
        if "name" not in source:
            raise CalibrationKitValidationError("kit.name is required")
        if "standards" not in source:
            raise CalibrationKitValidationError("kit.standards is required")
        raw_standards = _mapping(source["standards"], "standards")
        standards = {
            str(name): KitStandard.from_dict(value, f"standards.{name}")
            for name, value in raw_standards.items()
        }
        valid_span = source.get("valid_span")
        return cls(
            name=_text(source["name"], "kit.name"),
            description=_text(source.get("description", ""), "kit.description"),
            connector=KitConnector.from_dict(
                source.get("connector", {"type": "unknown"})
            ),
            reference_impedance=_number(
                source.get(
                    "reference_impedance", SUPPORTED_REFERENCE_IMPEDANCE
                ),
                "kit.reference_impedance",
            ),
            valid_span=(
                KitValidSpan.from_dict(valid_span) if valid_span else None
            ),
            standards=standards,
            created=_text(source.get("created", ""), "kit.created"),
            source=_text(source.get("source", ""), "kit.source"),
            version=version,
        )

    # -- conversion to and from the existing calibration model ---------
    def _require_supported_reference(self) -> None:
        if abs(self.reference_impedance - SUPPORTED_REFERENCE_IMPEDANCE) > 1e-9:
            raise CalibrationKitError(
                f"kit {self.name!r} is defined for"
                f" {self.reference_impedance:g} ohm, but the calibration"
                f" equations assume {SUPPORTED_REFERENCE_IMPEDANCE:g} ohm;"
                " it cannot be applied without changing the model"
            )

    def to_cal_element(self, base_dir: str | None = None) -> CalElement:
        """Build the :class:`CalElement` the correction code consumes."""
        self._require_supported_reference()
        element = CalElement()
        for standard_name in STANDARD_NAMES:
            standard = self.standards[standard_name]
            state = _state_for_model(standard.model)
            if standard_name == "through":
                element.through_is_ideal = standard.model == MODEL_IDEAL
                _copy_coefficients(element, "through", standard)
                continue
            setattr(element, f"{standard_name}_state", state)
            setattr(
                element,
                f"{standard_name}_is_ideal",
                standard.model == MODEL_IDEAL,
            )
            if standard.model == MODEL_FILE:
                touchstone = _load_touchstone(
                    standard.file_ref, base_dir, standard_name, self.name
                )
                setattr(element, f"{standard_name}_touchstone", touchstone)
            _copy_coefficients(element, standard_name, standard)
        return element

    def apply_to(self, calibration, base_dir: str | None = None) -> CalElement:
        """Install this kit on ``calibration`` and record its provenance.

        Only the standard description is replaced; measured calibration data
        and the correction equations are untouched, and ``isCalculated`` is
        left alone so the caller decides when to recalculate.
        """
        element = self.to_cal_element(base_dir)
        calibration.cal_element = element
        calibration.kit_name = self.name
        calibration.kit_identity = self.calibration_identity()
        return element

    @classmethod
    def from_cal_element(  # noqa: PLR0913 - keyword-only capture options
        cls,
        element: CalElement,
        *,
        name: str,
        description: str = "",
        connector: KitConnector | None = None,
        reference_impedance: float = SUPPORTED_REFERENCE_IMPEDANCE,
        valid_span: KitValidSpan | None = None,
        created: str = "",
        source: str = "",
        base_dir: str | None = None,
    ) -> CalibrationKit:
        """Capture the current standard model of ``element`` as a kit."""
        standards: dict[str, KitStandard] = {}
        for standard_name in STANDARD_NAMES:
            if standard_name == "through":
                model = (
                    MODEL_IDEAL
                    if getattr(element, "through_is_ideal", True)
                    else MODEL_COEFF
                )
            else:
                model = _model_for_state(
                    getattr(element, f"{standard_name}_state", "")
                )
            coefficients: dict[str, float] = {}
            file_ref = ""
            if model == MODEL_COEFF:
                for coefficient in STANDARD_COEFFICIENTS[standard_name]:
                    coefficients[coefficient] = float(
                        getattr(
                            element,
                            f"{standard_name}_{coefficient}",
                        )
                    )
            elif model == MODEL_FILE:
                touchstone = getattr(
                    element, f"{standard_name}_touchstone", None
                )
                file_ref = _relative_file_ref(
                    getattr(touchstone, "filename", "") or "", base_dir
                )
                if not file_ref:
                    raise CalibrationKitError(
                        f"the {standard_name} standard uses a file but no"
                        " file name is recorded"
                    )
            standards[standard_name] = KitStandard(
                model=model, coefficients=coefficients, file_ref=file_ref
            )
        return cls(
            name=name,
            description=description,
            connector=connector if connector is not None else KitConnector(),
            reference_impedance=reference_impedance,
            valid_span=valid_span,
            standards=standards,
            created=created or _utc_now(),
            source=source,
        )


def _state_for_model(model: str) -> str:
    return {
        MODEL_IDEAL: "IDEAL",
        MODEL_COEFF: "COEFF",
        MODEL_FILE: "FILE",
    }[model]


def _model_for_state(state: str) -> str:
    """Map a ``CalElement`` state string to a kit model.

    An empty state means "use the coefficient model", which is what
    ``Calibration.gamma_*`` does for a freshly built ``CalElement``.
    """
    if state == "IDEAL":
        return MODEL_IDEAL
    if state == "FILE":
        return MODEL_FILE
    if state in ("", "COEFF"):
        return MODEL_COEFF
    raise CalibrationKitError(f"unknown calibration standard state {state!r}")


def _copy_coefficients(
    element: CalElement, standard_name: str, standard: KitStandard
) -> None:
    if standard.model != MODEL_COEFF:
        return
    for coefficient, value in standard.coefficients.items():
        setattr(element, f"{standard_name}_{coefficient}", value)


def _load_touchstone(
    file_ref: str,
    base_dir: str | None,
    standard_name: str,
    kit_name: str,
) -> Touchstone:
    path = _resolve_file_ref(file_ref, base_dir)
    if not os.path.isfile(path):
        raise CalibrationKitError(
            f"the {standard_name} standard of kit {kit_name!r} refers to"
            f" {path!r}, which does not exist"
        )
    touchstone = Touchstone(filename=path)
    touchstone.load()
    return touchstone


def _resolve_file_ref(file_ref: str, base_dir: str | None) -> str:
    path = os.path.expanduser(file_ref)
    if os.path.isabs(path) or not base_dir:
        return path
    return os.path.join(base_dir, path)


def _relative_file_ref(filename: str, base_dir: str | None) -> str:
    """Store a standard file relative to the kit document when possible."""
    if not filename:
        return ""
    if not base_dir:
        return filename
    try:
        relative = os.path.relpath(filename, base_dir)
    except ValueError:  # different drive on Windows
        return filename
    if relative.startswith(os.pardir):
        return filename
    return relative


def displayed_value(standard: str, key: str, value: float) -> float:
    """Convert a stored SI coefficient to the number the window shows.

    The calibration window enters inductance in nH, capacitance in pF and
    delays in ps while the kit stores H, F and s, so the display value is
    the stored value times the field's divisor.  Reusing the window's own
    divisor is what keeps a save/load cycle free of floating-point drift.
    """
    _, divisor, _ = COEFFICIENT_METADATA[(standard, key)]
    return value * divisor


def stored_value(standard: str, key: str, displayed: float) -> float:
    """Convert a number the window shows to the stored SI coefficient."""
    _, divisor, _ = COEFFICIENT_METADATA[(standard, key)]
    return displayed / divisor


def kit_from_legacy_values(
    name: str,
    values: Mapping[str, Any],
    *,
    description: str = "Imported from saved calibration standards",
    created: str = "",
) -> CalibrationKit:
    """Build a kit from one ``QSettings`` "CalibrationStandards" entry.

    The saved text holds the values exactly as the calibration window shows
    them (nH/pF/ps), so :data:`LEGACY_UNITS` converts them to the SI values
    a kit stores.  Missing keys fall back to the same defaults the
    calibration window uses, so old entries keep loading unchanged.
    """
    coefficients: dict[str, dict[str, float]] = {
        standard: {} for standard in STANDARD_NAMES
    }
    for key, (standard, coefficient, divisor) in LEGACY_UNITS.items():
        raw = values.get(key, LEGACY_DEFAULTS[key])
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raw = LEGACY_DEFAULTS[key]
        try:
            shown = float(raw)
        except (TypeError, ValueError) as exc:
            raise CalibrationKitValidationError(
                f"{key} of calibration standard {name!r} is not a number:"
                f" {raw!r}"
            ) from exc
        if not math.isfinite(shown):
            raise CalibrationKitValidationError(
                f"{key} of calibration standard {name!r} is not finite"
            )
        coefficients[standard][coefficient] = shown / divisor
    standards = {
        standard: KitStandard(
            model=MODEL_COEFF, coefficients=coefficients[standard]
        )
        for standard in STANDARD_NAMES
    }
    return CalibrationKit(
        name=name,
        description=description,
        standards=standards,
        created=created or _utc_now(),
        source="legacy-settings",
    )


def legacy_values(kit: CalibrationKit) -> dict[str, float]:
    """Return the window's text values for ``kit``'s coefficient model.

    The result is keyed by the historical ``QSettings`` names, so a kit can
    be written back into the "Calibration standards" array this build still
    reads.  A standard that is not coefficient based contributes its
    neutral defaults rather than pretending to have coefficients.
    """
    values: dict[str, float] = {}
    for standard, key, legacy_key, divisor, default in COEFFICIENT_FIELDS:
        standard_model = kit.standards[standard]
        if standard_model.model != MODEL_COEFF:
            values[legacy_key] = default
            continue
        coefficient = standard_model.coefficients.get(key, default / divisor)
        values[legacy_key] = coefficient * divisor
    return values


# ----------------------------------------------------------------------
# Built-in kits
# ----------------------------------------------------------------------


def _ideal_standards() -> dict[str, KitStandard]:
    return {name: KitStandard(model=MODEL_IDEAL) for name in STANDARD_NAMES}


def _sma_male_standards() -> dict[str, KitStandard]:
    """The coefficients this application has always used as defaults.

    ``CalElement`` defaults are taken verbatim, except that the short offset
    delay is expressed in seconds: the field is documented as ps and the
    calibration window divides the entered ps by 1e12, so -34.2 ps becomes
    -34.2e-12 s here.
    """
    return {
        "short": KitStandard(
            model=MODEL_COEFF,
            coefficients={
                "l0": 5.7e-12,
                "l1": -8.96e-20,
                "l2": -1.1e-29,
                "l3": -4.12e-37,
                "length": -34.2e-12,
            },
        ),
        "open": KitStandard(
            model=MODEL_COEFF,
            coefficients={
                "c0": 2.1e-14,
                "c1": 5.67e-23,
                "c2": -2.39e-31,
                "c3": 2.0e-40,
                "length": 0.0,
            },
        ),
        "load": KitStandard(
            model=MODEL_COEFF,
            coefficients={"r": 50.0, "l": 0.0, "c": 0.0, "length": 0.0},
        ),
        "through": KitStandard(model=MODEL_IDEAL),
    }


IDEAL_KIT = CalibrationKit(
    name="Ideal standards",
    description=(
        "Textbook short, open, load and through. Best used only as a"
        " baseline: a real kit's parasitics are what the coefficients model."
    ),
    connector=KitConnector(type="none"),
    standards=_ideal_standards(),
    created="2026-01-01T00:00:00Z",
    source="builtin",
)

SMA_MALE_KIT = CalibrationKit(
    name="SMA male (typical copper)",
    description=(
        "The application's historical default coefficients for a male SMA"
        " short, open and load, with an ideal through."
    ),
    connector=KitConnector(type="SMA", sex="male"),
    standards=_sma_male_standards(),
    created="2026-01-01T00:00:00Z",
    source="builtin",
)

#: Immutable built-in kits, in display order.
BUILTIN_KITS: tuple[CalibrationKit, ...] = (IDEAL_KIT, SMA_MALE_KIT)
BUILTIN_NAMES: tuple[str, ...] = tuple(kit.name for kit in BUILTIN_KITS)


def builtin_kits() -> tuple[CalibrationKit, ...]:
    """The built-in kits; never stored, always rebuilt from code."""
    return BUILTIN_KITS


def find_builtin(name: str) -> CalibrationKit | None:
    for kit in BUILTIN_KITS:
        if kit.name == name:
            return kit
    return None


# ----------------------------------------------------------------------
# User library
# ----------------------------------------------------------------------


def default_library_directory() -> str:
    """Platform application-data directory for the kit library.

    Evaluated lazily so nothing imports Qt (or touches the file system)
    until a library without an explicit directory is actually used.
    """
    fallback = os.path.join(
        os.path.expanduser("~"), ".local", "share", "NanoVNASaver"
    )
    try:
        from PySide6.QtCore import QStandardPaths
    except ImportError:  # pragma: no cover - Qt is a hard dependency
        return fallback
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    return location or fallback


def _atomic_write_json(filename: str, document: Mapping) -> None:
    payload = json.dumps(document, sort_keys=True, indent=2).encode("utf-8")
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=".nvcal-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, filename)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _read_json(filename: str) -> Any:
    try:
        with open(filename, encoding="utf-8") as source:
            return json.load(source)
    except OSError as exc:
        raise CalibrationKitError(
            f"could not read calibration kits from {filename}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CalibrationKitValidationError(
            f"{filename} is not valid JSON: {exc}"
        ) from exc


def kit_from_document(
    document: Any, where: str = "kit document"
) -> CalibrationKit:
    """Parse a single-kit JSON object with validation."""
    data = _mapping(document, where)
    try:
        return CalibrationKit.from_dict(data)
    except CalibrationKitValidationError as exc:
        raise CalibrationKitValidationError(f"{where}: {exc}") from exc


def kits_from_document(
    document: Any, where: str = "kit document"
) -> tuple[CalibrationKit, ...]:
    """Parse either a single kit or a library document."""
    data = _mapping(document, where)
    if data.get("format") == LIBRARY_FORMAT:
        version = data.get("version", LIBRARY_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise CalibrationKitValidationError(
                f"{where}: version must be an integer"
            )
        if version > LIBRARY_VERSION:
            raise CalibrationKitValidationError(
                f"{where}: library version {version} is newer than supported"
                f" ({LIBRARY_VERSION})"
            )
        _check_keys(data, frozenset({"format", "version", "kits"}), where)
        entries = data.get("kits", [])
        if not isinstance(entries, list):
            raise CalibrationKitValidationError(f"{where}: kits must be a list")
        return tuple(
            kit_from_document(entry, f"{where} kit {index}")
            for index, entry in enumerate(entries)
        )
    return (kit_from_document(data, where),)


class KitLibrary:
    """User kits stored as one versioned JSON document.

    ``directory`` is explicit for tests and for callers that already know
    where application data lives; without it the platform application-data
    directory is used, evaluated when first needed.  Mutation methods only
    change memory: :meth:`save` writes the whole document atomically, so a
    rejected operation leaves both memory and disk untouched.
    """

    def __init__(self, directory: str | None = None):
        self._directory = directory
        self._kits: list[CalibrationKit] = []
        self._transient: list[CalibrationKit] = []
        #: Why the file on disk could not be read, if it could not.
        self._unreadable = ""

    # -- location ------------------------------------------------------
    @property
    def directory(self) -> str:
        if self._directory is None:
            self._directory = default_library_directory()
        return self._directory

    @property
    def filename(self) -> str:
        return os.path.join(self.directory, LIBRARY_FILENAME)

    # -- reading -------------------------------------------------------
    @property
    def builtins(self) -> tuple[CalibrationKit, ...]:
        return builtin_kits()

    def user_kits(self) -> list[CalibrationKit]:
        return list(self._kits)

    def all_kits(self) -> list[CalibrationKit]:
        return list(BUILTIN_KITS) + list(self._kits) + list(self._transient)

    def get(self, name: str) -> CalibrationKit | None:
        for kit in (*self._kits, *self._transient):
            if kit.name == name:
                return kit
        return find_builtin(name)

    def load(self) -> KitLibrary:
        """Read the library; a failure also blocks :meth:`save`."""
        self._unreadable = ""
        try:
            return self._read()
        except (CalibrationKitError, OSError) as exc:
            self._unreadable = str(exc)
            raise

    def _read(self) -> KitLibrary:
        """Read the library document, replacing the in-memory user kits.

        A missing document is an empty library, not an error.  A malformed
        one raises and leaves the in-memory kits as they were, so the
        window can report the problem and still show what it had.
        """
        if not os.path.exists(self.filename):
            return self
        document = _read_json(self.filename)
        data = _mapping(document, self.filename)
        if data.get("format") != LIBRARY_FORMAT:
            raise CalibrationKitValidationError(
                f"{self.filename}: unexpected format"
                f" {data.get('format')!r}; expected {LIBRARY_FORMAT!r}"
            )
        version = data.get("version", LIBRARY_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise CalibrationKitValidationError(
                f"{self.filename}: version must be an integer"
            )
        if version > LIBRARY_VERSION:
            raise CalibrationKitValidationError(
                f"{self.filename}: library version {version} is newer than"
                f" supported ({LIBRARY_VERSION})"
            )
        _check_keys(
            data, frozenset({"format", "version", "kits"}), self.filename
        )
        entries = data.get("kits", [])
        if not isinstance(entries, list):
            raise CalibrationKitValidationError(
                f"{self.filename}: kits must be a list"
            )
        kits = [
            kit_from_document(entry, f"{self.filename} kit {index}")
            for index, entry in enumerate(entries)
        ]
        self._kits = self._without_duplicates(kits)
        return self

    @staticmethod
    def _without_duplicates(
        kits: Sequence[CalibrationKit],
    ) -> list[CalibrationKit]:
        seen: dict[str, CalibrationKit] = {}
        for kit in kits:
            if kit.name in BUILTIN_NAMES:
                logger.warning(
                    "Ignoring stored kit %r: a built-in kit has that name",
                    kit.name,
                )
                continue
            if kit.name in seen:
                logger.warning("Ignoring duplicate stored kit %r", kit.name)
                continue
            seen[kit.name] = kit
        return list(seen.values())

    # -- transient kits ------------------------------------------------
    #
    # Kits migrated from older storage (the "Calibration standards" array)
    # belong to that old storage: they are shown and can be applied, but
    # they are never written to this library, and the old settings are
    # never modified or removed.  Registering the same name twice keeps
    # the newest registration, so a re-read after the user saved a legacy
    # standard does not pile up duplicates.
    def register_transient(self, kit: CalibrationKit) -> CalibrationKit:
        if not isinstance(kit, CalibrationKit):
            raise CalibrationKitError("a CalibrationKit is required")
        self._transient = [
            existing
            for existing in self._transient
            if existing.name != kit.name
        ]
        self._transient.append(kit)
        return kit

    def transient_kits(self) -> list[CalibrationKit]:
        return list(self._transient)

    def clear_transient(self) -> None:
        self._transient = []

    # -- writing -------------------------------------------------------
    def to_document(self) -> dict:
        return {
            "format": LIBRARY_FORMAT,
            "version": LIBRARY_VERSION,
            "kits": [kit.to_dict() for kit in self._kits],
        }

    def save(self) -> None:
        if self._unreadable:
            # The file may hold kits this build could not read, for example
            # one written by a newer version.  Saving now would replace them
            # with whatever is in memory, permanently.
            raise CalibrationKitError(
                f"not overwriting the unreadable library {self.filename}:"
                f" {self._unreadable}. Repair or move that file, then restart."
            )
        _atomic_write_json(self.filename, self.to_document())

    # -- mutation ------------------------------------------------------
    def add(self, kit: CalibrationKit, replace: bool = False) -> CalibrationKit:
        """Store ``kit``; refuse a name collision unless ``replace``."""
        if not isinstance(kit, CalibrationKit):
            raise CalibrationKitError("a CalibrationKit is required")
        if kit.name in BUILTIN_NAMES:
            raise CalibrationKitNameError(
                f"{kit.name!r} is a built-in kit and cannot be replaced;"
                " duplicate it under a new name instead"
            )
        for index, existing in enumerate(self._kits):
            if existing.name == kit.name:
                if not replace:
                    raise CalibrationKitNameError(
                        f"a calibration kit named {kit.name!r} already"
                        " exists; pass replace=True to overwrite it"
                    )
                self._kits[index] = kit
                return kit
        if any(existing.name == kit.name for existing in self._transient):
            # A stored kit supersedes the legacy copy of the same name.
            self._transient = [
                existing
                for existing in self._transient
                if existing.name != kit.name
            ]
        self._kits.append(kit)
        return kit

    def duplicate(self, name: str, new_name: str) -> CalibrationKit:
        source = self.get(name)
        if source is None:
            raise CalibrationKitError(f"no calibration kit named {name!r}")
        if self.get(new_name) is not None:
            raise CalibrationKitNameError(
                f"a calibration kit named {new_name!r} already exists"
            )
        return self.add(replace(source, name=new_name, created=_utc_now()))

    def rename(self, name: str, new_name: str) -> CalibrationKit:
        for index, kit in enumerate(self._kits):
            if kit.name != name:
                continue
            if self.get(new_name) is not None:
                raise CalibrationKitNameError(
                    f"a calibration kit named {new_name!r} already exists"
                )
            renamed = kit.renamed(new_name)
            self._kits[index] = renamed
            return renamed
        if find_builtin(name) is not None:
            raise CalibrationKitNameError(
                f"{name!r} is a built-in kit and cannot be renamed;"
                " duplicate it instead"
            )
        raise CalibrationKitError(f"no user calibration kit named {name!r}")

    def delete(self, name: str) -> bool:
        for index, kit in enumerate(self._kits):
            if kit.name == name:
                del self._kits[index]
                return True
        if find_builtin(name) is not None:
            raise CalibrationKitNameError(
                f"{name!r} is a built-in kit and cannot be deleted;"
                " store a copy under a new name if you need to change it"
            )
        return False

    def replace_kit(
        self, kit: CalibrationKit, replace: bool = False
    ) -> CalibrationKit:
        """Alias of :meth:`add`, named after the operation it performs."""
        return self.add(kit, replace=replace)

    # -- transfer ------------------------------------------------------
    def export_kit(self, name: str, filename: str) -> str:
        kit = self.get(name)
        if kit is None:
            raise CalibrationKitError(f"no calibration kit named {name!r}")
        _atomic_write_json(filename, kit.to_dict())
        return filename

    def import_file(
        self, filename: str, replace: bool = False
    ) -> tuple[CalibrationKit, ...]:
        """Import one kit or a whole library document.

        Validation happens before anything is added, and a name collision
        rejects the whole import, so a failed import never leaves a partial
        set of kits behind.
        """
        kits = kits_from_document(_read_json(filename), filename)
        pending = list(self._kits)
        names = {kit.name for kit in pending}
        for kit in kits:
            if kit.name in BUILTIN_NAMES:
                raise CalibrationKitNameError(
                    f"{filename}: {kit.name!r} would override a built-in kit"
                )
            if kit.name in names and not replace:
                raise CalibrationKitNameError(
                    f"{filename}: a calibration kit named {kit.name!r}"
                    " already exists; pass replace=True to overwrite it"
                )
            names.add(kit.name)
            pending = [
                existing for existing in pending if existing.name != kit.name
            ]
            pending.append(kit)
        self._kits = pending
        return kits

    def import_kit(
        self, filename: str, replace: bool = False
    ) -> CalibrationKit:
        kits = self.import_file(filename, replace=replace)
        if len(kits) != 1:
            raise CalibrationKitError(
                f"{filename} holds {len(kits)} kits; use import_file() for"
                " a library document"
            )
        return kits[0]

    @classmethod
    def open(cls, directory: str | None = None) -> KitLibrary:
        """Open (and read) the library in ``directory``."""
        return cls(directory=directory).load()
