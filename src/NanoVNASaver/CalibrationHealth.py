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
"""Calibration age, metadata and verification (CAL-02).

Three separate statements, kept apart so one never reads as another:

* **Coverage** (:mod:`CalibrationCoverage`) says which frequencies a
  calibration spans.
* **Age** says how long ago the standards were measured.  A reminder fires at
  an interval the operator chose.  It is off by default: there is no number of
  hours after which a calibration is objectively expired.
* **Verification** says how far one re-measured standard, corrected by the
  calibration under test, lies from the model that calibration assumes.

None of them is an uncertainty.  A passed verification means one standard was
within the operator's tolerance at the moment it was checked.  It does not
bound the error of any other measurement, so nothing here produces an error
bar.

Metadata travels as one comment line in the calibration file.  Older readers
keep it as a note, and a file without it has an *unknown* age rather than a
guessed one.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from .RFTools import Datapoint

logger = logging.getLogger(__name__)

#: Marks the metadata comment in a calibration file.
METADATA_PREFIX = "cal-metadata: "
#: The reminder is an operator choice, so the default is no reminder at all.
DEFAULT_REMINDER_HOURS = 0.0
#: A starting point for the check, not a claim about any kit or instrument.
DEFAULT_TOLERANCE = 0.05
VERIFIABLE_STANDARDS = ("short", "open", "load")


class VerificationError(ValueError):
    """The standard could not be checked, so nothing is reported about it."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CalibrationMetadata:
    """What is known about how a calibration was made; all of it optional."""

    created: str = ""
    device: str = ""
    kit: str = ""
    connector: str = ""
    operator: str = ""

    def to_note(self) -> str:
        """The comment line for a calibration file, or ``""`` when empty."""
        values = {key: value for key, value in asdict(self).items() if value}
        if not values:
            return ""
        return METADATA_PREFIX + json.dumps(values, sort_keys=True)

    @classmethod
    def from_note(cls, note: str) -> "CalibrationMetadata":
        """Parse a metadata note; anything unreadable means *unknown*."""
        try:
            values = json.loads(note[len(METADATA_PREFIX) :])
        except json.JSONDecodeError:
            logger.warning("Ignoring unreadable calibration metadata")
            return cls()
        if not isinstance(values, dict):
            logger.warning("Ignoring unreadable calibration metadata")
            return cls()
        known = {field.name for field in fields(cls)}
        return cls(
            **{
                key: value
                for key, value in values.items()
                if key in known and isinstance(value, str)
            }
        )

    def describe(self) -> str:
        parts = (
            ("Device", self.device),
            ("Kit", self.kit),
            ("Connector", self.connector),
            ("Operator", self.operator),
        )
        return " · ".join(f"{name}: {value}" for name, value in parts if value)


def split_metadata(
    notes: Sequence[str],
) -> tuple[CalibrationMetadata, list[str]]:
    """Separate the metadata line from the operator's own notes."""
    metadata = CalibrationMetadata()
    remaining = []
    for note in notes:
        if note.startswith(METADATA_PREFIX):
            metadata = CalibrationMetadata.from_note(note)
        else:
            remaining.append(note)
    return metadata, remaining


class AgeState(Enum):
    UNKNOWN = "unknown"
    """No usable creation time, as for every file from an older version."""
    CURRENT = "current"
    """Known, and no reminder is due."""
    REMINDER = "reminder"
    """Older than the interval the operator asked to be reminded after."""
    FUTURE = "future"
    """Stamped later than now: a clock is wrong, so no age can be stated."""


@dataclass(frozen=True)
class CalibrationAge:
    state: AgeState
    seconds: float | None = None
    reminder_hours: float = DEFAULT_REMINDER_HOURS

    @property
    def warns(self) -> bool:
        return self.state in (AgeState.REMINDER, AgeState.FUTURE)

    def describe(self) -> str:
        if self.state is AgeState.UNKNOWN:
            return "Age unknown: this calibration has no creation time"
        if self.state is AgeState.FUTURE:
            return (
                "Age unavailable: created after the current time;"
                " check the system clock"
            )
        text = f"Age {_format_duration(self.seconds or 0.0)}"
        if self.state is AgeState.REMINDER:
            text += f", past your {self.reminder_hours:g} h reminder"
        return text


def _format_duration(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def parse_created(created: str) -> datetime | None:
    """A timezone-aware time, or ``None``: a naive one has no defined age."""
    if not created:
        return None
    try:
        parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed


def calibration_age(
    created: str,
    *,
    reminder_hours: float = DEFAULT_REMINDER_HOURS,
    now: datetime | None = None,
) -> CalibrationAge:
    moment = parse_created(created)
    if moment is None:
        return CalibrationAge(AgeState.UNKNOWN, reminder_hours=reminder_hours)
    current = now if now is not None else datetime.now(timezone.utc)
    seconds = (current - moment).total_seconds()
    if seconds < 0:
        return CalibrationAge(AgeState.FUTURE, reminder_hours=reminder_hours)
    due = reminder_hours > 0 and seconds >= reminder_hours * 3600
    return CalibrationAge(
        AgeState.REMINDER if due else AgeState.CURRENT,
        seconds=seconds,
        reminder_hours=reminder_hours,
    )


@dataclass(frozen=True)
class StandardVerification:
    """One re-measured standard against the model the calibration assumes.

    ``worst_residual`` is the largest ``|corrected - expected|`` in linear
    reflection-coefficient units, checked only where the calibration has data.
    """

    standard: str
    passed: bool
    tolerance: float
    worst_residual: float
    worst_frequency: int
    points: int
    skipped_points: int = 0
    simulated: bool = False
    identity: str = ""

    def describe(self) -> str:
        verdict = "within" if self.passed else "OUTSIDE"
        text = (
            f"{self.standard.capitalize()}: worst |ΔΓ| "
            f"{self.worst_residual:.4f} at {self.worst_frequency} Hz,"
            f" {verdict} your {self.tolerance:g} tolerance"
            f" ({self.points} points"
        )
        if self.skipped_points:
            text += f", {self.skipped_points} outside the calibrated span"
        text += ")"
        if self.simulated:
            text += " [simulated, not hardware evidence]"
        return text


def validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0:
        raise VerificationError("the tolerance must be a positive number")
    return value


def verify_standard(
    calibration,
    standard: str,
    raw: Sequence[Datapoint],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    simulated: bool = False,
) -> StandardVerification:
    """Check one re-measured standard against ``calibration``.

    ``raw`` must be the uncorrected S11 of that standard.  Using the raw
    capture is what keeps this a test of the calibration: a corrected or
    processed trace would carry an offset delay or smoothing into the result.

    Only frequencies inside the calibrated span count.  Beyond it the
    correction is held at an endpoint, so a residual there would measure the
    hold and not the calibration.
    """
    from .Measurement import compute_calibration_identity

    if standard not in VERIFIABLE_STANDARDS:
        raise VerificationError(
            f"{standard!r} cannot be verified; choose one of"
            f" {', '.join(VERIFIABLE_STANDARDS)}"
        )
    limit = validate_tolerance(tolerance)
    if not calibration.isCalculated:
        raise VerificationError("apply a calibration before verifying it")
    if not raw:
        raise VerificationError("the standard's sweep returned no S11 samples")
    start = calibration.dataset.freq_min()
    stop = calibration.dataset.freq_max()
    expected_at = getattr(calibration, f"gamma_{standard}")
    worst = -1.0
    worst_frequency = 0
    checked = 0
    for point in raw:
        if not start <= point.freq <= stop:
            continue
        if not (math.isfinite(point.re) and math.isfinite(point.im)):
            raise VerificationError(
                f"the sample at {point.freq} Hz is not a finite number"
            )
        corrected = calibration.correct11(point)
        residual = abs(corrected.z - expected_at(point.freq))
        checked += 1
        if residual > worst:
            worst = residual
            worst_frequency = point.freq
    if not checked:
        raise VerificationError(
            "this sweep does not overlap the calibrated span"
            f" ({start} Hz to {stop} Hz)"
        )
    return StandardVerification(
        standard=standard,
        passed=worst <= limit,
        tolerance=limit,
        worst_residual=float(worst),
        worst_frequency=worst_frequency,
        points=checked,
        skipped_points=len(raw) - checked,
        simulated=simulated,
        identity=compute_calibration_identity(calibration),
    )
