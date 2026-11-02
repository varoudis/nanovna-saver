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
"""Reference-impedance (Z0) propagation: the audit, and its guards (RF-02).

An S-parameter is a *ratio against a reference impedance*.  Printing
``R 25 ohm`` for the same complex gamma means two different physical loads
depending on whether the data is 50-ohm or 75-ohm normalized, so every place
that turns gamma into an impedance has to know which reference applies.  A
literal ``50`` in such a place is a silent unit conversion from the truth.

This module holds three things:

* :data:`AUDIT_TABLE` - the audit of every place in the code base that
  either honours the real reference impedance or still assumes 50 ohm,
  expressed as data so the list cannot rot into a comment.  Each entry
  carries a status, one of :data:`PROPAGATED`, :data:`HARD_CODED_REFUSED`
  or :data:`HARD_CODED_REMAINING`; :data:`AUDIT` is the plain
  ``location -> status`` mapping the audit is reported as.  A location is
  ``path:line``, which is precise but moves when the file above it is
  edited; the status is what the audit is *about*, and the test suite only
  enforces that each line names real code in a real file, so a stale line
  number is a documentation defect, not a silent wrong answer.
* :func:`require_reference_impedance` and
  :func:`format_reference_impedance` - the one rule for "is this a usable
  reference impedance?" and how the number is written into a Touchstone
  header.
* :func:`renormalization_blocker` - the predicate the reference-impedance
  control needs *before* it runs: renormalizing S21/S12 between unequal
  port impedances needs the full wave-cascading two-port transform, which
  is not implemented, so the combination is refused by name instead of
  approximated.  :mod:`NanoVNASaver.Processing` raises the same refusal;
  the two are kept in step by test.

Nothing here imports Qt, so the acquisition side, the export side and the
tests can all use it.  This module never renormalizes anything itself:
:func:`NanoVNASaver.Processing.renormalize_gamma` owns the arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from .Processing import FORWARD_CHANNELS, PORT1_CHANNELS, REVERSE_CHANNELS

#: Reference impedance of the application default, stated once here rather
#: than repeated as a literal in every default argument.
DEFAULT_REFERENCE_IMPEDANCE = 50.0

#: The real value is carried by the data and nothing overrides it.
PROPAGATED = "propagated"
#: A literal 50 is present, but the code refuses to act when the data
#: disagrees, so the literal never decides a published number.
HARD_CODED_REFUSED = "hard_coded_refused"
#: A literal 50 (or a 50 default) still decides a displayed, analysed or
#: exported number.  These are the call sites that need wiring.
HARD_CODED_REMAINING = "hard_coded_remaining"

#: Statuses the audit may use.
STATUSES: tuple[str, ...] = (
    PROPAGATED,
    HARD_CODED_REFUSED,
    HARD_CODED_REMAINING,
)

#: Channels whose gamma is a one-port reflection coefficient.
REFLECTION_CHANNELS: tuple[str, ...] = PORT1_CHANNELS
#: Channels that need a two-port transform to change reference impedance.
TRANSMISSION_CHANNELS: tuple[str, ...] = FORWARD_CHANNELS + REVERSE_CHANNELS


class ReferenceImpedanceError(ValueError):
    """A reference impedance is missing, non-positive or not finite."""


@dataclass(frozen=True)
class AuditEntry:
    """One audited location and what it does about the reference impedance."""

    location: str
    status: str
    note: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"{self.location}: unknown audit status {self.status!r};"
                f" use one of {', '.join(STATUSES)}"
            )
        if not self.location or not self.note:
            raise ValueError("an audit entry needs a location and a note")

    @property
    def pending(self) -> bool:
        """True when this call site still assumes 50 ohm."""
        return self.status == HARD_CODED_REMAINING


_AUDIT_ROWS: tuple[tuple[str, str, str], ...] = (
    # -- the reference impedance itself --------------------------------
    (
        "src/NanoVNASaver/Measurement.py:435",
        PROPAGATED,
        "MeasurementSnapshot carries reference_impedance as a required"
        " field; consumers read the snapshot, not a literal.",
    ),
    (
        "src/NanoVNASaver/Measurement.py:618",
        PROPAGATED,
        "the snapshot refuses a non-positive reference impedance at"
        " construction.",
    ),
    (
        "src/NanoVNASaver/Measurement.py:721",
        PROPAGATED,
        "snapshot_from_touchstone takes the reference impedance from the"
        " file header's R keyword instead of assuming 50.",
    ),
    (
        "src/NanoVNASaver/Measurement.py:840",
        PROPAGATED,
        "snapshot_to_touchstone writes the snapshot's reference impedance"
        " into the exported option line.",
    ),
    (
        "src/NanoVNASaver/Measurement.py:676",
        HARD_CODED_REMAINING,
        "deserialising a stored snapshot falls back to a literal 50 when"
        " the key is absent; correct only for sessions written before the"
        " field existed, and invisible to whoever reopens one.",
    ),
    # -- processing ----------------------------------------------------
    (
        "src/NanoVNASaver/Processing.py:558",
        PROPAGATED,
        "renormalize_gamma applies G' = (G + a)/(1 + a*G) with"
        " a = (Z0_old - Z0_new)/(Z0_old + Z0_new); non-positive,"
        " non-finite and singular inputs are refused.",
    ),
    (
        "src/NanoVNASaver/Processing.py:583",
        PROPAGATED,
        "apply_renormalize renormalizes S11/S22 only and refuses S21/S12"
        " by name instead of approximating the two-port transform.",
    ),
    (
        "src/NanoVNASaver/Processing.py:1077",
        PROPAGATED,
        "ProcessingResult.reference_impedance states which reference the"
        " returned samples are expressed against.",
    ),
    (
        "src/NanoVNASaver/Processing.py:1122",
        PROPAGATED,
        "apply_processing threads the running reference through the"
        " pipeline and updates it when a Z0 stage runs.",
    ),
    # -- exchange format ----------------------------------------------
    (
        "src/NanoVNASaver/Touchstone.py:164",
        PROPAGATED,
        "Options.parse reads the R keyword wherever it appears on the"
        " option line and rejects an unusable value.",
    ),
    (
        "src/NanoVNASaver/Touchstone.py:541",
        PROPAGATED,
        "the writer emits R <snapshot Z0> in the option line, so a 75"
        " ohm export stays 75 ohm.",
    ),
    # -- consumers that already ask the snapshot ----------------------
    (
        "src/NanoVNASaver/Analysis/Results.py:211",
        PROPAGATED,
        "resonance impedance is computed against snapshot.reference_impedance.",
    ),
    (
        "src/NanoVNASaver/Analysis/Results.py:550",
        PROPAGATED,
        "the |Z| statistics use snapshot.reference_impedance.",
    ),
    (
        "src/NanoVNASaver/Traces.py:536",
        PROPAGATED,
        "trace comparison refuses two traces with different reference"
        " impedances instead of comparing incomparable numbers.",
    ),
    (
        "src/NanoVNASaver/TraceMath.py:431",
        PROPAGATED,
        "trace math refuses to combine traces with different reference"
        " impedances.",
    ),
    (
        "src/NanoVNASaver/Reports.py:826",
        PROPAGATED,
        "the report states the reference impedance of every trace it reports.",
    ),
    (
        "src/NanoVNASaver/Fixture.py:307",
        PROPAGATED,
        "fixture de-embedding refuses a fixture measured against another"
        " reference impedance.",
    ),
    (
        "src/NanoVNASaver/Session.py:121",
        PROPAGATED,
        "the session stores the snapshot's reference impedance.",
    ),
    (
        "src/NanoVNASaver/Windows/Files.py:112",
        PROPAGATED,
        "export copies the reference impedance of the measurement it"
        " writes into the new Touchstone object.",
    ),
    (
        "src/NanoVNASaver/AntennaTuning.py:871",
        PROPAGATED,
        "the antenna export builds its file from the snapshot, so the"
        " header carries the measured reference impedance.",
    ),
    (
        "src/NanoVNASaver/Hardware/SimulatedScenarios.py:37",
        PROPAGATED,
        "the simulated device declares its 50 ohm reference instead of"
        " assuming one, and replay refuses a file that disagrees.",
    ),
    # -- Z0-independent by construction -------------------------------
    (
        "src/NanoVNASaver/Charts/Smith.py:29",
        PROPAGATED,
        "the Smith chart plots gamma on a normalized grid, so its geometry"
        " is reference-impedance independent by construction.",
    ),
    (
        "src/NanoVNASaver/Charts/VSWR.py:31",
        PROPAGATED,
        "VSWR is (1 + |G|)/(1 - |G|): no impedance, no reference.",
    ),
    (
        "src/NanoVNASaver/Charts/RI.py:34",
        PROPAGATED,
        "the Re/Im chart plots the components of gamma, which are"
        " dimensionless.",
    ),
    (
        "src/NanoVNASaver/Charts/Polar.py:29",
        PROPAGATED,
        "the polar chart plots gamma; it derives no impedance.",
    ),
    # -- literal 50 that is refused rather than applied ---------------
    (
        "src/NanoVNASaver/Calibration.py:627",
        HARD_CODED_REFUSED,
        "the short/open/load models divide by a literal 50 ohm;"
        " CalibrationKits refuses a kit declared for another reference,"
        " so the literal is never applied to 75 ohm data.",
    ),
    (
        "src/NanoVNASaver/CalibrationKits.py:629",
        HARD_CODED_REFUSED,
        "_require_supported_reference refuses to apply a kit whose"
        " reference is not SUPPORTED_REFERENCE_IMPEDANCE (50 ohm,"
        " CalibrationKits.py:197).",
    ),
    (
        "src/NanoVNASaver/Windows/CalibrationSettings.py:260",
        HARD_CODED_REFUSED,
        "the load standard defaults to a 50 ohm resistor, part of the same"
        " 50-ohm-only calibration model; a kit with another reference is"
        " refused when applied.",
    ),
    (
        "src/NanoVNASaver/Hardware/ReplayScenario.py:325",
        HARD_CODED_REFUSED,
        "a replayed file whose header R is not 50 is rejected with an"
        " actionable error instead of being silently read as 50 ohm.",
    ),
    # -- still assuming 50 ohm ----------------------------------------
    (
        "src/NanoVNASaver/RFTools.py:59",
        HARD_CODED_REMAINING,
        "Datapoint.impedance() defaults to 50 ohm and"
        " shuntImpedance/seriesImpedance/qFactor/capacitiveEquivalent/"
        "inductiveEquivalent (:62, :68, :74, :78, :83) inherit the same"
        " default, so every caller that omits the argument assumes 50.",
    ),
    (
        "src/NanoVNASaver/Charts/Chart.py:205",
        PROPAGATED,
        "Chart.setData accepts the reference impedance and Chart holds it,"
        " so every chart inherits one value from the data it draws.",
    ),
    (
        "src/NanoVNASaver/Charts/RIZ.py:115",
        PROPAGATED,
        "p.impedance() is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/RIZSeries.py:29",
        PROPAGATED,
        "p.seriesImpedance() is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/RIZShunt.py:29",
        PROPAGATED,
        "p.shuntImpedance() is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/MagnitudeZ.py:142",
        PROPAGATED,
        "abs(p.impedance()) is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/MagnitudeZSeries.py:29",
        PROPAGATED,
        "abs(p.seriesImpedance()) is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/MagnitudeZShunt.py:29",
        PROPAGATED,
        "abs(p.shuntImpedance()) is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/Permeability.py:98",
        PROPAGATED,
        "d.impedance() is given the chart's reference (also :110, :294, :311).",
    ),
    (
        "src/NanoVNASaver/Charts/RIMu.py:193",
        PROPAGATED,
        "p.impedance() is given the chart's reference.",
    ),
    (
        "src/NanoVNASaver/Charts/QFactor.py:56",
        PROPAGATED,
        "d.qFactor() is given the chart's reference (also :128).",
    ),
    (
        "src/NanoVNASaver/Charts/Capacitance.py:33",
        PROPAGATED,
        "x.capacitiveEquivalent() is given the chart's reference (:32"
        " opens the lambda).",
    ),
    (
        "src/NanoVNASaver/Charts/Inductance.py:33",
        PROPAGATED,
        "x.inductiveEquivalent() is given the chart's reference (:32"
        " opens the lambda).",
    ),
    (
        "src/NanoVNASaver/Marker/Widget.py:354",
        PROPAGATED,
        "_s11.impedance() is given the marker's reference (qFactor at :394,"
        " shunt/series magnitudes and complexes below it).",
    ),
    (
        "src/NanoVNASaver/Marker/Delta.py:63",
        PROPAGATED,
        "s11_a/s11_b.impedance() are given the marker's reference (qFactor"
        " at :125).",
    ),
    (
        "src/NanoVNASaver/Analysis/EFHWAnalysis.py:51",
        PROPAGATED,
        "d.impedance() is given the application's reference (also :88).",
    ),
    (
        "src/NanoVNASaver/Analysis/ResonanceAnalysis.py:72",
        PROPAGATED,
        "impedance is given the application's reference.  The vswr_49/"
        " vswr_4 columns below are unchanged by design: they are named"
        " for the reference they transform to, not the measurement's.",
    ),
    (
        "src/NanoVNASaver/Analysis/SimplePeakSearchAnalysis.py:120",
        PROPAGATED,
        "d.impedance() is given the application's reference (also :125).",
    ),
    (
        "src/NanoVNASaver/Windows/TDR.py:346",
        PROPAGATED,
        "the step impedance is computed against the application's"
        " reference, and the reflection coefficient that follows from"
        " it uses the same one.",
    ),
    (
        "src/NanoVNASaver/NanoVNASaver.py:315",
        PROPAGATED,
        "self.reference_impedance starts at 50 and is adopted from an"
        " imported file by adopt_reference_impedance(), which refuses"
        " an unusable value instead of reverting.",
    ),
    (
        "src/NanoVNASaver/Windows/Files.py:233",
        PROPAGATED,
        "loadSweepFile adopts the file's reference impedance before the"
        " charts and markers are refreshed.",
    ),
    (
        "src/NanoVNASaver/SweepWorker.py:743",
        PROPAGATED,
        "snapshot_from_datapoints is given the reference the grid was"
        " acquired at.",
    ),
    (
        "src/NanoVNASaver/SweepWorker.py:751",
        PROPAGATED,
        "with_processing is given the reference ProcessingResult reports"
        " for the data it produced, so a renormalized run cannot publish"
        " 75-ohm channels under a 50-ohm label.",
    ),
)

#: The audit, in reading order, with a note per location.
AUDIT_TABLE: tuple[AuditEntry, ...] = tuple(
    AuditEntry(location, status, note) for location, status, note in _AUDIT_ROWS
)

#: ``location -> status``, the form the audit is reported in.
AUDIT: Mapping[str, str] = MappingProxyType(
    {entry.location: entry.status for entry in AUDIT_TABLE}
)

if len(AUDIT) != len(AUDIT_TABLE):
    raise ValueError(
        "the reference-impedance audit lists one location twice:"
        f" {len(AUDIT_TABLE)} rows, {len(AUDIT)} distinct locations"
    )

#: ``location -> entry``, for the note behind each status.
AUDIT_BY_LOCATION: Mapping[str, AuditEntry] = MappingProxyType(
    {entry.location: entry for entry in AUDIT_TABLE}
)


def audit_with_status(status: str) -> tuple[AuditEntry, ...]:
    """Every entry carrying ``status``, in audit order."""
    if status not in STATUSES:
        raise ValueError(f"unknown audit status {status!r}")
    return tuple(entry for entry in AUDIT_TABLE if entry.status == status)


def remaining_call_sites() -> tuple[AuditEntry, ...]:
    """Call sites that still decide a number from a literal 50 ohm."""
    return audit_with_status(HARD_CODED_REMAINING)


def require_reference_impedance(
    value, *, source: str = "reference impedance"
) -> float:
    """Return ``value`` as a positive, finite reference impedance.

    Raises:
        ReferenceImpedanceError: the value is not a number, not finite, or
            not positive.  A reference impedance of zero would make the
            transformation singular and a negative one has no physical
            meaning, so neither is ever coerced to a default.
    """
    if isinstance(value, bool):
        # bool is an int subclass, and True would otherwise pass as 1 ohm.
        raise ReferenceImpedanceError(
            f"{source} must be a number, not {value!r}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ReferenceImpedanceError(
            f"{source} must be a positive finite number, not {value!r}"
        ) from None
    if not math.isfinite(number) or number <= 0.0:
        raise ReferenceImpedanceError(
            f"{source} must be a positive finite number, not {value!r}"
        )
    return number


def format_reference_impedance(value) -> str:
    """Format a reference impedance for a Touchstone option line.

    An integral value is written without a decimal point (``50``, not
    ``50.0``) so the common case keeps the conventional spelling; anything
    else keeps its value (``75.5``).
    """
    number = require_reference_impedance(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def renormalization_blocker(
    channels: Iterable[str], *, equal_references: bool = False
) -> str | None:
    """Why ``channels`` cannot be renormalized, or ``None`` if they can.

    Only one-port reflection data may be renormalized: changing the
    reference of a transmission coefficient between unequal port
    impedances is a two-port operation, and doing it with the one-port
    formula would return a plausible but wrong number.

    This is the pre-flight of a *changing* reference.  Renormalizing to the
    reference the data already carries is a no-op that returns the samples
    untouched, transmission channels included, so a caller that can prove
    the target equals the current reference passes
    ``equal_references=True`` and gets ``None``.

    Args:
        channels: channel names (a mapping of name -> samples is accepted,
            since iterating it yields the names).
        equal_references: ``True`` when the target reference impedance is
            the one the data is already referenced to, making the
            operation an identity.

    Returns:
        A sentence naming the offending channels and the missing
        transform, or ``None`` when every channel is a reflection channel
        or when no renormalization would actually happen.
    """
    if equal_references:
        return None
    present = tuple(name for name in channels if name in TRANSMISSION_CHANNELS)
    if not present:
        return None
    return (
        "renormalizing " + ", ".join(present) + " between unequal port"
        " impedances needs the full wave-cascading two-port transform,"
        " which is not implemented; only reflection channels ("
        + ", ".join(REFLECTION_CHANNELS)
        + ") can be renormalized"
    )
