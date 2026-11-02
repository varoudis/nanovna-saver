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
"""Immutable measurement snapshots and their provenance (CORE-01).

The snapshot is the single shared contract between acquisition, charts,
touchesstone import/export, sessions, presets and reports.  It never
references Qt widgets or the mutable buffers used during a sweep.

Design rules encoded here:

* A snapshot is deeply immutable: channels are tuples of ``Datapoint``.
* An unavailable channel is *absent* from the channel mapping.  A channel is
  never represented by zero-valued samples, because a measured zero is a
  legitimate reading.
* Unavailable metadata is ``None`` or an explicit ``UNKNOWN_*`` sentinel.
* Raw and corrected samples are both retained together with the identity of
  the correction that produced the corrected samples.  Re-applying the same
  correction is therefore detectable (see ``CorrectionState.applied``).
"""

from __future__ import annotations

import hashlib
import itertools as it
import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .Processing import ProcessingConfig
from .RFTools import Datapoint

S11 = "S11"
S21 = "S21"
S12 = "S12"
S22 = "S22"

#: Canonical Touchstone field order used throughout the application.
SPARAMETERS: tuple[str, ...] = (S11, S21, S12, S22)

#: Parameters the application can ever acquire from a NanoVNA.
ACQUIRED_PARAMETERS: tuple[str, ...] = (S11, S21)

#: Bumped to 2 when processing provenance was added.  Reading a version 1
#: document is supported (a missing ``processing`` field means "no
#: processing was applied", which is exactly what version 1 meant).
SNAPSHOT_FORMAT_VERSION = 2

UNKNOWN_SOURCE = "unknown"
UNKNOWN_DEVICE = "unknown"


class CompletionState(Enum):
    """Whether the acquisition that produced a channel finished."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"

    @classmethod
    def parse(cls, value: str | "CompletionState") -> "CompletionState":
        if isinstance(value, CompletionState):
            return value
        return cls(value)


class CorrectionSource(Enum):
    """Who performed the correction stored in ``corrected``."""

    NONE = "none"
    APPLICATION = "application"
    DEVICE = "device"
    UNKNOWN = "unknown"


class CorrectionAlreadyAppliedError(RuntimeError):
    """Raised when the same correction would be applied twice."""


def _encode_float(value: float) -> float | str:
    """JSON-safe encoding of a float, including non-finite values."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _decode_float(value: float | str | int) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "nan":
            return math.nan
        if lowered in {"inf", "+inf", "infinity"}:
            return math.inf
        if lowered in {"-inf", "-infinity"}:
            return -math.inf
        return float(lowered)
    return float(value)


def _encode_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _decode_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class SweepInfo:
    """Sweep settings captured at acquisition time."""

    start: int
    end: int
    points: int
    segments: int
    mode: str = "SINGLE"
    averages: int = 1
    truncates: int = 0
    logarithmic: bool = False

    @classmethod
    def from_sweep(cls, sweep) -> "SweepInfo":
        """Build from a :class:`NanoVNASaver.Settings.Sweep.Sweep`."""
        props = sweep.properties
        averages = props.averages[0] if str(props.mode.name) == "AVERAGE" else 1
        return cls(
            start=sweep.start,
            end=sweep.end,
            points=sweep.points,
            segments=sweep.segments,
            mode=props.mode.name,
            averages=averages,
            truncates=props.averages[1],
            logarithmic=props.logarithmic,
        )

    def declared_frequencies(self) -> tuple[int, ...]:
        """The frequency grid this acquisition was configured to capture.

        Reconstructed from the recorded settings with the same arithmetic the
        acquisition itself uses (:class:`NanoVNASaver.Settings.Sweep.Sweep`),
        so segment joins round exactly as they did during the sweep and a
        duplicated join frequency is preserved rather than smoothed away.

        Returns an empty tuple when the settings cannot describe a grid
        (fewer than two units, an internally illegal span, or generation that
        divides by a degenerate point count).  Consumers must then fall back
        to a data-only policy: these settings carry no authority about the
        intended resolution.
        """
        from .Settings.Sweep import Properties, Sweep

        if self.points < 1 or self.segments < 1:
            return ()
        if self.points * self.segments < 2:
            return ()
        try:
            sweep = Sweep(
                self.start,
                self.end,
                self.points,
                self.segments,
                Properties(logarithmic=self.logarithmic),
            )
            # ``get_frequencies`` is a generator: its arithmetic (which calls
            # raise on a degenerate point count) runs here, inside the guard,
            # not at the call site.
            return tuple(sweep.get_frequencies())
        except (ValueError, ZeroDivisionError, OverflowError):
            return ()

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "points": self.points,
            "segments": self.segments,
            "mode": self.mode,
            "averages": self.averages,
            "truncates": self.truncates,
            "logarithmic": self.logarithmic,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SweepInfo":
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            points=int(data["points"]),
            segments=int(data["segments"]),
            mode=str(data.get("mode", "SINGLE")),
            averages=int(data.get("averages", 1)),
            truncates=int(data.get("truncates", 0)),
            logarithmic=bool(data.get("logarithmic", False)),
        )


@dataclass(frozen=True)
class CalibrationIdentity:
    """Identity of the calibration that is (or was) in effect."""

    identity: str = ""
    source: str = UNKNOWN_SOURCE
    notes: str = ""
    one_port: bool = False
    two_port: bool = False
    coverage_spans: tuple[tuple[int, int], ...] = ()
    correction_source: CorrectionSource = CorrectionSource.UNKNOWN
    #: The named kit a calibration was applied from, and its content
    #: identity.  A calibration typed by hand has neither, and a record that
    #: names the kit is what lets an auditor tell "the SMA kit" from "the
    #: same numbers someone typed in once".
    kit_name: str = ""
    kit_identity: str = ""
    #: When the standards were measured, as recorded with the calibration, so
    #: a run keeps its calibration's age even after a newer one is loaded.
    #: Empty means unknown.
    created: str = ""

    @property
    def available(self) -> bool:
        return self.one_port or self.two_port

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "source": self.source,
            "notes": self.notes,
            "one_port": self.one_port,
            "two_port": self.two_port,
            "coverage_spans": [list(span) for span in self.coverage_spans],
            "correction_source": self.correction_source.value,
            "kit_name": self.kit_name,
            "kit_identity": self.kit_identity,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CalibrationIdentity":
        return cls(
            identity=str(data.get("identity", "")),
            source=str(data.get("source", UNKNOWN_SOURCE)),
            notes=str(data.get("notes", "")),
            one_port=bool(data.get("one_port", False)),
            two_port=bool(data.get("two_port", False)),
            coverage_spans=tuple(
                (int(span[0]), int(span[1]))
                for span in data.get("coverage_spans", ())
            ),
            correction_source=CorrectionSource(
                data.get("correction_source", "unknown")
            ),
            kit_name=str(data.get("kit_name", "")),
            kit_identity=str(data.get("kit_identity", "")),
            created=str(data.get("created", "")),
        )


@dataclass(frozen=True)
class CorrectionState:
    """Provenance of the correction that produced the corrected samples."""

    applied: bool = False
    identity: str = ""
    source: CorrectionSource = CorrectionSource.NONE
    offset_delay: float = 0.0

    def guard(self, identity: str) -> None:
        """Refuse to apply a correction that is already in effect."""
        if self.applied and identity and identity == self.identity:
            raise CorrectionAlreadyAppliedError(
                f"correction {identity!r} is already applied to this snapshot"
            )

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "identity": self.identity,
            "source": self.source.value,
            "offset_delay": _encode_float(self.offset_delay),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CorrectionState":
        return cls(
            applied=bool(data.get("applied", False)),
            identity=str(data.get("identity", "")),
            source=CorrectionSource(data.get("source", "none")),
            offset_delay=_decode_float(data.get("offset_delay", 0.0)),
        )


@dataclass(frozen=True)
class Provenance:
    """Where a snapshot came from and how trustworthy it is."""

    source: str = UNKNOWN_SOURCE
    simulated: bool = False
    device: str = UNKNOWN_DEVICE
    app_version: str = ""
    created: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    sweep: SweepInfo | None = None
    calibration: CalibrationIdentity | None = None
    completion: CompletionState = CompletionState.COMPLETE
    notes: str = ""
    #: Set when the data was generated by a named simulator scenario.
    simulator_scenario: str | None = None
    #: Post-calibration processing applied to ``corrected`` (DSP-01).
    #:
    #: ``None`` means no processing was applied, which is what every
    #: version-1 document and every unprocessed acquisition means.
    processing: ProcessingConfig | None = None
    #: Source trace this snapshot was derived from, when it is not an
    #: acquisition of its own (ANALYSIS-02).  An empty string means the
    #: snapshot *is* the acquisition.
    derived_from: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "simulated": self.simulated,
            "device": self.device,
            "app_version": self.app_version,
            "created": _encode_time(self.created),
            "sweep": self.sweep.to_dict() if self.sweep else None,
            "calibration": (
                self.calibration.to_dict() if self.calibration else None
            ),
            "completion": self.completion.value,
            "notes": self.notes,
            "simulator_scenario": self.simulator_scenario,
            "processing": (
                self.processing.to_dict() if self.processing else None
            ),
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Provenance":
        sweep = data.get("sweep")
        calibration = data.get("calibration")
        return cls(
            source=str(data.get("source", UNKNOWN_SOURCE)),
            simulated=bool(data.get("simulated", False)),
            device=str(data.get("device", UNKNOWN_DEVICE)),
            app_version=str(data.get("app_version", "")),
            created=_decode_time(data["created"])
            if data.get("created")
            else datetime.now(timezone.utc),
            sweep=SweepInfo.from_dict(sweep) if sweep else None,
            calibration=(
                CalibrationIdentity.from_dict(calibration)
                if calibration
                else None
            ),
            completion=CompletionState.parse(
                data.get("completion", CompletionState.COMPLETE.value)
            ),
            notes=str(data.get("notes", "")),
            simulator_scenario=data.get("simulator_scenario"),
            processing=(
                ProcessingConfig.from_dict(data["processing"])
                if data.get("processing")
                else None
            ),
            derived_from=str(data.get("derived_from", "")),
        )


def _freeze_channel(
    data: Iterable[Datapoint] | None,
) -> tuple[Datapoint, ...]:
    if not data:
        return ()
    frozen = []
    for item in data:
        frozen.append(item if isinstance(item, Datapoint) else Datapoint(*item))
    return tuple(frozen)


def compute_calibration_identity(calibration) -> str:
    """Stable digest of the numerically relevant calibration content."""
    dataset = calibration.dataset
    digest = hashlib.sha256()
    for freq in dataset.frequencies():
        cal = dataset.get(freq)
        digest.update(f"{freq}:".encode("ascii"))
        for name in (
            "short",
            "open",
            "load",
            "through",
            "thrurefl",
            "isolation",
        ):
            value = getattr(cal, name, complex(0.0, 0.0))
            digest.update(f"{name}={value.real!r},{value.imag!r};".encode())
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class MeasurementSnapshot:
    """An immutable, self-describing measurement.

    ``raw`` holds samples exactly as read from the device (or file) and
    ``corrected`` holds the same channels after calibration and offset-delay
    correction.  Both mappings only contain channels that actually exist.
    """

    id: str
    name: str
    timestamp: datetime
    reference_impedance: float
    raw: Mapping[str, tuple[Datapoint, ...]]
    corrected: Mapping[str, tuple[Datapoint, ...]]
    provenance: Provenance
    correction: CorrectionState = field(default_factory=CorrectionState)
    frequencies: tuple[int, ...] = ()
    format_version: int = SNAPSHOT_FORMAT_VERSION

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.raw, MappingProxyType):
            object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))
        if not isinstance(self.corrected, MappingProxyType):
            object.__setattr__(
                self, "corrected", MappingProxyType(dict(self.corrected))
            )
        if not self.frequencies and self.corrected:
            first = next(iter(self.corrected.values()))
            object.__setattr__(
                self, "frequencies", tuple(dp.freq for dp in first)
            )

    @property
    def available(self) -> tuple[str, ...]:
        """S-parameters present in the corrected mapping, in canonical order."""
        return tuple(p for p in SPARAMETERS if p in self.corrected)

    @property
    def available_raw(self) -> tuple[str, ...]:
        return tuple(p for p in SPARAMETERS if p in self.raw)

    @property
    def simulated(self) -> bool:
        return self.provenance.simulated

    # ------------------------------------------------------------------
    # Processing (DSP-01)
    # ------------------------------------------------------------------
    def with_processing(
        self,
        config: "ProcessingConfig | None",
        corrected: Mapping[str, Iterable[Datapoint]] | None = None,
        *,
        reference_impedance: float | None = None,
    ) -> "MeasurementSnapshot":
        """Return a copy carrying ``config`` as its processing provenance.

        ``corrected`` supplies the processed channels; ``raw`` and the
        calibration identity are carried over untouched, so the unprocessed
        acquisition stays available.  A configuration that changes the
        reference impedance must pass the new one explicitly: the number and
        the statement about it can never disagree.
        """
        provenance = replace(
            self.provenance,
            processing=config if config and not config.is_identity else None,
        )
        changes: dict = {"provenance": provenance}
        if corrected is not None:
            changes["corrected"] = {
                key: _freeze_channel(value)
                for key, value in corrected.items()
                if value
            }
        if reference_impedance is not None:
            changes["reference_impedance"] = float(reference_impedance)
        return replace(self, **changes)

    def derived(
        self,
        name: str,
        corrected: Mapping[str, Iterable[Datapoint]],
        *,
        source_id: str | None = None,
        notes: str = "",
    ) -> "MeasurementSnapshot":
        """Return a new snapshot computed from this one (ANALYSIS-02).

        The result keeps the source's provenance so its origin is auditable,
        records ``derived_from`` as the source snapshot id, and carries a
        fresh id of its own.  ``raw`` is empty: a computed trace is not a
        measurement, so it must never be exportable as if it had been read
        from a device.
        """
        provenance = replace(
            self.provenance,
            derived_from=source_id or self.id,
            notes="\n".join(filter(None, (self.provenance.notes, notes))),
        )
        return MeasurementSnapshot(
            id=new_snapshot_id(),
            name=name,
            timestamp=datetime.now(timezone.utc),
            reference_impedance=self.reference_impedance,
            raw={},
            corrected={
                key: _freeze_channel(value)
                for key, value in corrected.items()
                if value
            },
            provenance=provenance,
            correction=self.correction,
        )

    def processing_description(self) -> str:
        """Human-readable processing description for UI, reports and exports."""
        config = self.provenance.processing
        return "Raw (no processing)" if config is None else config.describe()

    @property
    def is_derived(self) -> bool:
        return bool(self.provenance.derived_from)

    def channel(
        self, parameter: str, corrected: bool = True
    ) -> tuple[Datapoint, ...] | None:
        """Return a channel, or ``None`` when that channel is unavailable."""
        source = self.corrected if corrected else self.raw
        return source.get(parameter)

    def has(self, parameter: str) -> bool:
        return parameter in self.corrected

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    def with_name(self, name: str) -> "MeasurementSnapshot":
        return replace(self, name=name)

    def with_provenance(self, provenance: Provenance) -> "MeasurementSnapshot":
        return replace(self, provenance=provenance)

    def mark_corrected(
        self,
        identity: str,
        corrected: Mapping[str, Iterable[Datapoint]],
        source: CorrectionSource = CorrectionSource.APPLICATION,
        offset_delay: float = 0.0,
    ) -> "MeasurementSnapshot":
        """Return a copy whose corrected channels are replaced.

        Raises :class:`CorrectionAlreadyAppliedError` when the identical
        correction is already present, which is the guard against applying
        calibration twice after loading a session.
        """
        self.correction.guard(identity)
        return replace(
            self,
            corrected={
                key: _freeze_channel(value)
                for key, value in corrected.items()
                if value is not None
            },
            correction=CorrectionState(
                applied=bool(identity),
                identity=identity,
                source=source if identity else CorrectionSource.NONE,
                offset_delay=offset_delay,
            ),
        )

    def validate(self) -> None:
        """Check internal consistency; raise ``ValueError`` when broken."""
        for mapping_name in ("raw", "corrected"):
            mapping = getattr(self, mapping_name)
            for parameter, points in mapping.items():
                if parameter not in SPARAMETERS:
                    raise ValueError(
                        f"unsupported parameter {parameter!r} in {mapping_name}"
                    )
                if not points:
                    raise ValueError(
                        f"empty channel {parameter!r} in {mapping_name};"
                        " drop the key instead"
                    )
                freqs = [dp.freq for dp in points]
                if any(b < a for a, b in it.pairwise(freqs)):
                    raise ValueError(
                        f"channel {parameter!r} in {mapping_name}"
                        " is not frequency ordered"
                    )
        if self.reference_impedance <= 0:
            raise ValueError("reference impedance must be positive")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        def channels(mapping: Mapping[str, tuple[Datapoint, ...]]) -> dict:
            return {
                parameter: [
                    [dp.freq, _encode_float(dp.re), _encode_float(dp.im)]
                    for dp in points
                ]
                for parameter, points in mapping.items()
            }

        return {
            "format_version": self.format_version,
            "id": self.id,
            "name": self.name,
            "timestamp": _encode_time(self.timestamp),
            "reference_impedance": _encode_float(self.reference_impedance),
            "frequencies": list(self.frequencies),
            "raw": channels(self.raw),
            "corrected": channels(self.corrected),
            "provenance": self.provenance.to_dict(),
            "correction": self.correction.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "MeasurementSnapshot":
        version = int(data.get("format_version", SNAPSHOT_FORMAT_VERSION))
        if version > SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"snapshot format version {version} is newer than supported"
                f" version {SNAPSHOT_FORMAT_VERSION}"
            )

        def channels(mapping: Mapping) -> dict:
            return {
                str(parameter): _freeze_channel(
                    Datapoint(
                        int(row[0]),
                        _decode_float(row[1]),
                        _decode_float(row[2]),
                    )
                    for row in rows
                )
                for parameter, rows in mapping.items()
            }

        snapshot = cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            timestamp=_decode_time(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            reference_impedance=_decode_float(
                data.get("reference_impedance", 50.0)
            ),
            raw=channels(data.get("raw", {})),
            corrected=channels(data.get("corrected", {})),
            provenance=Provenance.from_dict(data.get("provenance", {})),
            correction=CorrectionState.from_dict(data.get("correction", {})),
            frequencies=tuple(int(f) for f in data.get("frequencies", ())),
            format_version=version,
        )
        return snapshot


# ----------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------
def new_snapshot_id() -> str:
    return uuid.uuid4().hex


def snapshot_from_touchstone(  # noqa: PLR0913
    touchstone,
    *,
    name: str = "",
    source: str = "",
    simulated: bool = False,
    device: str = UNKNOWN_DEVICE,
    app_version: str = "",
    reference_impedance: float | None = None,
    completion: CompletionState = CompletionState.COMPLETE,
    simulator_scenario: str | None = None,
) -> MeasurementSnapshot:
    """Adapt a loaded :class:`Touchstone` into an immutable snapshot.

    Only channels carrying samples are retained, so an S1P file keeps S11 and
    an S2P file keeps whichever of the four channels were present.
    """
    raw: dict[str, tuple[Datapoint, ...]] = {}
    for parameter in SPARAMETERS:
        points = _freeze_channel(getattr(touchstone, parameter.lower()))
        if points:
            raw[parameter] = points

    impedance = (
        reference_impedance
        if reference_impedance is not None
        else float(touchstone.opts.resistance)
    )
    provenance = Provenance(
        source=source or touchstone.filename or UNKNOWN_SOURCE,
        simulated=simulated,
        device=device,
        app_version=app_version,
        sweep=None,
        completion=completion,
        notes="\n".join(touchstone.comments),
        simulator_scenario=simulator_scenario,
    )
    snapshot = MeasurementSnapshot(
        id=new_snapshot_id(),
        name=name or (source or touchstone.filename or "measurement"),
        timestamp=datetime.now(timezone.utc),
        reference_impedance=impedance,
        raw=raw,
        corrected=dict(raw),
        provenance=provenance,
    )
    return snapshot


def snapshot_from_datapoints(  # noqa: PLR0913
    s11: Iterable[Datapoint] | None,
    s21: Iterable[Datapoint] | None,
    *,
    name: str = "",
    source: str = "",
    simulated: bool = False,
    device: str = UNKNOWN_DEVICE,
    app_version: str = "",
    reference_impedance: float = 50.0,
    sweep=None,
    calibration=None,
    correction: CorrectionState | None = None,
    completion: CompletionState = CompletionState.COMPLETE,
    simulator_scenario: str | None = None,
    notes: str = "",
    raw11: Iterable[Datapoint] | None = None,
    raw21: Iterable[Datapoint] | None = None,
) -> MeasurementSnapshot:
    """Build a snapshot from the application's live acquisition buffers.

    ``notes`` carries acquisition-level provenance that has no structural
    field of its own, such as the Touchstone file a replayed sweep came
    from.
    """

    corrected: dict[str, tuple[Datapoint, ...]] = {}
    if s11:
        corrected[S11] = _freeze_channel(s11)
    if s21:
        corrected[S21] = _freeze_channel(s21)

    raw: dict[str, tuple[Datapoint, ...]] = {}
    if raw11:
        raw[S11] = _freeze_channel(raw11)
    elif S11 in corrected:
        raw[S11] = corrected[S11]
    if raw21:
        raw[S21] = _freeze_channel(raw21)
    elif S21 in corrected:
        raw[S21] = corrected[S21]

    cal_identity = None
    if calibration is not None:
        cal_identity = CalibrationIdentity(
            identity=compute_calibration_identity(calibration),
            source=str(getattr(calibration, "source", UNKNOWN_SOURCE)),
            notes="\n".join(getattr(calibration, "notes", []) or []),
            one_port=bool(calibration.isValid1Port()),
            two_port=bool(calibration.isValid2Port()),
            coverage_spans=(
                (
                    (
                        calibration.dataset.freq_min(),
                        calibration.dataset.freq_max(),
                    ),
                )
                if calibration.size()
                else ()
            ),
            correction_source=CorrectionSource.APPLICATION,
            kit_name=str(getattr(calibration, "kit_name", "") or ""),
            kit_identity=str(getattr(calibration, "kit_identity", "") or ""),
            created=str(
                getattr(getattr(calibration, "metadata", None), "created", "")
            ),
        )

    provenance = Provenance(
        source=source or UNKNOWN_SOURCE,
        simulated=simulated,
        device=device,
        app_version=app_version,
        sweep=SweepInfo.from_sweep(sweep) if sweep is not None else None,
        calibration=cal_identity,
        completion=completion,
        notes=notes,
        simulator_scenario=simulator_scenario,
    )
    return MeasurementSnapshot(
        id=new_snapshot_id(),
        name=name or source or "measurement",
        timestamp=datetime.now(timezone.utc),
        reference_impedance=reference_impedance,
        raw=raw,
        corrected=corrected,
        provenance=provenance,
        correction=correction or CorrectionState(),
    )


def snapshot_to_touchstone(
    snapshot: MeasurementSnapshot, corrected: bool = True
):
    """Adapt a snapshot back to a ``Touchstone`` for export."""
    from .Touchstone import Touchstone  # local import avoids a cycle

    touchstone = Touchstone()
    touchstone.opts.resistance = snapshot.reference_impedance
    source = snapshot.corrected if corrected else snapshot.raw
    for parameter in SPARAMETERS:
        points = source.get(parameter)
        if points:
            setattr(touchstone, parameter.lower(), list(points))
    comments = []
    if snapshot.provenance.notes:
        comments.extend(snapshot.provenance.notes.splitlines())
    comments.append(
        "! simulated measurement (scenario:"
        f" {snapshot.provenance.simulator_scenario or 'n/a'})"
        if snapshot.simulated
        else f"! source: {snapshot.provenance.source}"
    )
    touchstone.comments = comments
    return touchstone


def missing_parameters(snapshot: MeasurementSnapshot) -> tuple[str, ...]:
    """S-parameters the snapshot does not contain, in canonical order."""
    return tuple(p for p in SPARAMETERS if p not in snapshot.corrected)
