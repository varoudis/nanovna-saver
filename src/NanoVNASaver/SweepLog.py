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
"""Per-sweep verdict logging (NEXT-05).

An audit trail of what was measured and what was concluded, one line per
completed run.  It exists so a log can answer "was this unit inside limits at
14 MHz on Tuesday, and against which calibration" without re-running
anything — which means a record must be frozen at the end of the run, not
assembled later from live UI state that has since changed.

Design rules
------------
* **One record per completed run**, written once.  A run identity that is
  logged twice is refused rather than appended twice, so a duplicated
  completion signal cannot inflate a production log.
* **Cancellation and failure are separate kinds of record.**  Neither is a
  measurement, and neither may be read as a pass.  A cancelled run records
  ``cancelled`` with no verdict; a failed run records ``failed`` with the
  reason.
* **The limit identity is captured at the run boundary.**  Editing the
  catalogue afterwards cannot relabel a record that was already written.
* **A logging failure is reported, not swallowed.**  The measurement is
  unaffected, but the caller learns that the audit trail has a hole.
* **Appending never blocks acquisition**: the caller owns the file handle
  and decides when to flush.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .Limits import LimitEvaluation, Verdict
from .Measurement import MeasurementSnapshot

__all__ = [
    "CSV_FIELDS",
    "LogFormat",
    "LogRecord",
    "LogWriteError",
    "SweepLog",
    "cancelled_record",
    "failed_record",
    "record_for",
    "summarise",
]

#: Columns of the CSV form, in order.  Frozen so an old file stays readable.
CSV_FIELDS = (
    "timestamp",
    "run_id",
    "trace_id",
    "trace_name",
    "dut",
    "serial",
    "source",
    "device",
    "simulated",
    "calibration_identity",
    "processing_identity",
    "reference_impedance",
    "limit_identity",
    "outcome",
    "verdict",
    "rules_passed",
    "rules_failed",
    "rules_unavailable",
    "reason",
)


class LogFormat(Enum):
    CSV = "csv"
    JSONL = "jsonl"


class LogWriteError(OSError):
    """A record could not be written; the measurement is unaffected."""


class Outcome(Enum):
    """What kind of event the record describes."""

    MEASURED = "measured"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class LogRecord:
    """One immutable log line."""

    timestamp: str
    run_id: str
    outcome: Outcome
    trace_id: str = ""
    trace_name: str = ""
    dut: str = ""
    serial: str = ""
    source: str = ""
    device: str = ""
    simulated: bool = False
    calibration_identity: str = ""
    processing_identity: str = ""
    reference_impedance: float = 50.0
    limit_identity: str = ""
    verdict: str = ""
    rules_passed: int = 0
    rules_failed: int = 0
    rules_unavailable: int = 0
    reason: str = ""

    @property
    def is_pass(self) -> bool:
        """True only for a measured run with no failure and no gap."""
        return (
            self.outcome is Outcome.MEASURED
            and self.verdict == Verdict.PASS.value
            and self.rules_failed == 0
            and self.rules_unavailable == 0
            and self.rules_passed > 0
        )

    def to_row(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "dut": self.dut,
            "serial": self.serial,
            "source": self.source,
            "device": self.device,
            "simulated": self.simulated,
            "calibration_identity": self.calibration_identity,
            "processing_identity": self.processing_identity,
            "reference_impedance": self.reference_impedance,
            "limit_identity": self.limit_identity,
            "outcome": self.outcome.value,
            "verdict": self.verdict,
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "rules_unavailable": self.rules_unavailable,
            "reason": self.reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_row(), sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_for(  # noqa: PLR0913
    snapshot: MeasurementSnapshot,
    *,
    run_id: str,
    evaluations: Sequence[LimitEvaluation] = (),
    limit_identity: str = "",
    trace_id: str = "",
    trace_name: str = "",
    dut: str = "",
    serial: str = "",
    processing_identity: str = "",
    timestamp: str | None = None,
) -> LogRecord:
    """Build the record for one measured run.

    The evaluation outcomes are counted here, once, from the immutable
    evaluations; a later edit to the catalogue cannot change this record.
    """
    passed = sum(1 for item in evaluations if item.verdict is Verdict.PASS)
    failed = sum(1 for item in evaluations if item.verdict is Verdict.FAIL)
    unavailable = sum(
        1 for item in evaluations if item.verdict is Verdict.UNAVAILABLE
    )
    if failed:
        verdict = Verdict.FAIL
    elif unavailable:
        verdict = Verdict.UNAVAILABLE
    elif passed:
        verdict = Verdict.PASS
    else:
        # No rules evaluated is not a pass: nothing was checked.
        verdict = Verdict.UNAVAILABLE
    reason = ""
    if failed:
        reason = next(
            item.describe()
            for item in evaluations
            if item.verdict is Verdict.FAIL
        )
    elif unavailable:
        reason = next(
            item.describe()
            for item in evaluations
            if item.verdict is Verdict.UNAVAILABLE
        )
    elif not passed:
        reason = "no limit rules were evaluated"
    calibration = snapshot.provenance.calibration
    return LogRecord(
        timestamp=timestamp or _now(),
        run_id=run_id,
        outcome=Outcome.MEASURED,
        trace_id=trace_id,
        trace_name=trace_name,
        dut=dut,
        serial=serial,
        source=snapshot.provenance.source,
        device=snapshot.provenance.device,
        simulated=snapshot.simulated,
        calibration_identity=(
            calibration.identity if calibration is not None else ""
        ),
        processing_identity=processing_identity,
        reference_impedance=snapshot.reference_impedance,
        limit_identity=limit_identity,
        verdict=verdict.value,
        rules_passed=passed,
        rules_failed=failed,
        rules_unavailable=unavailable,
        reason=reason,
    )


def cancelled_record(
    *,
    run_id: str,
    reason: str = "cancelled before completion",
    **fields,
) -> LogRecord:
    """A record for a run that was cancelled: never a measurement."""
    return LogRecord(
        timestamp=fields.pop("timestamp", None) or _now(),
        run_id=run_id,
        outcome=Outcome.CANCELLED,
        verdict="",
        reason=reason,
        **fields,
    )


def failed_record(*, run_id: str, reason: str, **fields) -> LogRecord:
    """A record for a run that failed: never a measurement."""
    return LogRecord(
        timestamp=fields.pop("timestamp", None) or _now(),
        run_id=run_id,
        outcome=Outcome.FAILED,
        verdict="",
        reason=reason,
        **fields,
    )


class SweepLog:
    """Append-only record file, with schema checking and deduplication.

    The log opens lazily on first write, so enabling logging does not create
    an empty file for a run that never completes.
    """

    def __init__(
        self,
        path: str,
        *,
        log_format: LogFormat = LogFormat.CSV,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.path = path
        self.format = log_format
        self.metadata = dict(metadata or {})
        self._written: set[str] = set()
        self._validated = False
        self.records: list[LogRecord] = []

    # -- schema --------------------------------------------------------
    def _validate_existing(self) -> None:
        """Refuse to append to a file whose header does not match."""
        if self._validated or not os.path.exists(self.path):
            self._validated = True
            return
        if self.format is not LogFormat.CSV:
            self._validated = True
            return
        try:
            with open(self.path, encoding="utf-8", newline="") as stream:
                header = next(csv.reader(stream), None)
        except OSError as error:
            raise LogWriteError(f"cannot read {self.path}: {error}") from error
        if header is None:
            # An empty file is a fresh log.
            self._validated = True
            return
        if tuple(header) != CSV_FIELDS:
            raise LogWriteError(
                f"{self.path} has a different column layout; refusing to"
                " append a record that would be misread. Move the existing"
                " file aside to start a new log."
            )
        self._validated = True

    # -- writing -------------------------------------------------------
    def append(self, record: LogRecord) -> None:
        """Write ``record`` once; a duplicate run id is an error."""
        if record.run_id in self._written:
            raise LogWriteError(
                f"run {record.run_id!r} is already in this log; a run is"
                " recorded once"
            )
        # Mark it before the write: if the write fails after the bytes land,
        # a retry must not produce a second copy.
        self._written.add(record.run_id)
        try:
            self._validate_existing()
            if self.format is LogFormat.CSV:
                self._append_csv(record)
            else:
                self._append_jsonl(record)
        except OSError as error:
            raise LogWriteError(f"cannot write {self.path}: {error}") from error
        self.records.append(record)

    def _append_csv(self, record: LogRecord) -> None:
        new_file = (
            not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        )
        # Newline handling is explicit: csv needs ``newline=""`` to avoid
        # doubled line endings on Windows.
        with open(self.path, "a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(record.to_row())
            stream.flush()

    def _append_jsonl(self, record: LogRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as stream:
            if self.metadata and os.path.getsize(self.path) == 0:
                stream.write(json.dumps(self.metadata, sort_keys=True) + "\n")
            stream.write(record.to_json() + "\n")
            stream.flush()

    # -- reading -------------------------------------------------------
    def read(self) -> list[dict[str, object]]:
        """Read the file back, for verification and export."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8", newline="") as stream:
            if self.format is LogFormat.CSV:
                return list(csv.DictReader(stream))
            rows = []
            for raw in stream:
                text = raw.strip()
                if not text:
                    continue
                document = json.loads(text)
                if "run_id" in document:
                    rows.append(document)
            return rows


def unique_records(records: Iterable[LogRecord]) -> list[LogRecord]:
    """Deduplicate by run id, keeping the first record of each run."""
    seen: set[str] = set()
    out: list[LogRecord] = []
    for record in records:
        if record.run_id in seen:
            continue
        seen.add(record.run_id)
        out.append(record)
    return out


def summarise(records: Iterable[LogRecord]) -> dict[str, int]:
    """Count records by outcome, so a production run can report a total."""
    counts = {
        "records": 0,
        "passed": 0,
        "failed": 0,
        "unavailable": 0,
        "cancelled": 0,
        "errors": 0,
    }
    for record in records:
        counts["records"] += 1
        if record.outcome is Outcome.CANCELLED:
            counts["cancelled"] += 1
        elif record.outcome is Outcome.FAILED:
            counts["errors"] += 1
        elif record.is_pass:
            counts["passed"] += 1
        elif record.verdict == Verdict.FAIL.value:
            counts["failed"] += 1
        else:
            counts["unavailable"] += 1
    return counts
