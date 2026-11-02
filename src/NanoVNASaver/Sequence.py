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
"""Versioned, auditable test sequences (AUTO-03).

A sequence is a JSON document listing actions for one device under test: take
a sweep, check it, save it, write a report.  A production test is exactly
this, repeated per unit, and the value comes from being able to say
afterwards what ran, in what order, against which build, and what happened at
each step.

Design rules
------------
* **Strict schema.**  An unknown key is an error, not a warning.  A typo in
  ``"stop"`` silently ignored is a sequence that measures the wrong band and
  still reports success.
* **A sequence is never reported as passed when it is incomplete.**  A step
  that is unavailable, or that stopped the run, leaves the whole sequence
  unpassed, and the record says which step and why.
* **Paths cannot escape.**  Every output path is resolved against the
  sequence file's directory and refused if it lands outside the output
  directory, so a substituted unit name containing ``../`` cannot write over
  unrelated files.
* **Substitutions are a closed set.**  ``{serial}``, ``{date}`` and
  ``{name}`` are the only ones; anything else is an error rather than a
  literal brace in a filename.
* **No physical action is retried silently.**  A failed sweep is a failed
  step; the run stops or continues exactly as the document says, and the
  record shows it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable, Mapping

__all__ = [
    "ActionKind",
    "SequenceError",
    "SequenceFormatError",
    "SequenceResult",
    "SequenceStep",
    "StepResult",
    "StopPolicy",
    "Substitutions",
    "TestSequence",
    "load_sequence",
    "parse_sequence",
    "run_sequence",
    "sequence_identity",
]

SEQUENCE_FORMAT = "nanovna-saver-sequence"
SEQUENCE_VERSION = 1

#: ``fail``/``unavailable`` are accepted on every action, because every
#: action can fail and the operator's choice of what to do about it must not
#: depend on which step it was.
_POLICY_KEYS = frozenset({"fail", "unavailable"})

_ACTION_KEYS: dict[str, frozenset[str]] = {
    "sweep": frozenset(
        {"action", "start", "stop", "points", "average", "name", "save"}
    )
    | _POLICY_KEYS,
    "check": frozenset({"action", "limits"}) | _POLICY_KEYS,
    "save": frozenset({"action", "path"}) | _POLICY_KEYS,
    "report": frozenset({"action", "path", "title"}) | _POLICY_KEYS,
    "log": frozenset({"action", "message"}) | _POLICY_KEYS,
    "prompt": frozenset({"action", "message", "headless"}) | _POLICY_KEYS,
}

_TOP_KEYS = frozenset(
    {"format", "version", "name", "description", "substitutions", "steps"}
)

_SUBSTITUTION = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class SequenceError(Exception):
    """Base class for sequence problems."""


class SequenceFormatError(SequenceError):
    """The document does not match the schema; the message says where."""


class ActionKind(Enum):
    SWEEP = "sweep"
    CHECK = "check"
    SAVE = "save"
    REPORT = "report"
    LOG = "log"
    PROMPT = "prompt"


class StopPolicy(Enum):
    """What to do when a step does not succeed."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True)
class Substitutions:
    """The closed set of values that may appear in a path or message."""

    serial: str = ""
    name: str = ""
    date: str = field(default_factory=lambda: date.today().isoformat())

    def values(self) -> dict[str, str]:
        return {"serial": self.serial, "name": self.name, "date": self.date}

    def apply(self, text: str, where: str) -> str:
        """Expand ``{serial}``/``{date}``/``{name}``, rejecting others."""
        unknown = sorted(
            {
                match.group(1)
                for match in _SUBSTITUTION.finditer(text)
                if match.group(1) not in self.values()
            }
        )
        if unknown:
            raise SequenceFormatError(
                f"{where}: unknown substitution"
                f" {{{unknown[0]}}}; allowed are"
                f" {', '.join(sorted(self.values()))}"
            )
        return _SUBSTITUTION.sub(
            lambda match: self.values()[match.group(1)], text
        )


@dataclass(frozen=True)
class SequenceStep:
    """One validated action."""

    kind: ActionKind
    index: int
    on_failure: StopPolicy = StopPolicy.STOP
    on_unavailable: StopPolicy = StopPolicy.STOP
    fields: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        detail = ", ".join(
            f"{key}={value}"
            for key, value in self.fields.items()
            if value not in ("", None)
        )
        return f"{self.index + 1}. {self.kind.value}" + (
            f" ({detail})" if detail else ""
        )


@dataclass(frozen=True)
class TestSequence:
    """A parsed, validated sequence document."""

    name: str
    description: str
    steps: tuple[SequenceStep, ...]
    substitutions: Substitutions
    source: str = ""
    version: int = SEQUENCE_VERSION

    def identity(self) -> str:
        """A stable digest of the document, so a record can name it."""
        return sequence_identity(self)


class StepStatus(Enum):
    OK = "ok"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    index: int
    kind: ActionKind
    status: StepStatus
    detail: str = ""
    artifact: str = ""

    @property
    def success(self) -> bool:
        return self.status is StepStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.index + 1,
            "action": self.kind.value,
            "status": self.status.value,
            "detail": self.detail,
            "artifact": self.artifact,
        }


@dataclass
class SequenceResult:
    """The auditable record of one run."""

    sequence: TestSequence
    serial: str
    steps: list[StepResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stopped_at: int | None = None

    @property
    def completed(self) -> bool:
        return self.stopped_at is None and len(self.steps) == len(
            self.sequence.steps
        )

    @property
    def passed(self) -> bool:
        """True only when every step ran and every step succeeded."""
        return self.completed and all(step.success for step in self.steps)

    def summary(self) -> dict[str, int]:
        counts = {status.value: 0 for status in StepStatus}
        for step in self.steps:
            counts[step.status.value] += 1
        return counts

    def describe(self) -> str:
        state = "PASS" if self.passed else "NOT PASSED"
        counts = self.summary()
        return (
            f"{state}: {self.sequence.name or 'sequence'}"
            f" for serial {self.serial or 'unset'}"
            f" ({counts['ok']} ok, {counts['failed']} failed,"
            f" {counts['unavailable']} unavailable,"
            f" {counts['skipped']} skipped)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SEQUENCE_FORMAT,
            "version": self.sequence.version,
            "sequence": {
                "name": self.sequence.name,
                "description": self.sequence.description,
                "identity": self.sequence.identity(),
                "steps": len(self.sequence.steps),
            },
            "serial": self.serial,
            "metadata": self.metadata,
            "passed": self.passed,
            "completed": self.completed,
            "stopped_at": self.stopped_at,
            "summary": self.summary(),
            "steps": [step.to_dict() for step in self.steps],
        }


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SequenceFormatError(message)


def _policy(value: Any, where: str, key: str) -> StopPolicy:
    if value is None:
        return StopPolicy.STOP
    try:
        return StopPolicy(str(value))
    except ValueError as exc:
        raise SequenceFormatError(
            f"{where}: {key} must be 'continue' or 'stop', not {value!r}"
        ) from exc


def _step(document: Any, index: int) -> SequenceStep:
    where = f"step {index + 1}"
    _require(isinstance(document, Mapping), f"{where}: must be an object")
    action = document.get("action")
    _require(
        isinstance(action, str) and action in _ACTION_KEYS,
        f"{where}: action must be one of"
        f" {', '.join(sorted(_ACTION_KEYS))}, not {action!r}",
    )
    allowed = _ACTION_KEYS[action]
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise SequenceFormatError(
            f"{where}: unknown key {unknown[0]!r} for action {action!r};"
            f" allowed are {', '.join(sorted(allowed))}"
        )
    on_failure = _policy(document.get("fail"), where, "fail")
    on_unavailable = _policy(document.get("unavailable"), where, "unavailable")
    fields = {key: value for key, value in document.items() if key != "action"}
    if action == "sweep":
        # A sweep cannot proceed without a band; a sequence that omitted one
        # must be rejected when it is read, not when the instrument is open.
        for key in ("start", "stop"):
            _require(
                isinstance(fields.get(key), str),
                f"{where}: sweep needs a {key} string such as '14.0M'",
            )
        if "points" in fields:
            _require(
                isinstance(fields["points"], int) and fields["points"] >= 2,
                f"{where}: points must be an integer of at least two",
            )
    if action == "check":
        _require(
            isinstance(fields.get("limits"), str),
            f"{where}: check needs a limits path",
        )
    if action in ("save", "report"):
        _require(
            isinstance(fields.get("path"), str) and fields["path"],
            f"{where}: {action} needs a path",
        )
    if action == "prompt":
        fallback = fields.get("headless", "stop")
        _require(
            fallback in ("stop", "continue", "skip"),
            f"{where}: headless must be 'stop', 'continue' or 'skip',"
            f" not {fallback!r}",
        )
    return SequenceStep(
        kind=ActionKind(action),
        index=index,
        on_failure=on_failure,
        on_unavailable=on_unavailable,
        fields=fields,
    )


def parse_sequence(
    text: str, *, source: str = "", substitutions: Substitutions | None = None
) -> TestSequence:
    """Parse and validate a sequence document."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise SequenceFormatError(
            f"invalid JSON at line {error.lineno}: {error.msg}"
        ) from error
    _require(isinstance(document, Mapping), "the document must be an object")
    unknown = sorted(set(document) - _TOP_KEYS)
    if unknown:
        # ``_require`` would evaluate its message eagerly, indexing a list
        # that is empty on the success path; branch before building it.
        raise SequenceFormatError(
            f"unknown top-level key {unknown[0]!r}; allowed are"
            f" {', '.join(sorted(_TOP_KEYS))}"
        )
    _require(
        document.get("format") == SEQUENCE_FORMAT,
        f"format must be {SEQUENCE_FORMAT!r}",
    )
    version = document.get("version")
    _require(
        version == SEQUENCE_VERSION,
        f"unsupported sequence version {version!r};"
        f" this build reads version {SEQUENCE_VERSION}",
    )
    steps_document = document.get("steps")
    _require(
        isinstance(steps_document, list) and steps_document,
        "steps must be a non-empty list",
    )
    steps = tuple(
        _step(entry, index) for index, entry in enumerate(steps_document)
    )
    supplied = substitutions or Substitutions()
    declared = document.get("substitutions") or {}
    _require(
        isinstance(declared, Mapping),
        "substitutions must be an object",
    )
    unknown = sorted(set(declared) - set(supplied.values()))
    if unknown:
        raise SequenceFormatError(f"unknown substitution {unknown[0]!r}")
    # Values declared in the document become defaults, so a sequence can
    # carry a unit name but a caller can still override the serial.
    merged = supplied.values()
    unset = Substitutions().values()
    for key, value in declared.items():
        if merged[key] == unset[key]:
            merged[key] = str(value)
    effective = Substitutions(
        serial=str(merged["serial"]),
        name=str(merged["name"]),
        date=str(merged["date"]),
    )
    return TestSequence(
        name=str(document.get("name", "")),
        description=str(document.get("description", "")),
        steps=steps,
        substitutions=effective,
        source=source,
        version=int(version),
    )


def load_sequence(
    path: str, *, substitutions: Substitutions | None = None
) -> TestSequence:
    """Read and validate a sequence file."""
    try:
        with open(path, encoding="utf-8") as stream:
            text = stream.read()
    except OSError as error:
        raise SequenceFormatError(f"cannot read {path}: {error}") from error
    return parse_sequence(text, source=path, substitutions=substitutions)


def sequence_identity(sequence: TestSequence) -> str:
    """Digest of the steps and set-up, ignoring the file's location."""
    import hashlib

    payload = json.dumps(
        {
            "name": sequence.name,
            "description": sequence.description,
            "steps": [
                {
                    "action": step.kind.value,
                    "fail": step.on_failure.value,
                    "unavailable": step.on_unavailable.value,
                    "fields": dict(sorted(step.fields.items())),
                }
                for step in sequence.steps
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def resolve_output(
    path: str, base_directory: str, output_directory: str, where: str
) -> str:
    """Resolve ``path`` and refuse anything outside the output directory.

    Both the sequence file's directory and an explicit output directory are
    checked, so neither a relative escape (``../../etc/x``) nor an absolute
    path can write somewhere the operator did not intend.
    """
    resolved = os.path.realpath(os.path.join(base_directory, path))
    allowed = {
        os.path.realpath(base_directory),
        os.path.realpath(output_directory),
    }
    for root in allowed:
        if os.path.commonpath([root, resolved]) == root:
            return resolved
    raise SequenceFormatError(
        f"{where}: {path!r} resolves outside the allowed output"
        f" directories ({', '.join(sorted(allowed))})"
    )


#: An action handler receives the step and the run context and returns
#: ``(status, detail, artifact)``.  Handlers are supplied by the caller so
#: this module stays free of acquisition and file-format concerns.
Handler = Callable[[SequenceStep, "RunContext"], tuple[StepStatus, str, str]]


@dataclass
class RunContext:
    """State shared by the steps of one run."""

    sequence: TestSequence
    base_directory: str
    output_directory: str
    substitutions: Substitutions
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot: Any = None
    result: SequenceResult | None = None
    #: Run state that must not reach the serialized record -- handler
    #: observations such as the limit evaluations of the last check.  The
    #: caller supplies the dict and keeps a reference, so it can freeze a
    #: verdict log entry at the run boundary without the record growing
    #: fields the file format does not know.
    extras: dict[str, Any] = field(default_factory=dict)

    def path_for(self, step: SequenceStep) -> str:
        """Resolve a written artifact against the *output* directory.

        Relative paths belong to the output directory, not to the sequence
        file: a sequence is a program, and its results should land where the
        operator said they should, not beside the program.
        """
        raw = str(step.fields.get("path", ""))
        expanded = self.substitutions.apply(raw, step.describe())
        return resolve_output(
            expanded,
            self.output_directory,
            self.output_directory,
            step.describe(),
        )

    def read_path(self, step: SequenceStep, key: str) -> str:
        """Resolve an input (limits) against the sequence's own directory."""
        raw = str(step.fields.get(key, ""))
        expanded = self.substitutions.apply(raw, step.describe())
        return resolve_output(
            expanded,
            self.base_directory,
            self.output_directory,
            step.describe(),
        )

    def text(self, step: SequenceStep, key: str) -> str:
        """Expand substitutions in a free-text field."""
        return self.substitutions.apply(
            str(step.fields.get(key) or ""), step.describe()
        )


def run_sequence(  # noqa: PLR0913
    sequence: TestSequence,
    handlers: Mapping[ActionKind, Handler],
    *,
    base_directory: str,
    output_directory: str,
    metadata: Mapping[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> SequenceResult:
    """Execute ``sequence`` and return its record.

    A step whose action has no handler is an error, not a silent skip: a
    sequence that quietly ignores a check is worse than one that fails.
    """
    # The output directory is where results belong; creating it is part of
    # running the sequence, not of any single step, so a first run into a
    # fresh directory succeeds instead of failing at the first save.
    os.makedirs(output_directory, exist_ok=True)
    for step in sequence.steps:
        if step.kind not in handlers and step.kind is not ActionKind.PROMPT:
            raise SequenceError(
                f"no handler for action {step.kind.value!r};"
                " supply one so the step can actually run"
            )
    context = RunContext(
        sequence=sequence,
        base_directory=base_directory,
        output_directory=output_directory,
        substitutions=sequence.substitutions,
        metadata=dict(metadata or {}),
        extras={} if extras is None else extras,
    )
    result = SequenceResult(
        sequence=sequence,
        serial=sequence.substitutions.serial,
        metadata=dict(metadata or {}),
    )
    context.result = result
    for step in sequence.steps:
        handler = handlers.get(step.kind)
        if handler is None:
            # Only an unanswered prompt can be resolved by a policy; any
            # other missing handler was refused above.
            fallback = str(step.fields.get("headless", "stop"))
            detail = f"headless: prompt not answered ({fallback})"
            if fallback == "continue":
                status = StepStatus.OK
            elif fallback == "skip":
                status = StepStatus.SKIPPED
            else:
                status = StepStatus.UNAVAILABLE
            artifact = ""
        else:
            status, detail, artifact = handler(step, context)
        result.steps.append(
            StepResult(
                index=step.index,
                kind=step.kind,
                status=status,
                detail=detail,
                artifact=artifact,
            )
        )
        if status is StepStatus.FAILED and step.on_failure is StopPolicy.STOP:
            result.stopped_at = step.index
            break
        if (
            status is StepStatus.UNAVAILABLE
            and step.on_unavailable is StopPolicy.STOP
        ):
            result.stopped_at = step.index
            break
    return result
