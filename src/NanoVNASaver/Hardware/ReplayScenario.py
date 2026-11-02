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
"""Touchstone replay scenario (REPLAY-01).

A replay scenario turns a Touchstone file into the response of the
simulated device, so a recorded measurement can be swept through the normal
simulator/worker pipeline (segments, averaging, cancellation, calibration
state, provenance) instead of through a separate offline trace import.

Interpolation contract
----------------------
* A frequency present in the file yields that sample unchanged: the real
  and imaginary parts are returned bit-exactly.
* Between two file samples the real part and the imaginary part are
  interpolated linearly and independently - ordinary complex linear
  interpolation on the real/imaginary pair.
* Outside the file's span nothing is produced.  ``ReplayOutOfSpanError`` is
  raised; held endpoint values, zeros or any other invented sample are
  never returned.

Honesty constraints
-------------------
* A one-port file replays S11 only.  S21 stays absent from every published
  snapshot; it is never replaced by zero-valued samples.
* Only S11 and S21 can be replayed, because that is the whole channel set
  the sweep pipeline carries.  S12/S22 present in a two-port file are
  reported as ignored instead of being silently presented as measurements.
* A reference impedance other than the simulated device's 50 ohms is
  rejected with an actionable error.  The samples are not renormalized, so
  the simulator's reference impedance always describes the data it emits.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..RFTools import Datapoint
from ..Touchstone import PARAMETERS, Touchstone
from .Capability import SourceSpan, SweepCapability
from .SimulatedScenarios import REFERENCE_IMPEDANCE, ScenarioResult

logger = logging.getLogger(__name__)

#: Largest file accepted, before anything is parsed.
MAX_REPLAY_BYTES = 32 * 1024 * 1024

#: Largest number of samples per channel accepted after parsing.
MAX_REPLAY_POINTS = 250_000

#: Supported Touchstone file types and their number of ports.
SUPPORTED_SUFFIXES: dict[str, int] = {".s1p": 1, ".s2p": 2}

#: Channels the sweep pipeline can carry, in worker order.
CHANNEL_S11 = "data 0"
CHANNEL_S21 = "data 1"

#: S-parameters a file may contribute to a replay.
REPLAYABLE_PARAMETERS: tuple[str, ...] = ("S11", "S21")


class ReplayError(Exception):
    """A file cannot be used for replay; the message is meant for the user."""


class ReplayOutOfSpanError(ReplayError, RuntimeError):
    """A frequency outside the file's own span was requested.

    Deliberately not a ``ValueError``: the worker retries value errors as
    transient device trouble, while this is a deterministic request error.
    """


class ReplayReferenceImpedanceError(ReplayError, ValueError):
    """The file's reference impedance differs from the device's."""


class ReplayChannelError(ReplayError, RuntimeError):
    """A channel that the file does not provide was requested."""


def _check_finite(value: float, what: str, filename: str) -> float:
    if not math.isfinite(value):
        raise ReplayError(f"{filename}: {what} is not a finite number")
    return value


def _validate_channel(
    parameter: str,
    samples: list[Datapoint],
    expected_freqs: tuple[int, ...],
    filename: str,
) -> tuple[Datapoint, ...]:
    """Validate one channel of a loaded file and freeze it."""
    if len(samples) != len(expected_freqs):
        raise ReplayError(
            f"{filename}: {parameter} holds {len(samples)} samples but S11"
            f" holds {len(expected_freqs)}"
        )
    frozen = []
    for index, dp in enumerate(samples):
        if dp.freq != expected_freqs[index]:
            raise ReplayError(
                f"{filename}: {parameter} and S11 disagree on the frequency"
                f" of sample {index} ({dp.freq} Hz vs"
                f" {expected_freqs[index]} Hz)"
            )
        _check_finite(dp.re, f"{parameter} sample {index} real part", filename)
        _check_finite(dp.im, f"{parameter} sample {index} imag part", filename)
        frozen.append(Datapoint(dp.freq, dp.re, dp.im))
    return tuple(frozen)


@dataclass(frozen=True)
class ReplayData:
    """A validated Touchstone file, ready to be replayed.

    ``s11``/``s21`` are exact copies of the file's samples (real/imaginary
    pairs in hertz).  ``s21`` is empty for a one-port file, which is what
    keeps such a replay S11-only all the way to export.
    """

    path: str
    kind: str
    reference_impedance: float
    s11: tuple[Datapoint, ...]
    s21: tuple[Datapoint, ...] = ()
    ignored_parameters: tuple[str, ...] = ()

    @cached_property
    def frequencies(self) -> tuple[int, ...]:
        return tuple(dp.freq for dp in self.s11)

    @property
    def has_s21(self) -> bool:
        return bool(self.s21)

    @property
    def start(self) -> int:
        return self.frequencies[0]

    @property
    def stop(self) -> int:
        return self.frequencies[-1]

    @property
    def name(self) -> str:
        return Path(self.path).name

    def channel(self, parameter: str) -> tuple[Datapoint, ...]:
        if parameter == "S11":
            return self.s11
        if parameter == "S21":
            return self.s21
        raise KeyError(parameter)

    def is_exact(self, freq: float) -> bool:
        """True when ``freq`` is one of the file's own sample frequencies."""
        index = bisect_left(self.frequencies, freq)
        return index < len(self.frequencies) and self.frequencies[index] == freq

    def value_at(self, parameter: str, freq: float) -> complex:
        """Sample ``parameter`` at ``freq``, exactly or interpolated.

        Raises ``ReplayOutOfSpanError`` rather than extrapolating.
        """
        samples = self.channel(parameter)
        if not samples:
            raise ReplayChannelError(
                f"{self.name} provides no {parameter} channel"
            )
        freqs = self.frequencies
        index = bisect_left(freqs, freq)
        if index < len(freqs) and freqs[index] == freq:
            dp = samples[index]
            return complex(dp.re, dp.im)
        if index == 0 or index >= len(freqs):
            raise ReplayOutOfSpanError(
                f"{freq:.0f} Hz is outside the replay range {self.start} Hz"
                f" to {self.stop} Hz of {self.name}; replay does not"
                " extrapolate beyond the file"
            )
        lower, upper = index - 1, index
        f_lower, f_upper = freqs[lower], freqs[upper]
        t = (freq - f_lower) / (f_upper - f_lower)
        low_dp, high_dp = samples[lower], samples[upper]
        return complex(
            low_dp.re + t * (high_dp.re - low_dp.re),
            low_dp.im + t * (high_dp.im - low_dp.im),
        )

    def summary(self) -> str:
        """One-line, human-readable description of the replayed file."""
        channels = "S11 and S21" if self.has_s21 else "S11 only"
        text = (
            f"{self.kind} · {len(self.s11)} points ·"
            f" {self.start} Hz to {self.stop} Hz · {channels}"
        )
        if self.ignored_parameters:
            text += f" · {', '.join(self.ignored_parameters)} ignored"
        return text

    def provenance_text(self) -> str:
        """Replay source recorded in snapshots, exports and reports."""
        channels = "S11 and S21" if self.has_s21 else "S11 only (no S21)"
        text = (
            f"replay of {self.path} ({self.kind}, {channels},"
            " complex linear interpolation, no extrapolation)"
        )
        if self.ignored_parameters:
            text += (
                f"; {', '.join(self.ignored_parameters)} from the file are"
                " not replayed"
            )
        return text


def _validate_layout(
    dataset: Touchstone, ports: int, filename: str
) -> tuple[str, ...]:
    present = dataset.present_parameters()
    if "S11" not in present:
        raise ReplayError(
            f"{filename}: no S11 samples found; a replay needs at least a"
            " one-port S11 column"
        )
    expected = {
        1: {("S11",)},
        2: {("S11", "S21"), ("S11", "S21", "S12", "S22")},
    }[ports]
    if present not in expected:
        names = ", ".join(present) or "none"
        wanted = " or ".join(", ".join(e) for e in sorted(expected))
        raise ReplayError(
            f"{filename}: {len(present)} S-parameter column(s)"
            f" ({names}) cannot be replayed as a {ports}-port file;"
            f" expected {wanted}"
        )
    return present


def load_replay_file(path: str | Path) -> ReplayData:
    """Load and validate ``path``; raise ``ReplayError`` when unusable.

    Validation covers the file type, size, emptiness, option line, column
    layout, S11 presence, channel lengths, frequency ordering (strictly
    ascending, positive, no duplicates), finite sample values and the
    reference impedance.
    """
    file = Path(path).expanduser()
    ports = SUPPORTED_SUFFIXES.get(file.suffix.lower())
    if ports is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ReplayError(
            f"{file.name}: unsupported file type; choose a Touchstone file"
            f" ({supported})"
        )
    if not file.is_file():
        raise ReplayError(f"{file} does not exist or is not a file")
    try:
        size = file.stat().st_size
    except OSError as exc:
        raise ReplayError(f"{file} cannot be read: {exc}") from exc
    if size == 0:
        raise ReplayError(f"{file.name} is empty; it holds no samples")
    if size > MAX_REPLAY_BYTES:
        raise ReplayError(
            f"{file.name} is {size} bytes; at most {MAX_REPLAY_BYTES} bytes"
            " can be replayed"
        )
    try:
        with file.open(encoding="utf-8") as infile:
            text = infile.read(MAX_REPLAY_BYTES + 1)
    except OSError as exc:
        raise ReplayError(f"{file} cannot be read: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ReplayError(
            f"{file.name} is not valid UTF-8 text: {exc}"
        ) from exc
    if len(text) > MAX_REPLAY_BYTES:
        raise ReplayError(
            f"{file.name} exceeds the {MAX_REPLAY_BYTES} byte replay limit"
        )
    if not text.strip():
        raise ReplayError(f"{file.name} is empty; it holds no samples")

    filename = file.name
    dataset = Touchstone(str(file))
    try:
        dataset.loads_strict(text)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ReplayError(
            f"{filename}: not a usable Touchstone 1.x file: {exc}"
        ) from exc

    if dataset.opts.parameter != "s":
        raise ReplayError(
            f"{filename} holds"
            f" {dataset.opts.parameter.upper()}-parameters; only"
            " S-parameters can be replayed"
        )

    present = _validate_layout(dataset, ports, filename)

    reference = float(dataset.opts.resistance)
    _check_finite(reference, "reference impedance", filename)
    if reference <= 0:
        raise ReplayError(
            f"{filename}: reference impedance must be positive,"
            f" not {reference:g}"
        )
    if not math.isclose(reference, REFERENCE_IMPEDANCE, rel_tol=1e-9):
        raise ReplayReferenceImpedanceError(
            f"{filename} is referenced to {reference:g} ohm, but the"
            f" simulated device is a {REFERENCE_IMPEDANCE:g} ohm system."
            " Replay does not renormalize: export the file at 50 ohm first"
            " (for example with `scikit-rf`'s `Network.renormalize`) or"
            " use a 50 ohm file."
        )

    s11 = list(dataset.s11)
    if not s11:
        raise ReplayError(f"{filename} contains no samples")
    if len(s11) > MAX_REPLAY_POINTS:
        raise ReplayError(
            f"{filename} holds {len(s11)} samples; at most"
            f" {MAX_REPLAY_POINTS} can be replayed"
        )

    previous = -1
    for index, dp in enumerate(s11):
        if dp.freq <= 0:
            raise ReplayError(
                f"{filename}: sample {index} has non-positive frequency"
                f" {dp.freq} Hz"
            )
        if dp.freq <= previous:
            raise ReplayError(
                f"{filename}: duplicate or non-ascending frequency"
                f" {dp.freq} Hz at sample {index}; each frequency may appear"
                " only once"
            )
        previous = dp.freq
    freqs = tuple(dp.freq for dp in s11)

    frozen_s11 = _validate_channel("S11", s11, freqs, filename)
    frozen_s21: tuple[Datapoint, ...] = ()
    if "S21" in present:
        frozen_s21 = _validate_channel(
            "S21", list(dataset.s21), freqs, filename
        )

    ignored = tuple(
        p for p in PARAMETERS if p not in REPLAYABLE_PARAMETERS and p in present
    )
    kind = "S1P" if ports == 1 else "S2P"
    logger.info(
        "Loaded replay source %s: %s, %d points",
        file,
        kind,
        len(frozen_s11),
    )
    return ReplayData(
        path=str(file),
        kind=kind,
        reference_impedance=reference,
        s11=frozen_s11,
        s21=frozen_s21,
        ignored_parameters=ignored,
    )


@dataclass(frozen=True)
class ReplayScenario:
    """Callable scenario replaying one Touchstone file.

    ``scenario_key``/``display_name`` are read by ``SimulatedDevice`` for
    provenance, so a replayed sweep is labelled as simulated *and* as
    derived from a file.
    """

    scenario_key = "replay"
    display_name = "Touchstone replay"

    data: ReplayData

    @property
    def supported_channels(self) -> tuple[str, ...]:
        if self.data.has_s21:
            return (CHANNEL_S11, CHANNEL_S21)
        return (CHANNEL_S11,)

    def __call__(self, freq: float) -> ScenarioResult:
        s11 = self.data.value_at("S11", freq)
        # The S21 slot is never published for a one-port file: the device
        # rejects the channel before sampling (see ``supported_channels``).
        s21 = (
            self.data.value_at("S21", freq)
            if self.data.has_s21
            else complex(0.0, 0.0)
        )
        return ScenarioResult(s11, s21)

    def preflight(self, start: int, stop: int) -> str | None:
        """Return a user-facing problem, or ``None`` when a range is usable.

        Called by the worker for every segment it is about to request, so a
        span that would need extrapolation is rejected before a single
        sample is read.
        """
        if stop < start:
            return (
                f"The sweep span is invalid: {start} Hz to {stop} Hz."
                " The stop frequency must not be below the start frequency."
            )
        data = self.data
        if start < data.start or stop > data.stop:
            return (
                f"The sweep {start} Hz to {stop} Hz is outside the replay"
                f" range {data.start} Hz to {data.stop} Hz of {data.name}."
                " Replay never extrapolates; set the sweep inside the"
                " file's own span."
            )
        return None

    def provenance_text(self) -> str:
        return self.data.provenance_text()

    def sweep_span(self) -> SourceSpan:
        """The file's own span, as a capability source.

        The text is reused verbatim in rejection messages, so a planner
        names the range the user has to stay inside.
        """
        return SourceSpan(
            self.data.start,
            self.data.stop,
            name=self.data.name,
            label="the replay range",
        )

    def capability(self, vna=None) -> SweepCapability:
        """The sweep capability of replaying this file.

        The span is the file's own range (never extrapolated), channels are
        the ones the file really holds, and the file's reference impedance
        is the one every sample is stated in.  When ``vna`` is given, its
        declared limits bound the result as well, so a planned range is
        never wider than the device that will request it.
        """
        span = self.sweep_span()
        if vna is None:
            return SweepCapability.from_vna(
                ReplayDeviceLimits(self.data), source=span
            )
        return SweepCapability.from_vna(vna, source=span)


class ReplayDeviceLimits:
    """Declared limits of a replay source, in the shape ``from_vna`` reads.

    A file holds no Nyquist grid, so there is no maximum point count to
    declare; only the span, the channels and the impedance are known.
    """

    def __init__(self, data: ReplayData) -> None:
        self.supported_channels = (
            (CHANNEL_S11, CHANNEL_S21) if data.has_s21 else (CHANNEL_S11,)
        )
        self.reference_impedance = data.reference_impedance
