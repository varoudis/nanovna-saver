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
"""Observable sweep lifecycle (SWEEP-01).

The acquisition worker moves through an explicit state machine instead of
exposing a bare percentage.  Exactly one terminal state is published per
sweep, and every terminal state has a matching cleanup path in the worker.

Legal transitions::

    IDLE ──► PREPARING ──► MEASURING ──► AVERAGING ──► COMPLETED
              │  ▲            │  ▲           │
              │  └─ RETRYING ─┘  └───────────┘
              │       │
              ▼       ▼
          FAILED   CANCELLED

``IDLE`` is the only non-terminal state that may be entered after a terminal
one, so a worker can be restarted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SweepState(Enum):
    """Lifecycle states of a single sweep."""

    IDLE = "idle"
    PREPARING = "preparing"
    MEASURING = "measuring"
    AVERAGING = "averaging"
    RETRYING = "retrying"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def active(self) -> bool:
        return self in ACTIVE_STATES


ACTIVE_STATES = frozenset(
    {
        SweepState.PREPARING,
        SweepState.MEASURING,
        SweepState.AVERAGING,
        SweepState.RETRYING,
        SweepState.CANCELLING,
    }
)
TERMINAL_STATES = frozenset(
    {
        SweepState.COMPLETED,
        SweepState.CANCELLED,
        SweepState.FAILED,
    }
)

#: Allowed successor states.  ``IDLE`` follows any terminal state so the same
#: worker can run again.
ALLOWED_TRANSITIONS: dict[SweepState, frozenset[SweepState]] = {
    SweepState.IDLE: frozenset({SweepState.PREPARING}),
    SweepState.PREPARING: frozenset(
        {
            SweepState.MEASURING,
            SweepState.RETRYING,
            SweepState.CANCELLING,
            SweepState.FAILED,
        }
    ),
    SweepState.MEASURING: frozenset(
        {
            SweepState.MEASURING,
            SweepState.AVERAGING,
            SweepState.RETRYING,
            SweepState.CANCELLING,
            SweepState.FAILED,
            SweepState.COMPLETED,
        }
    ),
    SweepState.AVERAGING: frozenset(
        {
            SweepState.AVERAGING,
            SweepState.MEASURING,
            SweepState.RETRYING,
            SweepState.CANCELLING,
            SweepState.FAILED,
            SweepState.COMPLETED,
        }
    ),
    SweepState.RETRYING: frozenset(
        {
            SweepState.MEASURING,
            SweepState.AVERAGING,
            SweepState.CANCELLING,
            SweepState.FAILED,
        }
    ),
    SweepState.CANCELLING: frozenset({SweepState.CANCELLED, SweepState.FAILED}),
    SweepState.COMPLETED: frozenset({SweepState.IDLE}),
    SweepState.CANCELLED: frozenset({SweepState.IDLE}),
    SweepState.FAILED: frozenset({SweepState.IDLE}),
}


class IllegalSweepTransitionError(ValueError):
    """Raised when a state change does not follow ``ALLOWED_TRANSITIONS``."""


@dataclass
class SweepStateMachine:
    """Guards transitions and guarantees a single terminal outcome."""

    sweep_id: int = 0
    state: SweepState = SweepState.IDLE
    terminal_count: int = 0
    history: list[SweepState] = field(default_factory=list)

    def can_transition(self, new_state: SweepState) -> bool:
        return new_state in ALLOWED_TRANSITIONS[self.state]

    def transition(self, new_state: SweepState) -> bool:
        """Move to ``new_state``; return False for a no-op repeat.

        Repeated non-terminal states (another segment, another average) are
        legal and do not extend the history.
        """
        if new_state is self.state:
            if new_state.terminal:
                raise IllegalSweepTransitionError(
                    f"sweep {self.sweep_id} already finished as"
                    f" {new_state.value}"
                )
            return False
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise IllegalSweepTransitionError(
                f"sweep {self.sweep_id}: {self.state.value}"
                f" -> {new_state.value} is not allowed"
            )
        self.state = new_state
        self.history.append(new_state)
        if new_state.terminal:
            self.terminal_count += 1
        return True

    def reset(self, sweep_id: int) -> None:
        """Prepare the machine for the next sweep."""
        self.sweep_id = sweep_id
        self.state = SweepState.IDLE
        self.history.clear()


@dataclass(frozen=True)
class SweepProgress:
    """Structured progress for one sweep (SWEEP-01.2).

    ``percentage`` is the completed fraction of the whole sweep.
    ``estimate_seconds`` stays ``None`` until enough timing data exists, so
    the UI can show "estimating" instead of a fabricated number.
    """

    sweep_id: int
    state: SweepState
    segment: int = 0
    segments: int = 1
    average: int = 0
    averages: int = 1
    completed_segments: int = 0
    completed_averages: int = 0
    retry_count: int = 0
    retry_reason: str = ""
    elapsed_seconds: float = 0.0
    estimate_seconds: float | None = None
    continuous: bool = False
    message: str = ""

    @property
    def percentage(self) -> float:
        if self.continuous:
            # Continuous mode never completes; a percentage would mislead.
            return 0.0
        total = self.segments * self.averages
        if total <= 0:
            return 0.0
        done = self.completed_segments * self.averages + self.completed_averages
        return min(100.0, 100.0 * done / total)

    @property
    def estimate_available(self) -> bool:
        return self.estimate_seconds is not None


def estimate_remaining(
    elapsed_seconds: float,
    completed_units: int,
    total_units: int,
) -> float | None:
    """Remaining time from observed durations.

    Returns ``None`` until at least one unit has completed, because an
    estimate from zero observations is meaningless.
    """
    if completed_units <= 0 or total_units <= 0:
        return None
    if total_units <= completed_units:
        return 0.0
    per_unit = elapsed_seconds / completed_units
    return per_unit * (total_units - completed_units)
