"""NanoVNA-Saver.

Importing this package must not pull in Qt: the automation API, the
processing pipeline and the analysis modules are usable from a headless
script, and an eager import here would drag in every widget the moment a
script touched a single helper.  ``__main__`` and the tests import the window
class explicitly.

``from NanoVNASaver import NanoVNASaver`` still works — it is resolved on
first access — but it is no longer a side effect of importing the package.
"""

from typing import TYPE_CHECKING

__all__ = ["NanoVNASaver"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .NanoVNASaver import NanoVNASaver


def __getattr__(name: str):
    if name == "NanoVNASaver":
        from .NanoVNASaver import NanoVNASaver as window

        globals()["NanoVNASaver"] = window
        return window
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
