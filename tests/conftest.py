"""Shared Qt lifetime for GUI integration tests.

Qt objects created by these tests must never be released by the cyclic
garbage collector while their C++ objects are still alive.  PySide keeps its
Python-slot bookkeeping on the *sender*: when a receiver wrapper is freed it
erases the connection from the sender's map.  If that sender wrapper is
freed in the same (or an earlier) collection pass, the erase reads and
writes released memory, and the process dies inside
``QMetaObject::Connection``/``QHash`` - a SIGSEGV/SIGABRT that surfaces at an
unrelated point later in the run.  The failure is timing dependent, which is
why it appears in a mixed suite and not in single modules.

So windows are pinned: every top-level widget (and, through it, the whole
Qt object graph it owns) is kept strongly referenced by ``_PINNED_WINDOWS``
for the rest of the session.  Pinning makes that graph uncollectable, so no
collection pass can free a sender before the receivers connected to it.

Destruction through Qt still happens, deterministically: the session
teardown stops every worker thread and lets Qt delete the windows while
``QApplication`` and the Python callbacks it holds are alive.  The wrappers
stay pinned afterwards, so interpreter shutdown has no PySide connection
bookkeeping left to replay.

Tests are free to create any number of windows or QObject wrappers; nothing
here depends on test names, order or counts.
"""

import gc
import os

import pytest

#: Python wrappers of every top-level widget created during this session.
#:
#: Keeping the wrapper alive keeps its C++ object reachable through the
#: Python callbacks Qt holds, so a cyclic collection pass can neither free a
#: live widget nor free a sender before its receivers.
_PINNED_WINDOWS: list = []

#: ``id()`` of the widgets already held by :data:`_PINNED_WINDOWS`.
_PINNED_IDS: set[int] = set()


def _application():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance()


def _is_valid(obj) -> bool:
    """True while *obj* still has a live C++ object behind its wrapper."""
    import shiboken6

    try:
        return shiboken6.isValid(obj)
    except RuntimeError:
        # A wrapper whose C++ object is gone cannot be inspected either.
        return False


def _flush_deferred_deletes() -> None:
    from PySide6 import QtCore

    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete
    )


def _pin_windows() -> None:
    """Take a strong reference to every top-level widget not pinned yet."""
    application = _application()
    if application is None:
        return
    for widget in application.topLevelWidgets():
        if id(widget) in _PINNED_IDS:
            continue
        _PINNED_IDS.add(id(widget))
        _PINNED_WINDOWS.append(widget)


def _stop_workers() -> None:
    """Stop every sweep worker before its window is destroyed.

    A running ``QThread`` outliving its window would be destroyed from the
    wrong thread; each one is asked to stop and given time to leave its run
    loop while the widgets it belongs to are still intact.
    """
    from PySide6 import QtCore

    for widget in _PINNED_WINDOWS:
        # An invalidated wrapper cannot be inspected at all: raising here
        # would mask the real teardown.
        worker = getattr(widget, "worker", None) if _is_valid(widget) else None
        if not isinstance(worker, QtCore.QThread) or not _is_valid(worker):
            continue
        worker.quit()
        worker.wait(10_000)


def _delete_windows() -> None:
    """Hand every window to Qt, in reverse creation order."""
    deleted = False
    for widget in reversed(_PINNED_WINDOWS):
        if not _is_valid(widget):
            continue
        widget.deleteLater()
        deleted = True
    if deleted:
        _flush_deferred_deletes()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    yield
    # Fixture setup is where tests create their windows: pin them before the
    # test body runs, so a collection pass during the test cannot release a
    # live Qt object graph.
    _pin_windows()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    _pin_windows()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    # Fixture teardown has dropped its own references by this point; pin
    # whatever it left behind.
    yield
    _pin_windows()


@pytest.fixture(scope="session", autouse=True)
def _qt_application():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        []
    )
    yield application
    # Tests construct windows without running exec(), so deferred deletion
    # otherwise waits until interpreter shutdown, after Qt has begun tearing
    # down.  Destroy the windows here instead, while QApplication and their
    # Python callbacks are alive, in a known order: workers first, then the
    # widgets, then any deletion the widgets themselves deferred.
    _pin_windows()
    _stop_workers()
    _delete_windows()
    for _ in range(3):
        gc.collect()
        _flush_deferred_deletes()


@pytest.fixture(scope="session")
def qt_app(_qt_application):
    return _qt_application
