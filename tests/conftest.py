"""Shared Qt lifetime for GUI integration tests."""

import gc
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _qt_application():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        []
    )
    yield application
    # Tests construct windows without running exec(), so deferred deletion
    # otherwise waits until interpreter shutdown, after Qt begins tearing down.
    # Delete windows while QApplication and their Python callbacks are alive.
    for widget in application.topLevelWidgets():
        worker = getattr(widget, "worker", None)
        if isinstance(worker, QtCore.QThread):
            worker.quit()
            worker.wait(1000)
        widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete
    )
    gc.collect()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete
    )


@pytest.fixture(scope="session")
def qt_app(_qt_application):
    return _qt_application
