"""Driver-level tests for how each firmware scans a sweep.

The fake serial interface answers the ASCII commands the driver sends, so a
driver can be constructed in-process for firmware versions whose ``help``
output does not advertise every command the driver knows about.
"""

from __future__ import annotations

import pytest

from NanoVNASaver.Hardware.SV4401A import SV4401A
from NanoVNASaver.Hardware.SV6301A import SV6301A

from .scan_fakes import FakeInterface, help_lines


def _interface(version: str, *, mask: bool = False) -> FakeInterface:
    extra = ("scan mask",) if mask else ()
    return FakeInterface(
        replies={
            "version": [version],
            "help": [help_lines(*extra)],
            "frequencies": ["1000000", "2000000"],
        }
    )


@pytest.mark.parametrize("driver", [SV4401A, SV6301A])
@pytest.mark.parametrize("version", ["0.6.0", "0.7.1"])
def test_scan_mask_removal_survives_missing_feature(driver, version):
    """Firmware without ``scan mask`` must not crash the driver."""
    vna = driver(_interface(version))

    assert "Scan mask command" not in vna.features
    assert "Scan command" in vna.features
    assert vna.sweep_method == "scan"


@pytest.mark.parametrize("driver", [SV4401A, SV6301A])
def test_old_firmware_declares_exactly_scan_command(driver):
    """A pre-scan-mask device declares the plain ``scan`` command only."""
    vna = driver(_interface("0.6.0"))

    assert vna.features == {"Scan command", "Customizable data points"}
    assert vna.sweep_method == "scan"
