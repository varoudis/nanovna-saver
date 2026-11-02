#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
import logging
import platform
from collections import namedtuple
from time import monotonic, sleep

import serial
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from .AVNA import AVNA
from .JNCRadio_VNA_3G import JNCRadio_VNA_3G
from .LiteVNA64 import LiteVNA64
from .NanoVNA import NanoVNA
from .NanoVNA_F import NanoVNA_F
from .NanoVNA_F_V2 import NanoVNA_F_V2
from .NanoVNA_F_V3 import NanoVNA_F_V3
from .NanoVNA_H import NanoVNA_H
from .NanoVNA_H4 import NanoVNA_H4
from .NanoVNA_V2 import NanoVNA_V2
from .Serial import Interface, drain_serial
from .SV4401A import SV4401A
from .SV6301A import SV6301A
from .TinySA import TinySA, TinySA_Ultra
from .VNA import VNA

logger = logging.getLogger(__name__)

USBDevice = namedtuple("USBDevice", "vid pid name")

USBDEVICETYPES = (
    USBDevice(0x0483, 0x5740, "NanoVNA"),
    USBDevice(0x16C0, 0x0483, "AVNA"),
    USBDevice(0x04B4, 0x0008, "S-A-A-2"),
)
RETRIES = 3
TIMEOUT = 0.2
WAIT = 0.05

#: Total budget (seconds) for enumerating and probing candidate interfaces.
#: Probing an unresponsive port costs a drain/retry/read cycle per attempt
#: (see :func:`detect_version`), so the scan is bounded instead of letting a
#: wedged port block the caller for minutes.
SCAN_BUDGET = 8.0


def probe_deadline(budget: float | None) -> float | None:
    """Turn a relative budget in seconds into an absolute deadline."""
    if budget is None or budget <= 0:
        return None
    return monotonic() + float(budget)


def deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and monotonic() >= deadline


NAME2DEVICE = {
    "S-A-A-2": NanoVNA_V2,
    "AVNA": AVNA,
    "H4": NanoVNA_H4,
    "H": NanoVNA_H,
    "F_V2": NanoVNA_F_V2,
    "F_V3": NanoVNA_F_V3,
    "F": NanoVNA_F,
    "NanoVNA": NanoVNA,
    "tinySA": TinySA,
    "tinySA_Ultra": TinySA_Ultra,
    "JNCRadio": JNCRadio_VNA_3G,
    "SV4401A": SV4401A,
    "SV6301A": SV6301A,
    "LiteVNA64": LiteVNA64,
    "Unknown": NanoVNA,
}


# The USB Driver for NanoVNA V2 seems to deliver an
# incompatible hardware info like:
# 'PORTS\\VID_04B4&PID_0008\\DEMO'
# This function will fix it.


def _fix_v2_hwinfo(dev):
    # if dev.hwid == r'PORTS\VID_04B4&PID_0008\DEMO':
    if r"PORTS\VID_04B4&PID_0008" in dev.hwid:
        dev.vid, dev.pid = 0x04B4, 0x0008
    return dev


def usb_typename(device: ListPortInfo) -> str:
    return next(
        (
            t.name
            for t in USBDEVICETYPES
            if device.vid == t.vid and device.pid == t.pid
        ),
        "",
    )


# Get list of interfaces with VNAs connected


def get_interfaces(
    budget: float | None = SCAN_BUDGET, *, needs_comment: bool = True
) -> list[Interface]:
    """Enumerate NanoVNA-like serial ports, optionally identifying each one.

    With ``needs_comment=True`` every candidate port is probed, i.e. opened
    and asked what firmware it runs.  That is a blocking, hardware-bound
    conversation, so the call belongs off the GUI thread; ``budget``
    (seconds) bounds the whole scan and the interfaces found so far are
    returned when it runs out, each already closed again, so a wedged port
    cannot hang the caller.  ``None`` or ``0`` disables the bound.

    With ``needs_comment=False`` only the port metadata reported by the OS
    is collected: no port is opened, no device is touched, and the returned
    :class:`Interface` objects carry only type, port and (empty) comment.
    """
    interfaces = []
    deadline = probe_deadline(budget) if needs_comment else None
    # serial like usb interfaces
    for d in list_ports.comports():
        if needs_comment and deadline_expired(deadline):
            logger.warning(
                "Stopped scanning serial ports after %.1f s; %d found",
                budget,
                len(interfaces),
            )
            break
        if platform.system() == "Windows" and d.vid is None:
            d = _fix_v2_hwinfo(d)  # noqa: PLW2901
        if not (typename := usb_typename(d)):
            continue
        logger.debug(
            "Found %s USB:(%04x:%04x) on port %s",
            typename,
            d.vid,
            d.pid,
            d.device,
        )
        iface = Interface("serial", typename)
        iface.port = d.device
        #: True only once a probe confirmed the firmware variant.
        iface.probed = False
        if needs_comment:
            if not _probe_interface(iface, deadline):
                continue
            iface.probed = True
        interfaces.append(iface)

    logger.debug("Interfaces: %s", interfaces)
    return interfaces


def list_serial_ports() -> list[Interface]:
    """Enumerate candidate ports without opening or probing any of them.

    This is the cheap, side effect free half of :func:`get_interfaces`: it
    only asks the OS which ports exist, so it may be called on the GUI
    thread.  The returned interfaces are marked ``probed = False``; a
    consumer that needs the firmware variant either scans properly
    (:func:`get_interfaces`, off the GUI thread) or probes the port it is
    about to use.
    """
    return get_interfaces(needs_comment=False)


def _probe_interface(iface: Interface, deadline: float | None) -> bool:
    """Open ``iface``, read its comment and close it again.

    Returns ``True`` when the port answered.  The port is closed on every
    path, including a failed open and a failing detection, so a scan never
    leaks a handle.
    """
    opened = False
    try:
        iface.open()
        opened = True
        if deadline_expired(deadline):
            logger.debug("Scan budget exhausted; not probing %s", iface.port)
            return False
        iface.comment = get_comment(iface)
        return True
    except Exception:
        logger.debug("Unable to probe %s", iface.port, exc_info=True)
        return False
    finally:
        if opened:
            try:
                iface.close()
            except Exception:
                logger.debug("Unable to close %s", iface.port, exc_info=True)


def get_portinfos() -> list[str]:
    portinfos = []
    # serial like usb interfaces
    for d in list_ports.comports():
        logger.debug("Found USB:(%04x:%04x) on port %s", d.vid, d.pid, d.device)
        iface = Interface("serial", "DEBUG")
        iface.port = d.device
        iface.open()
        version = detect_version(iface)
        iface.close()
        portinfos.append(version)
    return portinfos


def get_VNA(iface: Interface) -> VNA:
    # serial_port.timeout = TIMEOUT
    return NAME2DEVICE[iface.comment](iface)


def get_comment(iface: Interface) -> str:
    logger.info("Finding correct VNA type...")
    with iface.lock:
        vna_version = detect_version(iface)
    if vna_version == "v2":
        return "S-A-A-2"
    if vna_version == "lite_vna_64":
        return "LiteVNA64"

    logger.info("Finding firmware variant...")
    info = get_info(iface)
    for search, name in (
        ("AVNA + Teensy", "AVNA"),
        ("NanoVNA-H 4", "H4"),
        ("NanoVNA-H", "H"),
        ("NanoVNA-F_V2", "F_V2"),
        ("NanoVNA-F_V3", "F_V3"),
        ("NanoVNA-F", "F"),
        ("NanoVNA", "NanoVNA"),
        ("tinySA4", "tinySA_Ultra"),
        ("tinySA", "tinySA"),
        ("JNCRadio_VNA_3G", "JNCRadio"),
        ("SV4401A", "SV4401A"),
        ("SV6301A", "SV6301A"),
    ):
        if info.find(search) >= 0:
            return name
    logger.warning("Did not recognize NanoVNA type from firmware.")
    return "Unknown"


def detect_version(serial_port: serial.Serial) -> str:
    data = ""
    for i in range(RETRIES):
        drain_serial(serial_port)
        serial_port.write("\r".encode("ascii"))
        # workaround for some UnicodeDecodeError ... repeat ;-)
        drain_serial(serial_port)
        serial_port.write("\r".encode("ascii"))
        sleep(0.05)

        data = serial_port.read(128).decode("ascii")
        if data.startswith("ch> "):
            return "v1"
        # -H versions
        if data.startswith("\r\nch> "):
            return "vh"
        if data.startswith("\r\n?\r\nch> "):
            return "vh"
        if data.startswith("2"):
            return (
                "lite_vna_64" if LiteVNA64.is_lite_vna_64(serial_port) else "v2"
            )
        logger.debug("Retry detection: %s", i + 1)
    logger.error("No VNA detected. Hardware responded to CR with: %s", data)
    return ""


def get_info(serial_port: serial.Serial) -> str:
    for _ in range(RETRIES):
        drain_serial(serial_port)
        serial_port.write("info\r".encode("ascii"))
        lines = []
        retries = 0
        while True:
            line = serial_port.readline().decode("ascii").strip()
            if not line:
                retries += 1
                if retries > RETRIES:
                    return ""
                sleep(WAIT)
                continue
            if line == "info":  # suppress echo
                continue
            if line.startswith("ch>"):
                logger.debug("Needed retries: %s", retries)
                break
            lines.append(line)
        logger.debug("Info output: %s", lines)
        return "\n".join(lines)
    return ""
