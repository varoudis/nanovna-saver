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
"""AUTO-04: the local REST/WebSocket service, driven over real sockets.

Every request here goes through ``http.client`` or a raw ``socket`` against
an ephemeral loopback port, so what is asserted is what a script would
observe: real status codes, real JSON, real WebSocket frames.  The service
is only ever constructed with a simulated device and an export directory
inside ``tmp_path``, and every fixture stops it in a ``finally`` so no port
survives a failing test.
"""

import base64
import contextlib
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.WebService import (
    SCHEMA_VERSION,
    ConfigError,
    MeasurementService,
    ServiceConfig,
    Subscriber,
    TooManySubscribersError,
    decode_frames,
    encode_frame,
    is_loopback_host,
    summarise_checks,
    websocket_accept,
)

HOST = "127.0.0.1"
TOKEN = "correct-horse-battery"

#: A band the antenna scenario covers, with the point counts the simulator
#: produces exactly (no interpolation anywhere in the pipeline).
LOW_HZ = 1_000_000
HIGH_HZ = 10_000_000
POINTS = 11

#: The rule band lies outside the sweep: the check must be UNAVAILABLE,
#: which is not a pass and must never be reported as one.
UNCOVERED_LIMITS = {
    "format": "nanovna-saver-limits",
    "version": 2,
    "name": "outside the sweep",
    "rules": [
        {
            "id": "far",
            "label": "far band",
            "metric": "s11_vswr",
            "segments": [
                {
                    "start_hz": 100_000_000,
                    "stop_hz": 200_000_000,
                    "maximum": 2.0,
                }
            ],
        }
    ],
}

#: VSWR cannot be below 1, so this rule fails every sample.
IMPOSSIBLE_LIMITS = {
    "format": "nanovna-saver-limits",
    "version": 2,
    "name": "impossible match",
    "rules": [
        {
            "id": "match",
            "label": "perfect match",
            "metric": "s11_vswr",
            "segments": [
                {
                    "start_hz": LOW_HZ,
                    "stop_hz": HIGH_HZ,
                    "maximum": 0.5,
                }
            ],
        }
    ],
}


def _factory(scenario: str = "antenna", **kwargs):
    """A device source that hands out a fresh simulator per run."""

    def make():
        return make_simulated_vna(scenario, seed=3, noise_floor=0.0, **kwargs)

    return make


@contextlib.contextmanager
def running_service(*, device=True, factory=None, **config):
    """Start a service on an ephemeral loopback port and always stop it."""
    config.setdefault("port", 0)
    if factory is None:
        factory = _factory("antenna") if device else None
    service = MeasurementService(
        ServiceConfig(**config), device_factory=factory
    )
    host, port = service.start_server()
    try:
        yield service, host, port
    finally:
        service.stop()


class Client:
    """One HTTP connection; every call returns ``(status, json)``."""

    def __init__(self, host: str, port: int, *, token: str = "") -> None:
        self._host = host
        self._port = port
        self._token = token
        self._connection: http.client.HTTPConnection | None = None

    def __enter__(self) -> "Client":
        self._connection = http.client.HTTPConnection(
            self._host, self._port, timeout=10.0
        )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._connection is not None:
            self._connection.close()

    def call(
        self,
        method: str,
        path: str,
        document: dict | None = None,
        *,
        body: str | None = None,
    ) -> tuple[int, dict]:
        headers = {}
        payload: bytes | None = None
        if body is not None:
            payload = body.encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif document is not None:
            payload = json.dumps(document).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        assert self._connection is not None
        self._connection.request(method, path, body=payload, headers=headers)
        response = self._connection.getresponse()
        raw = response.read()
        return response.status, (json.loads(raw) if raw else {})


def get(
    host: str,
    port: int,
    path: str,
    *,
    token: str = "",
    body: dict | None = None,
) -> tuple[int, dict]:
    with Client(host, port, token=token) as client:
        return client.call("GET", path, body)


def post(
    host: str, port: int, path: str, document: dict, *, token: str = ""
) -> tuple[int, dict]:
    with Client(host, port, token=token) as client:
        return client.call("POST", path, document)


def sweep_document(**overrides) -> dict:
    sweep = {
        "start_hz": LOW_HZ,
        "stop_hz": HIGH_HZ,
        "points": POINTS,
        "average": 1,
    }
    sweep.update(overrides)
    return sweep


def start(
    host: str, port: int, *, limits: dict | None = None, **sweep
) -> tuple[int, dict]:
    document = {
        "schema_version": SCHEMA_VERSION,
        "sweep": sweep_document(**sweep),
    }
    if limits is not None:
        document["limits"] = limits
    return post(host, port, "/v1/start", document)


def completed_run(
    service: MeasurementService, host: str, port: int, **sweep
) -> str:
    status, document = start(host, port, **sweep)
    assert status == 200, document
    run_id = document["result"]["run_id"]
    state = service.wait_for_result(run_id, timeout=30)
    assert state["state"] == "completed", state
    return run_id


# ----------------------------------------------------------------------
# WebSocket client
# ----------------------------------------------------------------------
def client_frame(opcode: int, payload: bytes = b"", *, masked: bool = True):
    """One client-to-server frame; masked unless a test wants it refused."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    mask_bit = 0x80 if masked else 0
    if length < 126:
        header.append(mask_bit | length)
    elif length < 1 << 16:
        header.append(mask_bit | 126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(mask_bit | 127)
        header.extend(length.to_bytes(8, "big"))
    if not masked:
        return bytes(header) + payload
    key = os.urandom(4)
    body = bytes(byte ^ key[index % 4] for index, byte in enumerate(payload))
    return bytes(header) + key + body


class WebSocket:
    """A small RFC 6455 client: handshake, frames, close."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        path: str = "/v1/events",
        token: str = "",
        timeout: float = 15.0,
    ) -> None:
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.key = base64.b64encode(os.urandom(16)).decode("ascii")
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {self.key}",
            "Sec-WebSocket-Version: 13",
        ]
        if token:
            lines.append(f"Authorization: Bearer {token}")
        self.socket.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = self.socket.recv(1)
            if not chunk:
                raise AssertionError("the server closed during the handshake")
            raw += chunk
        self.handshake = raw.decode("latin-1")

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.socket.sendall(client_frame(0x8, b"\x03\xe8"))
        with contextlib.suppress(OSError):
            self.socket.close()

    def __enter__(self) -> "WebSocket":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- frames --------------------------------------------------------
    def _exactly(self, count: int) -> bytes:
        buffer = b""
        while len(buffer) < count:
            chunk = self.socket.recv(count - len(buffer))
            if not chunk:
                raise AssertionError("the server closed the stream")
            buffer += chunk
        return buffer

    def read_frame(self) -> tuple[int, bytes]:
        first, second = self._exactly(2)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(self._exactly(2), "big")
        elif length == 127:
            length = int.from_bytes(self._exactly(8), "big")
        key = self._exactly(4) if masked else b""
        payload = self._exactly(length)
        if masked:
            payload = bytes(
                byte ^ key[index % 4] for index, byte in enumerate(payload)
            )
        return opcode, payload

    def read_event(self) -> dict:
        """The next text event, answering pings as a client must."""
        while True:
            opcode, payload = self.read_frame()
            if opcode == 0x9:
                self.socket.sendall(client_frame(0xA, payload))
                continue
            if opcode == 0x8:
                raise AssertionError(
                    f"the server closed the stream: {payload!r}"
                )
            assert opcode == 0x1, opcode
            return json.loads(payload)


def free_port(host: str, port: int) -> bool:
    """True when nothing is listening on ``port`` any more."""
    with contextlib.suppress(OSError):
        with socket.create_connection((host, port), timeout=1.0):
            return False
    return True


# ----------------------------------------------------------------------
# Configuration (no socket is ever created)
# ----------------------------------------------------------------------
class TestConfiguration:
    @staticmethod
    def test_loopback_detection_is_syntactic():
        for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.1.2.3"):
            assert is_loopback_host(host), host
        for host in ("0.0.0.0", "192.168.1.4", "example.com", ""):
            assert not is_loopback_host(host), host

    @staticmethod
    def test_a_remote_bind_needs_allow_remote_and_a_token(tmp_path):
        with pytest.raises(ConfigError, match="allow_remote"):
            ServiceConfig(host="0.0.0.0", port=0)
        with pytest.raises(ConfigError, match="token"):
            ServiceConfig(host="0.0.0.0", port=0, allow_remote=True)
        with pytest.raises(ConfigError, match="at least"):
            ServiceConfig(
                host="0.0.0.0", port=0, allow_remote=True, token="short"
            )
        # The guarded combination is accepted, and nothing was bound.
        config = ServiceConfig(
            host="0.0.0.0", port=0, allow_remote=True, token=TOKEN
        )
        assert not config.loopback

    @staticmethod
    def test_an_export_directory_must_already_exist(tmp_path):
        with pytest.raises(ConfigError, match="export_dir"):
            ServiceConfig(port=0, export_dir=str(tmp_path / "missing"))

    @staticmethod
    def test_a_refused_configuration_never_binds(tmp_path):
        with pytest.raises(ConfigError):
            MeasurementService(
                ServiceConfig(host="0.0.0.0", port=0),
                device_factory=_factory(),
            )

    @staticmethod
    def test_the_module_imports_without_qt():
        script = (
            "import sys\n"
            "import NanoVNASaver.WebService as web\n"
            "assert web.SCHEMA_VERSION == 1\n"
            "widgets = [m for m in sys.modules\n"
            "           if m.startswith('PySide6')]\n"
            "print('OK' if not widgets else 'QT:' + widgets[0])\n"
        )
        finished = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert finished.returncode == 0, finished.stderr
        assert finished.stdout.strip() == "OK"


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------
class TestProtocol:
    @staticmethod
    def test_info_describes_a_fresh_server_without_a_device():
        with running_service(device=False) as (service, host, port):
            status, document = get(host, port, "/v1/info")
            assert status == 200
            assert document["schema_version"] == SCHEMA_VERSION
            assert document["ok"] is True
            assert document["command"] == "info"
            result = document["result"]
            assert result["protocol"]["schema_version"] == SCHEMA_VERSION
            assert result["device"] == {"configured": False, "open": False}
            assert result["runs"]["live"] is None
            assert result["export"]["enabled"] is False
            assert result["server"]["port"] == port
            assert result["server"]["loopback"] is True
            assert service.port == port

    @staticmethod
    def test_a_start_without_a_device_is_a_structured_failure():
        with running_service(device=False) as (_service, host, port):
            status, document = start(host, port)
            assert status == 503
            assert document["error"]["code"] == "no_device"

    @staticmethod
    def test_malformed_json_is_reported_as_invalid_json():
        with running_service() as (_service, host, port):
            with Client(host, port) as client:
                status, document = client.call(
                    "POST", "/v1/start", body="{not json"
                )
            assert status == 400
            assert document["error"]["code"] == "invalid_json"
            assert "line" in document["error"]["message"]

    @staticmethod
    def test_an_oversized_body_is_refused_before_it_is_read():
        with running_service(max_body_bytes=1024) as (_service, host, port):
            with socket.create_connection((host, port), timeout=10.0) as sock:
                head = (
                    "POST /v1/start HTTP/1.1\r\n"
                    f"Host: {host}\r\n"
                    f"Content-Length: {1024 * 64}\r\n"
                    "\r\n"
                ).encode("ascii")
                sock.sendall(head)
                raw = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
            assert raw.startswith(b"HTTP/1.1 413")
            document = json.loads(raw.split(b"\r\n\r\n", 1)[1])
            assert document["error"]["code"] == "payload_too_large"
            assert document["error"]["details"]["max_body_bytes"] == 1024

    @staticmethod
    def test_an_unknown_command_is_404():
        with running_service() as (_service, host, port):
            status, document = get(host, port, "/v1/teleport")
            assert status == 404
            assert document["error"]["code"] == "unknown_command"

    @staticmethod
    def test_an_unsupported_path_version_is_refused():
        with running_service() as (_service, host, port):
            status, document = get(host, port, "/v2/info")
            assert status == 400
            assert document["error"]["code"] == "unsupported_version"
            assert document["error"]["details"]["supported_versions"] == [1]

    @staticmethod
    def test_a_body_must_carry_schema_version():
        with running_service() as (_service, host, port):
            status, document = post(
                host, port, "/v1/start", {"sweep": sweep_document()}
            )
            assert status == 400
            assert document["error"]["code"] == "unsupported_version"
            assert "schema_version" in document["error"]["message"]

    @staticmethod
    def test_a_body_version_must_match_the_path():
        with running_service() as (_service, host, port):
            status, document = post(
                host,
                port,
                "/v1/start",
                {"schema_version": 7, "sweep": sweep_document()},
            )
            assert status == 400
            assert document["error"]["code"] == "unsupported_version"

    @staticmethod
    def test_a_query_version_must_match_the_path():
        with running_service() as (_service, host, port):
            status, document = get(host, port, "/v1/info?schema_version=3")
            assert status == 400
            assert document["error"]["code"] == "unsupported_version"
            status, document = get(host, port, "/v1/info?schema_version=1")
            assert status == 200

    @staticmethod
    def test_reading_a_command_that_changes_state_is_405():
        with running_service() as (_service, host, port):
            status, document = get(host, port, "/v1/start")
            assert status == 405
            assert document["error"]["code"] == "method_not_allowed"

    @staticmethod
    def test_a_body_command_must_match_the_path():
        with running_service() as (_service, host, port):
            status, document = post(
                host,
                port,
                "/v1/start",
                {
                    "schema_version": SCHEMA_VERSION,
                    "command": "cancel",
                    "sweep": sweep_document(),
                },
            )
            assert status == 400
            assert "cancel" in document["error"]["message"]

    @staticmethod
    def test_an_unknown_sweep_key_is_named_in_the_error():
        with running_service() as (_service, host, port):
            status, document = post(
                host,
                port,
                "/v1/start",
                {
                    "schema_version": SCHEMA_VERSION,
                    "sweep": {**sweep_document(), "power": 3},
                },
            )
            assert status == 400
            assert "power" in document["error"]["message"]

    @staticmethod
    def test_a_reversed_sweep_is_refused_before_the_device_is_touched():
        with running_service() as (service, host, port):
            status, document = start(host, port, start_hz=HIGH_HZ, stop_hz=1)
            assert status == 400
            assert "below" in document["error"]["message"]
            assert service.info()["runs"]["live"] is None

    @staticmethod
    def test_limits_may_not_be_a_filename():
        with running_service() as (_service, host, port):
            status, document = post(
                host,
                port,
                "/v1/start",
                {
                    "schema_version": SCHEMA_VERSION,
                    "sweep": sweep_document(),
                    "limits": "/etc/passwd",
                },
            )
            assert status == 400
            assert "never opens a path" in document["error"]["message"]

    @staticmethod
    def test_many_clients_are_served_concurrently():
        with running_service() as (_service, host, port):
            results: list[int] = []
            lock = threading.Lock()

            def ask() -> None:
                with Client(host, port) as client:
                    status, _document = client.call("GET", "/v1/info")
                with lock:
                    results.append(status)

            threads = [threading.Thread(target=ask) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
            assert results == [200] * 12

    @staticmethod
    def test_one_connection_can_carry_several_requests():
        with running_service() as (service, host, port):
            with Client(host, port) as client:
                for _ in range(3):
                    status, _document = client.call("GET", "/v1/info")
                    assert status == 200
                # A rejected request whose body was consumed leaves the
                # connection usable, and a run started on it still works.
                status, _document = client.call(
                    "POST", "/v1/start", {"schema_version": 1, "sweep": {}}
                )
                assert status == 400
                status, document = client.call(
                    "POST",
                    "/v1/start",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sweep": sweep_document(),
                    },
                )
                assert status == 200, document
                run_id = document["result"]["run_id"]
            assert (
                service.wait_for_result(run_id, timeout=30)["state"]
                == "completed"
            )

    @staticmethod
    def test_a_refused_body_closes_the_connection():
        """An unread body cannot be skipped, so reuse must not be offered."""
        with running_service(max_body_bytes=512) as (_service, host, port):
            with socket.create_connection((host, port), timeout=10.0) as sock:
                sock.sendall(
                    (
                        "POST /v1/start HTTP/1.1\r\n"
                        f"Host: {host}\r\n"
                        "Content-Length: 4096\r\n"
                        "\r\n"
                    ).encode("ascii")
                )
                raw = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
            head, _, payload = raw.partition(b"\r\n\r\n")
            assert head.startswith(b"HTTP/1.1 413")
            assert b"Connection: close" in head
            assert json.loads(payload)["error"]["code"] == "payload_too_large"

    @staticmethod
    def test_the_client_limit_refuses_rather_than_queues():
        with running_service(max_clients=1) as (service, host, port):
            assert service.begin_client() is True
            try:
                status, document = get(host, port, "/v1/info")
            finally:
                service.end_client()
            assert status == 503
            assert document["error"]["code"] == "too_many_clients"
            assert service.refused_clients == 1
            status, _document = get(host, port, "/v1/info")
            assert status == 200


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
class TestRuns:
    @staticmethod
    def test_a_run_publishes_a_frozen_result_and_its_checks():
        with running_service() as (service, host, port):
            run_id = completed_run(service, host, port)
            status, document = get(host, port, "/v1/last_result")
            assert status == 200
            result = document["result"]
            assert result["run_id"] == run_id
            assert result["sweep"]["points"] == POINTS
            assert result["summary"] == {
                "checked": False,
                "verdict": None,
                "passed": 0,
                "failed": 0,
                "unavailable": 0,
            }
            snapshot = result["snapshot"]
            assert sorted(snapshot["corrected"]) == ["S11"]
            assert len(snapshot["frequencies"]) == POINTS
            # A second read is a fresh document, not the stored mapping.
            result["sweep"]["points"] = 1
            again = get(host, port, "/v1/last_result")[1]["result"]
            assert again["sweep"]["points"] == POINTS

    @staticmethod
    def test_a_one_port_run_never_reports_s21():
        with running_service() as (service, host, port):
            run_id = completed_run(service, host, port)
            document = get(host, port, "/v1/last_result")[1]["result"]
            assert document["sweep"]["channels"] == ["S11"]
            assert "S21" not in document["snapshot"]["corrected"]
            assert "S21" not in document["snapshot"]["raw"]
            checks = get(host, port, f"/v1/checks?run_id={run_id}")[1]
            assert checks["result"]["run_id"] == run_id

    @staticmethod
    def test_asking_a_one_port_device_for_s21_fails_the_run():
        with running_service() as (service, host, port):
            status, document = start(host, port, channels=["S11", "S21"])
            assert status == 200, document
            run_id = document["result"]["run_id"]
            state = service.wait_for_result(run_id, timeout=30)
            assert state["state"] == "failed"
            assert state["error"]["code"] == "unsupported"
            assert state["has_result"] is False
            status, _document = get(host, port, "/v1/last_result")
            assert status == 404

    @staticmethod
    def test_a_second_start_is_refused_while_a_run_is_live():
        slow = _factory("filter", latency=0.05)
        with running_service(factory=slow) as (service, host, port):
            status, document = start(host, port, average=8)
            assert status == 200, document
            run_id = document["result"]["run_id"]
            status, document = start(host, port)
            assert status == 409
            assert document["error"]["code"] == "busy"
            assert document["error"]["details"]["run_id"] == run_id
            state = service.wait_for_result(run_id, timeout=30)
            assert state["state"] == "completed"
            # Once it is finished, the same server accepts a new run.
            completed_run(service, host, port)

    @staticmethod
    def test_cancel_stops_a_live_run_and_never_publishes_data():
        slow = _factory("filter", latency=0.1)
        with running_service(factory=slow) as (service, host, port):
            status, document = start(host, port, average=20)
            assert status == 200, document
            run_id = document["result"]["run_id"]
            status, document = post(
                host,
                port,
                "/v1/cancel",
                {"schema_version": 1, "run_id": run_id},
            )
            assert status == 200
            assert document["result"]["state"] == "cancelling"
            state = service.wait_for_result(run_id, timeout=30)
            assert state["state"] == "cancelled"
            assert state["has_result"] is False
            status, document = get(host, port, f"/v1/checks?run_id={run_id}")
            assert status == 409
            assert document["error"]["code"] == "run_incomplete"

    @staticmethod
    def test_cancelling_a_finished_run_is_a_stale_run():
        with running_service() as (service, host, port):
            run_id = completed_run(service, host, port)
            status, document = post(
                host,
                port,
                "/v1/cancel",
                {"schema_version": SCHEMA_VERSION, "run_id": run_id},
            )
            assert status == 409
            assert document["error"]["code"] == "stale_run"
            assert document["error"]["details"]["state"] == "completed"

    @staticmethod
    def test_an_unknown_run_is_404_everywhere():
        with running_service() as (_service, host, port):
            document = {"schema_version": SCHEMA_VERSION, "run_id": "nope"}
            for path in ("/v1/cancel",):
                status, body = post(host, port, path, document)
                assert status == 404, path
                assert body["error"]["code"] == "run_not_found"
            for path in (
                "/v1/status?run_id=nope",
                "/v1/checks?run_id=nope",
            ):
                status, body = get(host, port, path)
                assert status == 404, path
                assert body["error"]["code"] == "run_not_found"

    @staticmethod
    def test_status_without_a_run_describes_the_server():
        with running_service() as (_service, host, port):
            status, document = get(host, port, "/v1/status")
            assert status == 200
            assert document["result"]["state"] == "idle"
            assert document["result"]["run_id"] is None

    @staticmethod
    def test_checks_names_the_run_and_keeps_unavailable_out_of_pass():
        with running_service() as (service, host, port):
            status, document = start(host, port, limits=UNCOVERED_LIMITS)
            assert status == 200, document
            run_id = document["result"]["run_id"]
            service.wait_for_result(run_id, timeout=30)
            status, document = get(host, port, f"/v1/checks?run_id={run_id}")
            assert status == 200
            result = document["result"]
            assert result["checked"] is True
            assert result["summary"]["unavailable"] == 1
            assert result["summary"]["passed"] == 0
            assert result["summary"]["verdict"] == "unavailable"
            assert result["limits"][0]["verdict"] == "unavailable"
            # The same run's result carries the same verdict.
            latest = get(host, port, "/v1/last_result")[1]["result"]
            assert latest["summary"]["verdict"] == "unavailable"

    @staticmethod
    def test_checks_report_a_failure():
        with running_service() as (service, host, port):
            status, document = start(host, port, limits=IMPOSSIBLE_LIMITS)
            assert status == 200, document
            run_id = document["result"]["run_id"]
            service.wait_for_result(run_id, timeout=30)
            result = get(host, port, f"/v1/checks?run_id={run_id}")[1]["result"]
            assert result["summary"]["verdict"] == "fail"
            assert result["summary"]["failed"] == 1
            assert result["limits"][0]["samples"] > 0

    @staticmethod
    def test_summarise_checks_never_calls_an_unchecked_run_a_pass():
        assert summarise_checks(())["verdict"] is None
        assert summarise_checks(())["checked"] is False

    @staticmethod
    def test_a_run_keeps_its_own_identity_across_a_later_run():
        with running_service() as (service, host, port):
            first = completed_run(service, host, port, points=POINTS)
            second = completed_run(service, host, port, points=POINTS + 1)
            assert first != second
            latest = get(host, port, "/v1/last_result")[1]["result"]
            assert latest["run_id"] == second
            old = get(host, port, f"/v1/status?run_id={first}")[1]["result"]
            assert old["state"] == "completed"
            assert old["sweep"]["points"] == POINTS

    @staticmethod
    def test_retention_forgets_old_runs_but_keeps_the_last_result():
        with running_service(max_retained_runs=2) as (service, host, port):
            first = completed_run(service, host, port)
            second = completed_run(service, host, port)
            third = completed_run(service, host, port)
            info = get(host, port, "/v1/info")[1]["result"]
            assert info["runs"]["retained"] <= 2
            # The most recent result survives pruning by admission.
            assert (
                get(host, port, "/v1/last_result")[1]["result"]["run_id"]
                == third
            )
            # An old run may be gone, but what remains is never confused
            # with a newer one.
            for run_id in (first, second):
                status, _body = get(host, port, f"/v1/status?run_id={run_id}")
                assert status in (200, 404)
            assert service.status(third)["state"] == "completed"


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
class TestExport:
    @staticmethod
    def test_export_writes_inside_the_directory_and_refuses_to_clobber(
        tmp_path,
    ):
        with running_service(export_dir=str(tmp_path)) as (
            service,
            host,
            port,
        ):
            run_id = completed_run(service, host, port)
            document = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "format": "touchstone",
                "filename": "run.s1p",
            }
            status, body = post(host, port, "/v1/export", document)
            assert status == 200, body
            target = tmp_path / "run.s1p"
            assert body["result"]["path"] == os.path.realpath(target)
            assert target.read_text().startswith("!")
            assert body["result"]["bytes"] == target.stat().st_size

            status, body = post(host, port, "/v1/export", document)
            assert status == 409
            assert body["error"]["code"] == "export_exists"

            document["overwrite"] = True
            status, body = post(host, port, "/v1/export", document)
            assert status == 200, body

    @staticmethod
    def test_every_format_writes_its_own_file(tmp_path):
        with running_service(export_dir=str(tmp_path)) as (
            service,
            host,
            port,
        ):
            run_id = completed_run(service, host, port)
            for format_name, filename in (
                ("csv", "report.csv"),
                ("html", "report.html"),
                ("json", "report.json"),
            ):
                status, body = post(
                    host,
                    port,
                    "/v1/export",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "format": format_name,
                        "filename": filename,
                    },
                )
                assert status == 200, body
                written = tmp_path / filename
                assert written.stat().st_size == body["result"]["bytes"]
            as_json = json.loads((tmp_path / "report.json").read_text())
            assert as_json["run_id"] == run_id
            assert (
                (tmp_path / "report.html")
                .read_text()
                .startswith("<!DOCTYPE html>")
            )
            assert (tmp_path / "report.csv").read_text().startswith("trace_id,")

    @staticmethod
    def test_the_extension_must_match_the_data(tmp_path):
        with running_service(export_dir=str(tmp_path)) as (
            service,
            host,
            port,
        ):
            run_id = completed_run(service, host, port)
            status, body = post(
                host,
                port,
                "/v1/export",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "format": "touchstone",
                    "filename": "wrong.s2p",
                },
            )
            assert status == 400
            assert ".s1p" in body["error"]["message"]
            assert os.listdir(tmp_path) == []

    @staticmethod
    def test_a_path_outside_the_directory_is_refused_with_the_reason(
        tmp_path,
    ):
        with running_service(export_dir=str(tmp_path)) as (
            service,
            host,
            port,
        ):
            run_id = completed_run(service, host, port)
            root = os.path.realpath(tmp_path)
            for filename in (
                "../../etc/passwd",
                "/etc/passwd",
                "..",
                "nested/../../escape.csv",
                "with\x00null.csv",
            ):
                status, body = post(
                    host,
                    port,
                    "/v1/export",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "format": "csv",
                        "filename": filename,
                    },
                )
                assert status == 403, filename
                assert body["error"]["code"] == "path_refused"
                details = body["error"]["details"]
                assert details["requested"] == filename
                assert details["allowed_root"] == root
            assert os.listdir(tmp_path) == []

    @staticmethod
    def test_export_is_disabled_without_a_directory():
        with running_service() as (service, host, port):
            run_id = completed_run(service, host, port)
            status, body = post(
                host,
                port,
                "/v1/export",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "format": "csv",
                    "filename": "x.csv",
                },
            )
            assert status == 409
            assert body["error"]["code"] == "export_disabled"

    @staticmethod
    def test_exporting_a_cancelled_run_reports_that_it_has_no_result():
        with running_service(export_dir="") as (service, host, port):
            status, body = post(
                host,
                port,
                "/v1/export",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "missing",
                    "format": "csv",
                    "filename": "x.csv",
                },
            )
            assert status == 404
            assert body["error"]["code"] == "run_not_found"

    @staticmethod
    def test_a_symlink_out_of_the_directory_is_refused(tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "exports"
        root.mkdir()
        (root / "link.csv").symlink_to(outside / "target.csv")
        with running_service(export_dir=str(root)) as (service, host, port):
            run_id = completed_run(service, host, port)
            status, body = post(
                host,
                port,
                "/v1/export",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "format": "csv",
                    "filename": "link.csv",
                },
            )
            assert status == 403
            assert body["error"]["code"] == "path_refused"
            assert body["error"]["details"]["allowed_root"] == str(
                os.path.realpath(root)
            )
            assert not (outside / "target.csv").exists()


# ----------------------------------------------------------------------
# WebSocket event stream
# ----------------------------------------------------------------------
class TestEventStream:
    @staticmethod
    def test_the_handshake_is_a_correct_rfc6455_accept():
        with running_service() as (_service, host, port):
            with WebSocket(host, port) as stream:
                assert stream.handshake.startswith(
                    "HTTP/1.1 101 Switching Protocols"
                )
                assert websocket_accept(stream.key) in stream.handshake
                hello = stream.read_event()
                assert hello["type"] == "hello"
                assert hello["schema_version"] == SCHEMA_VERSION

    @staticmethod
    def test_the_stream_reports_a_whole_run():
        with running_service() as (service, host, port):
            with WebSocket(host, port) as stream:
                assert stream.read_event()["type"] == "hello"
                status, document = start(host, port, average=2)
                assert status == 200, document
                run_id = document["result"]["run_id"]
                seen = []
                event = stream.read_event()
                while True:
                    seen.append(event["type"])
                    assert event["run_id"] == run_id
                    if event["type"] == "completed":
                        break
                    event = stream.read_event()
                assert seen[0] == "started"
                assert "progress" in seen
                assert seen[-1] == "completed"
                assert event["result"]["run_id"] == run_id
                assert event["result"]["snapshot"]["frequencies"]

    @staticmethod
    def test_the_stream_reports_a_cancellation():
        slow = _factory("filter", latency=0.1)
        with running_service(factory=slow) as (service, host, port):
            with WebSocket(host, port) as stream:
                assert stream.read_event()["type"] == "hello"
                status, document = start(host, port, average=20)
                assert status == 200, document
                run_id = document["result"]["run_id"]
                # The run is genuinely live: a cancel is accepted, not
                # rejected as stale, and the stream must report it.
                status, _body = post(
                    host,
                    port,
                    "/v1/cancel",
                    {"schema_version": SCHEMA_VERSION, "run_id": run_id},
                )
                assert status == 200
                types = []
                while True:
                    event = stream.read_event()
                    types.append(event["type"])
                    assert event["run_id"] == run_id
                    if event["type"] in ("cancelled", "failed"):
                        break
                assert "cancelling" in types
                assert types[-1] == "cancelled"
                assert "completed" not in types

    @staticmethod
    def test_ping_is_answered_and_an_unmasked_frame_is_protocol_error():
        with running_service() as (_service, host, port):
            with WebSocket(host, port) as stream:
                assert stream.read_event()["type"] == "hello"
                stream.socket.sendall(client_frame(0x9, b"hi"))
                opcode, payload = stream.read_frame()
                while opcode == 0x1:  # tolerate a server ping event
                    opcode, payload = stream.read_frame()
                assert opcode == 0xA
                assert payload == b"hi"
                stream.socket.sendall(client_frame(0x9, b"bye", masked=False))
                opcode, payload = stream.read_frame()
                while opcode == 0x1:
                    opcode, payload = stream.read_frame()
                assert opcode == 0x8
                assert int.from_bytes(payload[:2], "big") == 1002

    @staticmethod
    def test_a_disconnect_leaves_the_run_servable():
        slow = _factory("filter", latency=0.1)
        with running_service(factory=slow) as (service, host, port):
            stream = WebSocket(host, port)
            assert stream.read_event()["type"] == "hello"
            status, document = start(host, port, average=6)
            assert status == 200, document
            run_id = document["result"]["run_id"]
            stream.close()
            state = service.wait_for_result(run_id, timeout=30)
            assert state["state"] == "completed"
            status, document = get(host, port, "/v1/last_result")
            assert status == 200
            assert document["result"]["run_id"] == run_id

    @staticmethod
    def test_a_slow_subscriber_is_dropped_without_blocking_the_run():
        with running_service(subscriber_queue=1) as (service, host, port):
            subscriber = service.subscribe()
            run_id = completed_run(service, host, port, average=4)
            assert subscriber.closed
            assert subscriber.reason == "lagged"
            assert service.dropped_subscribers == 1
            assert service.subscriber_count == 0
            # The run finished even though nobody drained the subscriber.
            assert service.status(run_id)["state"] == "completed"

    @staticmethod
    def test_a_subscriber_that_keeps_up_sees_every_event_in_order():
        with running_service() as (service, host, port):
            subscriber = service.subscribe()
            try:
                status, _document = start(host, port, average=2)
                assert status == 200
                seen = []
                while True:
                    event = subscriber.poll(5.0)
                    assert event is not None
                    seen.append(event["type"])
                    if event["type"] in ("completed", "failed"):
                        break
            finally:
                service.unsubscribe(subscriber)
            assert seen[0] == "started"
            assert seen[-1] == "completed"

    @staticmethod
    def test_the_subscriber_limit_refuses_rather_than_dropping_silently():
        with running_service(max_subscribers=1) as (service, host, port):
            subscriber = service.subscribe()
            try:
                with pytest.raises(TooManySubscribersError) as info:
                    service.subscribe()
                assert info.value.code == "too_many_subscribers"
            finally:
                service.unsubscribe(subscriber)
            # Once the slot is free again a new stream is accepted.
            with WebSocket(host, port) as stream:
                assert stream.read_event()["type"] == "hello"

    @staticmethod
    def test_an_oversized_websocket_frame_closes_the_stream():
        with running_service(max_body_bytes=512) as (_service, host, port):
            with WebSocket(host, port) as stream:
                assert stream.read_event()["type"] == "hello"
                stream.socket.sendall(client_frame(0x1, b"x" * 4096))
                opcode, payload = stream.read_frame()
                while opcode == 0x1:
                    opcode, payload = stream.read_frame()
                assert opcode == 0x8
                assert int.from_bytes(payload[:2], "big") == 1009

    @staticmethod
    def test_a_plain_get_on_the_event_path_explains_the_upgrade():
        with running_service() as (_service, host, port):
            status, document = get(host, port, "/v1/events")
            assert status == 400
            assert "Upgrade" in document["error"]["message"]


class TestSubscriber:
    @staticmethod
    def test_events_are_delivered_in_order():
        subscriber = Subscriber(4)
        for index in range(4):
            assert subscriber.offer({"index": index}) is True
        assert [subscriber.poll(0.0)["index"] for _ in range(4)] == [0, 1, 2, 3]

    @staticmethod
    def test_a_full_queue_drops_the_subscriber_instead_of_blocking():
        subscriber = Subscriber(1)
        assert subscriber.offer({"index": 0}) is True
        assert subscriber.offer({"index": 1}) is False
        assert subscriber.closed
        assert subscriber.reason == "lagged"
        assert subscriber.dropped == 1
        assert subscriber.offer({"index": 2}) is False

    @staticmethod
    def test_a_closed_subscriber_accepts_nothing():
        subscriber = Subscriber(2)
        subscriber.close("server stopping")
        assert subscriber.offer({"index": 0}) is False
        assert subscriber.reason == "server stopping"


# ----------------------------------------------------------------------
# Access policy and lifecycle
# ----------------------------------------------------------------------
class TestAccessAndLifecycle:
    @staticmethod
    def test_a_configured_token_is_required():
        with running_service(token=TOKEN) as (_service, host, port):
            status, document = get(host, port, "/v1/info")
            assert status == 401
            assert document["error"]["code"] == "unauthorized"
            status, _document = get(
                host, port, "/v1/info", token="wrong-token-value"
            )
            assert status == 401
            status, _document = get(host, port, "/v1/info", token=TOKEN)
            assert status == 200
            with Client(host, port, token=TOKEN) as client:
                status, _document = client.call("GET", "/v1/info")
            assert status == 200

    @staticmethod
    def test_the_token_header_is_accepted_as_an_alternative():
        with running_service(token=TOKEN) as (_service, host, port):
            with socket.create_connection((host, port), timeout=10.0) as sock:
                sock.sendall(
                    (
                        "GET /v1/info HTTP/1.1\r\n"
                        f"Host: {host}\r\n"
                        f"X-NanoVNA-Token: {TOKEN}\r\n\r\n"
                    ).encode("ascii")
                )
                raw = b""
                while b"\r\n\r\n" not in raw:
                    raw += sock.recv(4096)
            assert raw.startswith(b"HTTP/1.1 200")

    @staticmethod
    def test_stopping_releases_the_port():
        service = MeasurementService(
            ServiceConfig(port=0), device_factory=_factory()
        )
        host, port = service.start_server()
        assert not free_port(host, port)
        service.stop()
        assert free_port(host, port)
        # Stopping twice is harmless, so a caller's ``finally`` is safe.
        service.stop()
        assert service.running is False
        assert service.port == 0

    @staticmethod
    def test_stopping_cancels_a_live_run():
        service = MeasurementService(
            ServiceConfig(port=0),
            device_factory=_factory("filter", latency=0.1),
        )
        host, port = service.start_server()
        status, document = start(host, port, average=50)
        assert status == 200, document
        run_id = document["result"]["run_id"]
        service.stop()
        state = service.status(run_id)
        assert state["state"] == "cancelled"

    @staticmethod
    def test_the_context_manager_starts_and_stops(tmp_path):
        with running_service(export_dir=str(tmp_path), device=False) as (
            service,
            host,
            port,
        ):
            assert service.running is True
            assert service.address == (host, port)
        assert service.running is False
        assert free_port(host, port)


class TestFraming:
    @staticmethod
    def test_a_masked_text_frame_decodes_to_its_payload():
        frame = encode_frame(0x1, b"hello")
        assert frame == b"\x81\x05hello"
        masked = client_frame(0x1, b"hello")
        frames = decode_frames(bytearray(masked), 1024)
        assert frames == [(0x1, b"hello")]

    @staticmethod
    def test_a_partial_frame_is_kept_for_the_next_read():
        masked = client_frame(0x1, b"hello")
        buffer = bytearray(masked[:4])
        assert decode_frames(buffer, 1024) == []
        assert bytes(buffer) == masked[:4]
        buffer.extend(masked[4:])
        assert decode_frames(buffer, 1024) == [(0x1, b"hello")]
        assert buffer == bytearray()


@pytest.fixture(scope="module")
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestWindowControl:
    """AUTO-04: the service is opt-in and belongs to the window."""

    @staticmethod
    def test_nothing_listens_until_the_operator_asks(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        window.session.dirty = False
        try:
            assert window.web_service is None
            assert window.service_label.text() == "Stopped"
            assert window.toggle_web_service() is True
            host, port = window.web_service.address
            assert host == "127.0.0.1"
            assert port > 0
            assert "Listening on 127.0.0.1" in window.service_label.text()
            # A second press stops it rather than starting another.
            assert window.toggle_web_service() is False
            assert window.web_service is None
        finally:
            window.session.dirty = False
            window.stop_web_service()
            window.worker.quit()
            window.worker.wait(5000)
            window.deleteLater()

    @staticmethod
    def test_closing_the_window_releases_the_port(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        window.session.dirty = False
        window.toggle_web_service()
        service = window.web_service
        port = service.port
        window.session.dirty = False
        window.close()
        assert window.web_service is None
        assert not service.running
        # The port is free again: a fresh service may bind it.
        from NanoVNASaver.WebService import MeasurementService, ServiceConfig

        replacement = MeasurementService(ServiceConfig(port=port))
        try:
            assert replacement.start_server()[1] == port
        finally:
            replacement.stop()


def _raw_request(  # noqa: PLR0913
    host, port, method, path, *, headers=None, body=None
):
    connection = http.client.HTTPConnection(host, port, timeout=10.0)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


def _start_body(**sweep) -> bytes:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "sweep": sweep_document(**sweep)}
    ).encode("utf-8")


def _wait_until_finished(host, port, run_id, **kwargs) -> dict:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        _, document = get(host, port, f"/v1/status?run_id={run_id}", **kwargs)
        if document["result"]["state"] not in ("starting", "running"):
            return document["result"]
        time.sleep(0.02)
    raise AssertionError("the run did not finish")


class TestBrowserRequestsAreRefused:
    """A loopback bind does not keep out a page in the operator's browser."""

    @staticmethod
    def test_a_cross_site_post_starts_nothing():
        with running_service(factory=_factory()) as (service, host, port):
            status, document = _raw_request(
                host,
                port,
                "POST",
                "/v1/start",
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://evil.example",
                },
                body=_start_body(),
            )
            assert status == 403
            assert document["error"]["code"] == "forbidden_request"
            assert get(host, port, "/v1/status")[1]["result"]["state"] == "idle"

    @staticmethod
    def test_a_simple_text_plain_post_starts_nothing():
        """The no-preflight form, as sent by something that omits Origin."""
        with running_service(factory=_factory()) as (_service, host, port):
            status, document = _raw_request(
                host,
                port,
                "POST",
                "/v1/start",
                headers={"Content-Type": "text/plain"},
                body=_start_body(),
            )
            assert status == 415
            assert document["error"]["code"] == "unsupported_media_type"
            assert get(host, port, "/v1/status")[1]["result"]["state"] == "idle"

    @staticmethod
    def test_a_read_with_an_origin_is_refused():
        with running_service() as (_service, host, port):
            status, _ = _raw_request(
                host,
                port,
                "GET",
                "/v1/info",
                headers={"Origin": "https://evil.example"},
            )
            assert status == 403

    @staticmethod
    def test_a_rebound_host_name_is_refused():
        with running_service() as (_service, host, port):
            status, document = _raw_request(
                host, port, "GET", "/v1/info", headers={"Host": "evil.example"}
            )
            assert status == 403
            assert "loopback" in document["error"]["message"]

    @staticmethod
    def test_loopback_names_are_accepted():
        with running_service() as (_service, host, port):
            for name in (f"127.0.0.1:{port}", f"localhost:{port}", "[::1]"):
                status, _ = _raw_request(
                    host, port, "GET", "/v1/info", headers={"Host": name}
                )
                assert status == 200, name

    @staticmethod
    def test_the_event_stream_refuses_a_web_page():
        with running_service() as (_service, host, port):
            status, _ = _raw_request(
                host,
                port,
                "GET",
                "/v1/events",
                headers={
                    "Origin": "https://evil.example",
                    "Connection": "Upgrade",
                    "Upgrade": "websocket",
                    "Sec-WebSocket-Version": "13",
                    "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
                },
            )
            assert status == 403


class TestExportFailuresAreStructured:
    @staticmethod
    def test_touchstone_without_s11_is_refused_and_writes_nothing(tmp_path):
        with running_service(
            factory=_factory("filter"), export_dir=str(tmp_path)
        ) as (_service, host, port):
            _, document = start(host, port, channels=["S21"])
            run_id = document["result"]["run_id"]
            assert (
                _wait_until_finished(host, port, run_id)["state"] == "completed"
            )
            status, document = post(
                host,
                port,
                "/v1/export",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "format": "touchstone",
                    "filename": "x.s2p",
                },
            )
            assert status == 409
            assert document["error"]["code"] == "export_refused"
            assert "S11" in document["error"]["message"]
            assert list(tmp_path.iterdir()) == []

    @staticmethod
    def test_an_unexpected_defect_is_reported_not_dropped(monkeypatch):
        with running_service() as (service, host, port):

            def broken(_document):
                raise RuntimeError("secret detail")

            monkeypatch.setattr(service, "export", broken)
            status, document = post(
                host,
                port,
                "/v1/export",
                {"schema_version": SCHEMA_VERSION, "run_id": "x"},
            )
            assert status == 500
            assert document["error"]["code"] == "internal_error"
            assert "secret detail" not in document["error"]["message"]


class TestWindowLendsItsDevice:
    @staticmethod
    @pytest.fixture
    def window(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        app = NanoVNASaver()
        app.session.dirty = False
        try:
            yield app
        finally:
            app.session.dirty = False
            app.stop_web_service()
            app.worker.quit()
            app.worker.wait(5000)
            app.deleteLater()

    @staticmethod
    def test_cancelling_a_reversal_keeps_the_running_service(window):
        window.toggle_web_service()
        service = window.web_service
        matrix = object()
        window.s_matrix = matrix
        window.cancel_reversal_capture()
        assert window.web_service is service
        assert window.s_matrix is matrix
        assert service.running

    @staticmethod
    def test_a_service_run_sweeps_and_leaves_the_device_connected(window):
        device = make_simulated_vna("filter", seed=3, noise_floor=0.0)
        points = device.datapoints
        window.vna = device
        window.toggle_web_service()
        token = window.web_service.config.token
        assert token and token in window.service_label.text()
        host, port = window.web_service.address

        status, document = post(
            host,
            port,
            "/v1/start",
            {"schema_version": SCHEMA_VERSION, "sweep": sweep_document()},
            token=token,
        )
        assert status == 200
        run_id = document["result"]["run_id"]
        finished = _wait_until_finished(host, port, run_id, token=token)
        assert finished["state"] == "completed"
        assert device.connected()
        assert device.datapoints == points
        assert window._device_lent is False

    @staticmethod
    def test_the_service_requires_its_token(window):
        window.toggle_web_service()
        host, port = window.web_service.address
        status, document = get(host, port, "/v1/info")
        assert status == 401
        assert document["error"]["code"] == "unauthorized"

    @staticmethod
    def test_no_connected_device_fails_the_run_in_words(window):
        with pytest.raises(RuntimeError, match="no device is connected"):
            window._lend_device_to_service()

    @staticmethod
    def test_a_sweep_and_a_run_exclude_each_other(window, monkeypatch):
        window.vna = make_simulated_vna("filter", seed=3, noise_floor=0.0)
        errors = []
        monkeypatch.setattr(window, "showError", errors.append)

        borrowed = window._lend_device_to_service()
        with pytest.raises(RuntimeError, match="already lent"):
            window._lend_device_to_service()
        window.sweep_start()
        assert not window.worker.isRunning()
        assert "automation service" in errors[0]

        borrowed.disconnect()
        monkeypatch.setattr(window.worker, "isRunning", lambda: True)
        with pytest.raises(RuntimeError, match="running a sweep"):
            window._lend_device_to_service()
