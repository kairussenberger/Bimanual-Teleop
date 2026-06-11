"""Tiny localhost command channel into a RUNNING teleop engine.

The dashboard owns the engine *process* (spawn/stop buttons), but runtime
actions — starting the operator neutral-pose calibration — need to reach the
live engine without a restart. This is a deliberately minimal stdlib TCP
line-JSON server (mirroring the render bridge's no-deps philosophy): one
request per connection, newline-delimited JSON both ways, bound to 127.0.0.1.

    request:  {"cmd": "calibrate"}\n
    reply:    {"ok": true, "msg": "calibration started"}\n

Commands only set thread-safe request flags on the engine; the engine's own
tick (control-loop thread) consumes them. Nothing here touches IK state."""
from __future__ import annotations

import json
import socket
import threading

from .logging_utils import get_logger

log = get_logger("control")


class ControlServer:
    COMMANDS = ("calibrate", "calibrate_cancel", "calibrate_clear", "status")

    def __init__(self, engine, port: int, host: str = "127.0.0.1"):
        self.engine = engine
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, int(port)))
        self._sock.listen()
        self._sock.settimeout(0.25)
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.endpoint = f"tcp://{host}:{int(port)}"

    def _loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(1.0)
                line = conn.makefile("r", encoding="utf-8").readline()
                reply = self._handle(line)
                conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, line: str) -> dict:
        try:
            cmd = json.loads(line or "{}").get("cmd", "")
        except json.JSONDecodeError:
            return {"ok": False, "msg": "bad JSON"}
        if cmd not in self.COMMANDS:
            return {"ok": False, "msg": f"unknown cmd {cmd!r}"}
        if cmd == "calibrate":
            self.engine.request_calibration()
            return {"ok": True, "msg": "calibration started"}
        if cmd == "calibrate_cancel":
            self.engine.request_calibration_cancel()
            return {"ok": True, "msg": "calibration cancelled"}
        if cmd == "calibrate_clear":
            self.engine.request_calibration_clear()
            return {"ok": True, "msg": "calibration cleared"}
        return {"ok": True, "calib": self.engine.calib_status,
                "applied": self.engine.calib_summary}

    def close(self) -> None:
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass


def send_command(cmd: str, port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> dict:
    """One-shot client (used by the dashboard): send a command, return the reply."""
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.sendall((json.dumps({"cmd": cmd}) + "\n").encode("utf-8"))
        line = sock.makefile("r", encoding="utf-8").readline()
    return json.loads(line or "{}")
