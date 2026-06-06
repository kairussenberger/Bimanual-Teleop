"""Observability tests (spec Section 6): telemetry ring buffer + CSV, rate meter,
logger, and the Rerun logger's no-op degradation when the dep is absent.

    uv run pytest tests/test_observability.py -q
"""
from __future__ import annotations

import csv

import numpy as np

from bimanual_teleop.logging_utils import RateMeter, TelemetryRing, get_logger
from bimanual_teleop.viz.rerun_log import RerunLogger


def test_ring_wraps_at_capacity_latest_wins():
    ring = TelemetryRing(capacity=10)
    for i in range(25):
        ring.append(t=float(i), pos_err=i * 0.1)
    assert len(ring) == 10                      # bounded
    assert ring.latest()["t"] == 24.0          # newest kept
    assert ring.records()[0]["t"] == 15.0      # oldest 15 dropped


def test_ring_to_csv_roundtrip(tmp_path):
    ring = TelemetryRing(capacity=100)
    ring.append(t=0.0, pos_err=1.0, q=[1.0, 2.0, 3.0])
    ring.append(t=1.0, rot_err=2.5)            # different keys -> union columns
    p = tmp_path / "telemetry.csv"
    ring.to_csv(str(p))
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"t", "pos_err", "q", "rot_err"}
    assert rows[0]["q"] == "1;2;3"             # sequence stringified
    assert rows[1]["t"] == "1.0" and rows[1]["pos_err"] == ""   # missing -> blank


def test_rate_meter_converges():
    rm = RateMeter(alpha=0.2)
    for _ in range(200):
        rm.update(1.0 / 120.0)
    assert abs(rm.hz - 120.0) < 1.0


def test_logger_is_levelled_and_singleton():
    """Logger emits at/above its level and is a process-wide singleton. (It uses
    propagate=False to avoid double-logging, so we capture via its own handler
    rather than pytest's caplog, which relies on propagation.)"""
    import logging
    lg = get_logger("test")
    seen: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, r):
            seen.append(r.getMessage())

    lg.addHandler(_Cap())
    lg.warning("left-hand tracking LOST")
    lg.debug("noise below INFO")                 # filtered by the INFO level
    assert any("LOST" in m for m in seen)
    assert all("noise" not in m for m in seen)
    assert get_logger("test") is lg             # singleton


def test_rerun_logger_noop_when_disabled():
    """Disabled (or rerun-absent) logger must accept every call without raising."""
    rl = RerunLogger(enabled=False)
    assert rl.enabled is False
    rl.set_time(1.0)
    rl.transform("a", np.eye(4))
    rl.triad("b", [0, 0, 0], np.eye(3))
    rl.points("c", np.zeros((5, 3)))
    rl.scalar("d", 1.0)
    rl.text("e", "hello")                       # no exception == pass


def test_rerun_logger_default_construct_safe():
    """Constructing with defaults must not raise even if rerun is not installed
    (it just stays disabled)."""
    rl = RerunLogger(spawn=False)
    assert isinstance(rl.enabled, bool)
    rl.scalar("x", 0.0)                         # safe regardless of enabled state
