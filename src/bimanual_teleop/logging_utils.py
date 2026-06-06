"""Structured logging + a per-tick telemetry ring buffer (spec Section 6B).

Two cheap, dependency-free pieces of the observability layer:
  - `get_logger`  — a levelled, human-readable logger so warnings read like the spec
    examples: "WARN left-hand tracking LOST", "WARN J3 at limit".
  - `TelemetryRing` — a fixed-capacity ring of per-tick records (latest-wins, drops
    oldest), dumpable to CSV for offline plotting. Log every signal in Section 6.
  - `RateMeter` — an EWMA loop-rate estimator for the HUD ("loop rate (Hz)").

The richer live dashboard (3D transforms, scalar time-series, scrub/replay) is the
Rerun backbone in viz/rerun_log.py; this module is the always-on, no-extra-deps
fallback that also feeds the in-sim HUD.
"""
from __future__ import annotations

import csv
import logging
from collections import deque
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "teleop", level: int = logging.INFO) -> logging.Logger:
    """A process-wide levelled logger with a compact, readable format."""
    global _CONFIGURED
    if not _CONFIGURED:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        root = logging.getLogger("teleop")
        root.addHandler(h)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    lg = logging.getLogger(name if name.startswith("teleop") else f"teleop.{name}")
    lg.setLevel(level)
    return lg


class RateMeter:
    """Exponentially-weighted loop-rate estimator (Hz). Feed it dt each tick."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self._dt_ewma: float | None = None

    def update(self, dt: float) -> float:
        dt = max(float(dt), 1e-6)
        self._dt_ewma = dt if self._dt_ewma is None else (
            self.alpha * dt + (1 - self.alpha) * self._dt_ewma)
        return self.hz

    @property
    def hz(self) -> float:
        return 0.0 if not self._dt_ewma else 1.0 / self._dt_ewma


class TelemetryRing:
    """Fixed-capacity ring buffer of per-tick telemetry dicts (latest-wins). Each
    record is an arbitrary flat dict of scalars/short sequences; CSV export uses the
    union of all keys seen (missing values blank)."""

    def __init__(self, capacity: int = 20000):
        self.capacity = int(capacity)
        self._buf: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    def append(self, record: dict[str, Any] | None = None, **fields: Any) -> None:
        rec = dict(record) if record else {}
        rec.update(fields)
        self._buf.append(rec)

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> dict[str, Any] | None:
        return self._buf[-1] if self._buf else None

    def records(self) -> list[dict[str, Any]]:
        return list(self._buf)

    def to_csv(self, path: str) -> str:
        """Write all buffered records to CSV. Sequence values are stringified."""
        rows = self.records()
        keys: list[str] = []
        seen = set()
        for r in rows:                      # stable column order = first-seen order
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: _csv_val(r.get(k)) for k in keys})
        return path


def _fmt(x: Any) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _csv_val(v: Any) -> Any:
    if hasattr(v, "tolist"):                # numpy array/scalar -> python first
        if getattr(v, "ndim", 0) == 0:
            return _fmt(v.item())
        v = v.tolist()
    if isinstance(v, (list, tuple)):        # one formatter for every sequence
        return ";".join(_fmt(x) for x in v)
    return v
