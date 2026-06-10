"""Rerun telemetry backbone — optional live 3D + time-series dashboard.

Rerun is useful for inspecting the headless Python runtime: timestamped 3D
transforms (operator body frame, commanded/achieved EE frames), scalar time-series
(pose error, loop rate, solve time, joint margins), and text/status that can be
scrubbed and replayed.

This is an OPTIONAL dependency (`uv sync --extra telemetry`). The logger degrades to
a silent no-op when `rerun` is not installed or when constructed with enabled=False,
so the core teleop runtime and headless CI never need it. Import is lazy for the
same reason.

    rr_log = RerunLogger(spawn=True)          # opens the Rerun viewer
    rr_log.set_time(t)
    rr_log.transform("world/ee_R/achieved", T_world)     # 4x4
    rr_log.scalar("err/right/pos_cm", 1.3)
    rr_log.text("status", "clutch ENGAGED", level="INFO")
"""
from __future__ import annotations

import numpy as np

# X=red, Y=green, Z=blue triad colours (uint8), matching viz/overlay.AXIS_RGB.
_AXIS_RGB = np.array([[255, 50, 50], [50, 255, 50], [90, 90, 255]], dtype=np.uint8)


class RerunLogger:
    """Thin wrapper over the Rerun SDK that is safe to call unconditionally."""

    def __init__(self, app_id: str = "bimanual_teleop", *, spawn: bool = False,
                 enabled: bool = True, save_path: str | None = None):
        self.rr = None
        if not enabled:
            return
        try:
            import rerun as rr
        except Exception:                       # not installed -> silent no-op
            return
        self.rr = rr
        try:
            rr.init(app_id)
            # NOTE: rr.save() REDIRECTS the stream to the file — it does not add a
            # sink. With both a viewer and a file requested, tee explicitly via
            # set_sinks, or the spawned viewer sits empty while the .rrd gets
            # everything (the bug this replaced).
            if spawn and save_path:
                rr.spawn(connect=False)         # launch the viewer app
                rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(save_path)))
            elif spawn:
                rr.spawn()                      # launch viewer + connect the stream
            elif save_path:
                rr.save(str(save_path))         # file only (open later with `rerun FILE`)
        except Exception:
            self.rr = None                      # init failed -> stay a no-op

    @property
    def enabled(self) -> bool:
        return self.rr is not None

    # --- time ---------------------------------------------------------------- #
    def set_time(self, seconds: float) -> None:
        if self.rr is None:
            return
        try:
            self.rr.set_time("sim_time", duration=float(seconds))
        except Exception:
            try:                                # older Rerun API
                self.rr.set_time_seconds("sim_time", float(seconds))
            except Exception:
                pass

    # --- 3D ------------------------------------------------------------------ #
    def transform(self, path: str, T: np.ndarray) -> None:
        """Log a 4x4 homogeneous transform as a Rerun Transform3D."""
        if self.rr is None:
            return
        T = np.asarray(T, float).reshape(4, 4)
        self.rr.log(path, self.rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))

    def triad(self, path: str, pos, R, length: float = 0.12) -> None:
        """Log an RGB axis triad at pos with orientation R (3x3) as 3 arrows."""
        if self.rr is None:
            return
        R = np.asarray(R, float).reshape(3, 3)
        origins = np.tile(np.asarray(pos, float), (3, 1))
        vectors = (R * length).T               # columns of R, scaled -> rows
        self.rr.log(path, self.rr.Arrows3D(origins=origins, vectors=vectors, colors=_AXIS_RGB))

    def points(self, path: str, pts, radius: float = 0.006) -> None:
        if self.rr is None:
            return
        self.rr.log(path, self.rr.Points3D(np.asarray(pts, float).reshape(-1, 3), radii=radius))

    def linestrip(self, path: str, pts, radius: float = 0.004, color=None) -> None:
        """Log a polyline (e.g. an arm link chain) as one LineStrips3D entity."""
        if self.rr is None:
            return
        pts = np.asarray(pts, float).reshape(-1, 3)
        kw = {"radii": radius}
        if color is not None:
            kw["colors"] = np.asarray(color, dtype=np.uint8)
        self.rr.log(path, self.rr.LineStrips3D([pts], **kw))

    def arrow(self, path: str, origin, vector, color=(255, 200, 40)) -> None:
        if self.rr is None:
            return
        self.rr.log(path, self.rr.Arrows3D(origins=[np.asarray(origin, float)],
                                           vectors=[np.asarray(vector, float)],
                                           colors=np.asarray(color, dtype=np.uint8)))

    def clear(self, path: str, *, recursive: bool = True) -> None:
        """Remove an entity (e.g. the command marker on clutch release)."""
        if self.rr is None:
            return
        try:
            self.rr.log(path, self.rr.Clear(recursive=recursive))
        except Exception:
            pass

    # --- scalars / text ------------------------------------------------------ #
    def scalar(self, path: str, value: float) -> None:
        if self.rr is None:
            return
        try:
            self.rr.log(path, self.rr.Scalars(float(value)))
        except Exception:
            try:
                self.rr.log(path, self.rr.Scalar(float(value)))   # older API
            except Exception:
                pass

    def text(self, path: str, text: str, level: str = "INFO") -> None:
        if self.rr is None:
            return
        try:
            self.rr.log(path, self.rr.TextLog(text, level=level))
        except Exception:
            pass
