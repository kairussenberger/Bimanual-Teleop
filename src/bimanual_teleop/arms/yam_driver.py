"""Real i2rt YAM arm driver (Linux + SocketCAN only).

Import-guarded: importing this module does NOT require the i2rt SDK; it's only
needed when you actually construct a YamArm (i.e. on the Linux control host with
the arms wired up). On macOS/sim this module imports fine but YamArm() will raise
a clear error. The factory keeps the YAM's 400 ms motor watchdog enabled (the
hardware backstop deadman). The stock gripper is removed (gripper_type='no_gripper')
so the flange is free for the ORCA hand, which runs on its own bus.

Bring-up (Ubuntu):
    sudo ip link set can0 up type can bitrate 1000000
    git clone https://github.com/i2rt-robotics/i2rt && uv pip install -e i2rt
"""
from __future__ import annotations

import numpy as np


class YamArm:
    """6-DoF YAM over CAN. Exposes the same state()/command() seam as the sim arm."""

    def __init__(self, channel: str, gripper_type: str = "no_gripper"):
        try:
            from i2rt.robots.motor_chain_robot import get_yam_robot
        except ImportError as e:  # pragma: no cover - hardware only
            raise RuntimeError(
                "i2rt SDK not installed. Real YAM control is Linux/SocketCAN only — "
                "install on the control host: `uv pip install -e i2rt` (see module docstring)."
            ) from e
        self.robot = get_yam_robot(channel=channel, gripper_type=gripper_type)

    def state(self) -> np.ndarray:
        """Current joint positions (6,) in radians."""
        return np.asarray(self.robot.get_joint_pos(), dtype=float)[:6]

    def command(self, q) -> None:
        """Command 6 joint targets (radians); MIT-mode PD/impedance at ~250 Hz."""
        self.robot.command_joint_pos(np.asarray(q, dtype=float)[:6])

    def close(self) -> None:  # pragma: no cover - hardware only
        for fn in ("close", "disconnect", "stop"):
            if hasattr(self.robot, fn):
                getattr(self.robot, fn)()
                return
