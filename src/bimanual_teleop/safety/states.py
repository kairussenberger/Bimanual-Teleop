"""Teleop safety states. The supervisor owns transitions and decides, per side,
whether the operator is allowed to drive the robot (ENGAGED)."""
from __future__ import annotations

import enum


class State(enum.Enum):
    DISCONNECTED = "disconnected"  # no VR / bus down; motors off or damping
    HOMING = "homing"              # moving to a known neutral under limits
    IDLE = "idle"                  # connected, holding neutral, NOT following
    ENGAGED = "engaged"            # clutch/deadman held → following the operator
    ESTOP = "estop"                # latched emergency stop; torque released
