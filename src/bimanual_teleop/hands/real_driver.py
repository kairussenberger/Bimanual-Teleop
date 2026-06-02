"""Real ORCA hand driver, wrapping orca_core.OrcaHand.

Same set_hand seam as the sim. Lifecycle (from orca-teleop): connect → init_joints
→ [loop: set_joint_positions(dict_degrees)] → disable_torque + disconnect. Works on
macOS and Linux (USB-serial Feetech). Import-guarded so the package imports without
orca_core present, though orca_core is a normal dependency here.
"""
from __future__ import annotations


class RealHand:
    def __init__(self, model_name: str | None = None, config_path: str | None = None):
        from orca_core import OrcaHand
        self.hand = OrcaHand(config_path=config_path, model_name=model_name)
        ok, msg = self.hand.connect()
        if not ok:
            raise RuntimeError(f"ORCA hand connect failed ({model_name}): {msg}")
        self.hand.init_joints(move_to_neutral=True)

    def set_joint_positions(self, joints_deg: dict) -> None:
        self.hand.set_joint_positions(joints_deg)   # degrees, immediate (num_steps=1)

    def release(self) -> None:
        try:
            self.hand.disable_torque()
        finally:
            self.hand.disconnect()
