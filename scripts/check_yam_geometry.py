#!/usr/bin/env python
"""Verify the runtime YAM Pinocchio geometry against the source MJCF files.

The runtime no longer loads MuJoCo, but its kinematics are intentionally derived
from the measured MJCF body trees under `sim/models/yam_real/mjcf`. This check
prevents the Python/Unity path from drifting into a detached hand-coded robot.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from bimanual_teleop.arms import yam_pin


REPO = Path(__file__).resolve().parents[1]
MJCF = REPO / "src" / "bimanual_teleop" / "sim" / "models" / "yam_real" / "mjcf"
SIDES = ("left", "right")
TOL = 1e-9


def floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise AssertionError(f"missing body {name}")


def read_chain(side: str) -> list[dict]:
    root = ET.parse(MJCF / f"yam_{side}_body.xml").getroot()
    out = []
    for i in range(1, 7):
        body = find_body(root, f"{side}_arm_link{i}")
        joint = body.find("joint")
        if joint is None:
            raise AssertionError(f"{side} link{i}: missing joint")
        out.append({
            "name": f"j{i}",
            "body_pos": floats(body.get("pos"), (0.0, 0.0, 0.0)),
            "axis": floats(joint.get("axis"), (0.0, 0.0, 1.0)),
            "joint_pos": floats(joint.get("pos"), (0.0, 0.0, 0.0)),
            "range": floats(joint.get("range"), (0.0, 0.0)),
        })
    return out


def read_site(side: str, suffix: str) -> dict:
    root = ET.parse(MJCF / f"yam_{side}_body.xml").getroot()
    for site in root.iter("site"):
        if site.get("name") == f"{side}_{suffix}":
            return {
                "pos": floats(site.get("pos"), (0.0, 0.0, 0.0)),
                "euler": floats(site.get("euler"), (0.0, 0.0, 0.0)),
            }
    raise AssertionError(f"missing site {side}_{suffix}")


def assert_vec(name: str, got, expected) -> None:
    if not np.allclose(got, expected, atol=TOL, rtol=0.0):
        raise AssertionError(f"{name}: got {np.asarray(got)}, expected {np.asarray(expected)}")


def main() -> int:
    ref_chain = None
    for side in SIDES:
        chain = read_chain(side)
        if ref_chain is None:
            ref_chain = chain
        else:
            for left, right in zip(ref_chain, chain, strict=True):
                for key in ("body_pos", "axis", "joint_pos", "range"):
                    assert_vec(f"{side} chain matches left {left['name']} {key}", right[key], left[key])

        if len(chain) != len(yam_pin._CHAIN):
            raise AssertionError(f"{side}: expected {len(yam_pin._CHAIN)} joints, found {len(chain)}")
        for parsed, const in zip(chain, yam_pin._CHAIN, strict=True):
            name, axis, body_pos, joint_pos, lo, hi = const
            if parsed["name"] != name:
                raise AssertionError(f"{side}: joint order mismatch {parsed['name']} != {name}")
            assert_vec(f"{side} {name} axis", parsed["axis"], axis)
            assert_vec(f"{side} {name} body_pos", parsed["body_pos"], body_pos)
            assert_vec(f"{side} {name} joint_pos", parsed["joint_pos"], joint_pos)
            assert_vec(f"{side} {name} limits", parsed["range"], (lo, hi))

        wrist = read_site(side, "wrist")
        wrist_parent, wrist_pos, wrist_euler = yam_pin._WRIST_SITE
        if wrist_parent != 3:
            raise AssertionError("runtime wrist site must be attached to j4/link4")
        assert_vec(f"{side} wrist site pos", wrist["pos"], wrist_pos)
        assert_vec(f"{side} wrist site euler", wrist["euler"], wrist_euler)

        ee = read_site(side, "ee")
        ee_parent, ee_pos, ee_euler = yam_pin._EE_SITE[side]
        if ee_parent != 5:
            raise AssertionError(f"{side}: runtime EE site must be attached to j6/link6")
        assert_vec(f"{side} ee site pos", ee["pos"], ee_pos)
        assert_vec(f"{side} ee site euler", ee["euler"], ee_euler)

    # Smoke-build both Pinocchio models, including site frames, after the source
    # comparison so this check also catches broken runtime geometry construction.
    for side in SIDES:
        model = yam_pin.build_arm_model(side)
        for frame in (f"{side}_wrist", f"{side}_ee"):
            if not model.existFrame(frame):
                raise AssertionError(f"{side}: missing Pinocchio frame {frame}")

    print("YAM Pinocchio geometry matches source MJCF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
