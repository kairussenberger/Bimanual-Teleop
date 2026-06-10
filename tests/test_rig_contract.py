from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from bimanual_teleop.config import load_rig


def _load_contract():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_rig_contract.py"
    spec = importlib.util.spec_from_file_location("check_rig_contract", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rig_contract_rejects_drifted_arm_base_position(monkeypatch):
    contract = _load_contract()
    rig = copy.deepcopy(load_rig())
    rig["arms"]["left"]["base_pos"][2] += 0.01
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="base_pos changed"):
        contract.main()


def test_rig_contract_rejects_drifted_arm_base_quaternion(monkeypatch):
    contract = _load_contract()
    rig = copy.deepcopy(load_rig())
    rig["arms"]["right"]["base_quat"][0] += 0.001
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="base_quat changed"):
        contract.main()


def test_rig_contract_rejects_disabled_body_relative_default(monkeypatch):
    contract = _load_contract()
    rig = copy.deepcopy(load_rig())
    rig["vr"]["body_relative"] = False
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="body_relative"):
        contract.main()


def test_rig_contract_rejects_non_finite_torso_offset(monkeypatch):
    contract = _load_contract()
    rig = copy.deepcopy(load_rig())
    rig["vr"]["torso_from_head"][1] = float("nan")
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="torso_from_head.*finite"):
        contract.main()


def test_rig_contract_rejects_bad_mapping_defaults(monkeypatch):
    contract = _load_contract()
    rig = copy.deepcopy(load_rig())
    rig["mapping"]["pos_scale"] = 0.0
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="pos_scale"):
        contract.main()

    # Stale orientation-calibration knobs must not creep back into the config.
    rig = copy.deepcopy(load_rig())
    rig["mapping"]["abs_orientation"] = True
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="abs_orientation"):
        contract.main()


def test_rig_contract_rejects_nonzero_default_calib_seconds(monkeypatch):
    contract = _load_contract()
    rig = copy.deepcopy(load_rig())
    rig["vr"]["calib_seconds"] = 5.0
    monkeypatch.setattr(contract, "load_rig", lambda: rig)

    with pytest.raises(AssertionError, match="calib_seconds"):
        contract.main()
