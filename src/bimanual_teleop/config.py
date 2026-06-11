"""Load the rig configuration (config/rig.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RIG = REPO_ROOT / "config" / "rig.yaml"

SIDES = ("left", "right")


def load_rig(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_RIG
    with open(p) as f:
        return yaml.safe_load(f)


def side_axis(mapping: dict, key: str, side: str, default) -> list:
    """Read a hand-local axis that may be PER-SIDE ({left: ..., right: ...}) or a
    legacy shared flat list — the two hands are mirrored anatomically, so the
    per-side form is the accurate one (see config/rig.yaml)."""
    v = mapping.get(key, default)
    if isinstance(v, dict):
        v = v.get(side, default)
    return v
