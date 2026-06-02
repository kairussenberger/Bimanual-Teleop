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
