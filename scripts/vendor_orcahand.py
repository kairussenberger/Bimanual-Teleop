#!/usr/bin/env python
"""Vendor a render-grade simplified copy of the official ORCA hand description
into this repo, so fresh clones draw the REAL hand model with no sibling setup.

Source: github.com/orcahand/orcahand_description (MIT) cloned as a sibling repo.
Output: src/bimanual_teleop/sim/models/orcahand_v2/models/{mjcf,assets} —
identical structure to the original so the same loader handles both, with every
STL quadric-simplified (~600 tris per geom, plenty for the dashboard/GIFs).

    uv run python scripts/vendor_orcahand.py            # regenerate the vendored copy
"""
from __future__ import annotations

import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.viz.yam_meshes import ORCA_DESC_SIBLING, load_stl, simplify  # noqa: E402

OUT = REPO_ROOT / "src" / "bimanual_teleop" / "sim" / "models" / "orcahand_v2" / "models"
TARGET_TRIS = 600


def write_stl(path: Path, tris: np.ndarray) -> None:
    """Minimal binary STL writer (normals recomputed per face)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    with open(path, "wb") as f:
        f.write(b"bimanual-teleop vendored simplified ORCA mesh".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            f.write(np.concatenate([n[i], tris[i].reshape(-1)]).astype("<f4").tobytes())
            f.write(b"\0\0")


def main() -> int:
    if not (ORCA_DESC_SIBLING / "mjcf" / "orcahand_right.mjcf").exists():
        print(f"source description not found at {ORCA_DESC_SIBLING} — clone "
              "github.com/orcahand/orcahand_description next to this repo first")
        return 1
    (OUT / "mjcf").mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    for side in ("left", "right"):
        (OUT / "assets" / side).mkdir(parents=True, exist_ok=True)
        shutil.copy(ORCA_DESC_SIBLING / "mjcf" / f"orcahand_{side}_body.xml",
                    OUT / "mjcf" / f"orcahand_{side}_body.xml")
        tree = ET.parse(ORCA_DESC_SIBLING / "mjcf" / f"orcahand_{side}.mjcf")
        for m in tree.getroot().find("asset").findall("mesh"):
            rel = m.get("file")
            src = (ORCA_DESC_SIBLING.parent / rel).resolve()
            tris = load_stl(str(src))                       # native mm units
            simp = simplify(tris, TARGET_TRIS, cache_key=None)
            dst = OUT / "assets" / side / Path(rel).name
            write_stl(dst, simp)
            m.set("file", f"models/assets/{side}/{Path(rel).name}")
            total_in += len(tris)
            total_out += len(simp)
        tree.write(OUT / "mjcf" / f"orcahand_{side}.mjcf")
    shutil.copy(ORCA_DESC_SIBLING.parents[1] / "LICENSE", OUT.parent / "LICENSE")
    (OUT.parent / "README.md").write_text(
        "# Vendored ORCA hand model (simplified)\n\n"
        "Render-grade copy of the official ORCA hand v2 description —\n"
        "<https://github.com/orcahand/orcahand_description> (MIT, see LICENSE here) —\n"
        f"with every mesh quadric-simplified to ≤{TARGET_TRIS} triangles for the\n"
        "dashboard/GIF renderers. Kinematics (joints, placements) are unmodified.\n\n"
        "If the full-resolution description is cloned as a sibling repo\n"
        "(`../orcahand_description`), the loaders prefer it automatically.\n\n"
        "Regenerate with: `uv run python scripts/vendor_orcahand.py`\n")
    print(f"vendored {total_in} → {total_out} tris into {OUT.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
