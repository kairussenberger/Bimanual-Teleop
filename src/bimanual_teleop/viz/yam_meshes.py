"""Real YAM visual geometry for renderers (the GIF/dashboard look).

The repo's MJCF source files split assets and the body tree, so this module
synthesizes a loadable single-arm MJCF, lets Pinocchio place the visual geoms,
and parses the binary STLs with numpy — no trimesh, no MuJoCo. Used by
scripts/render_session.py (matplotlib GIFs) and scripts/dashboard.py (browser),
so both draw the identical robot.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

MJCF_DIR = Path(__file__).resolve().parents[1] / "sim" / "models" / "yam_real" / "mjcf"
STAND_DIR = Path(__file__).resolve().parents[1] / "sim" / "models" / "yam_real" / "assets" / "stand"


def load_stl(path: str) -> np.ndarray:
    """Binary STL → (n, 3, 3) triangle vertices."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], "little")
    rec = np.frombuffer(raw[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    return rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)


def decimate(tris: np.ndarray, max_tris: int) -> np.ndarray:
    """Keep the LARGEST faces: fine CAD tessellation means every-Nth sampling
    leaves triangle dust, while the biggest faces carry the visual mass."""
    if len(tris) <= max_tris:
        return tris
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    return tris[np.argsort(area)[-max_tris:]]


def load_arm_meshes(side: str, max_tris_per_link: int = 800):
    """(pin model, data, items) for one arm; items = [{tris, jid, place}] with
    triangle vertices in the GEOM frame."""
    root = ET.Element("mujoco", {"model": f"yam_{side}"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    asset = ET.SubElement(root, "asset")
    meshfile = {}
    src_asset = ET.parse(MJCF_DIR / f"yam_{side}.mjcf").getroot().find("asset")
    for m in src_asset.findall("mesh"):
        p = str((MJCF_DIR / m.get("file")).resolve())
        meshfile[m.get("name")] = (p, float(m.get("scale", "1 1 1").split()[0]))
        m.set("file", p)
        asset.append(m)
    body = ET.parse(MJCF_DIR / f"yam_{side}_body.xml").getroot().find("body")
    for g in body.iter("geom"):
        g.set("type", "mesh")                  # pin's parser wants the type explicit
    ET.SubElement(root, "worldbody").append(body)
    tmp = f"/tmp/yam_render_{side}.xml"
    ET.ElementTree(root).write(tmp)
    model = pin.buildModelFromMJCF(tmp)
    geom = pin.buildGeomFromMJCF(model, tmp, pin.GeometryType.VISUAL)
    items = []
    for g in geom.geometryObjects:
        path, scale = meshfile[g.name.rsplit("Geom_", 1)[0]]
        items.append({"tris": decimate(load_stl(path) * scale, max_tris_per_link),
                      "jid": g.parentJoint,
                      "place": g.placement.homogeneous})
    return model, model.createData(), items


def load_stand_meshes(z_offset: float, max_tris_per_part: int = 600) -> list[np.ndarray]:
    """The AgileX stand frame as static world-frame triangle soups. The six
    frame_part STLs share one assembly frame in millimetres; `z_offset` is the
    rig's stand.pos z (lifts the feet onto the floor)."""
    out = []
    for f in sorted(STAND_DIR.glob("*.stl")):
        tris = decimate(load_stl(str(f)), max_tris_per_part) * 0.001
        tris[:, :, 2] += float(z_offset)
        out.append(tris)
    return out


def geom_transforms(model, data, items, q, base_T) -> list[np.ndarray]:
    """World 4×4 per geom for joint vector q (FK included)."""
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    return [np.asarray(base_T @ data.oMi[it["jid"]].homogeneous @ it["place"]) for it in items]


def world_tris(model, data, items, q, base_T) -> list[np.ndarray]:
    """Transformed triangle soup per geom (used by the matplotlib renderer)."""
    out = []
    for it, T in zip(items, geom_transforms(model, data, items, q, base_T), strict=True):
        out.append(it["tris"] @ T[:3, :3].T + T[:3, 3])
    return out
