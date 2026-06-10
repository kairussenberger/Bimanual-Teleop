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
# The official ORCA hand description lives in a SIBLING repo (same convention as
# the orca-core editable dependency in pyproject).
ORCA_DESC_DIR = Path(__file__).resolve().parents[3].parent / "orcahand_description" / "v2" / "models"


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


def simplify(tris: np.ndarray, max_tris: int, cache_key: str | None = None) -> np.ndarray:
    """Quadric mesh simplification (fast-simplification) for dense organic shells
    where largest-face decimation leaves confetti. Falls back to `decimate` when
    the optional dependency is missing. Results are cached on disk per
    (cache_key, max_tris) because the source STLs run to ~100k tris."""
    if len(tris) <= max_tris:
        return tris
    cache = None
    if cache_key:
        cache = Path("/tmp") / f"bimanual_meshcache_{cache_key}_{max_tris}.npy"
        if cache.exists():
            try:
                return np.load(cache)
            except Exception:
                pass
    try:
        import fast_simplification as fs
    except ImportError:
        return decimate(tris, max_tris)
    verts = tris.reshape(-1, 3)
    uniq, inv = np.unique(np.round(verts, 6), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3).astype(np.int64)
    reduction = max(0.0, min(0.999, 1.0 - max_tris / len(faces)))
    out_v, out_f = fs.simplify(uniq.astype(np.float64), faces, target_reduction=reduction)
    out = np.asarray(out_v)[np.asarray(out_f)]
    if cache is not None:
        try:
            np.save(cache, out)
        except Exception:
            pass
    return out


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


# --------------------------------------------------------------------------- #
# Official ORCA hand (sibling repo orcahand_description, v2 MJCF + STLs). Root
# body `<side>_tower` == the rig's flange→ORCA transform == this repo's EE site
# frame, so geom transforms compose directly with the EE world pose.
# --------------------------------------------------------------------------- #
def orca_description_available() -> bool:
    return (ORCA_DESC_DIR / "mjcf" / "orcahand_right.mjcf").exists()


def load_orca_hand(side: str, max_tris_per_geom: int = 150):
    """(model, data, items) for the REAL ORCA hand; items carry per-geom `tris`
    (geom frame), `jid`, `place`, and an `rgb` skin/structure tint."""
    mjcf = ORCA_DESC_DIR / "mjcf"
    adoc = ET.parse(mjcf / f"orcahand_{side}.mjcf").getroot()
    default = adoc.find("default")
    dscale = 1.0
    if default is not None and default.find("mesh") is not None:
        dscale = float(default.find("mesh").get("scale", "1").split()[0])
    meshfile = {}
    asset_src = adoc.find("asset")
    for m in asset_src.findall("mesh"):
        rel = m.get("file")
        path = None
        for base in (mjcf, ORCA_DESC_DIR.parent, ORCA_DESC_DIR):
            cand = (base / rel).resolve()
            if cand.exists():
                path = cand
                break
        if path is None:
            raise FileNotFoundError(f"orca mesh not found: {rel}")
        meshfile[m.get("name")] = (str(path), float(m.get("scale", str(dscale)).split()[0]))
        m.set("file", str(path))
    root = ET.Element("mujoco", {"model": f"orcahand_{side}"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    root.append(asset_src)
    body = ET.parse(mjcf / f"orcahand_{side}_body.xml").getroot().find("body")
    for g in body.iter("geom"):
        g.set("type", "mesh")
        if "material" in g.attrib:                 # materials live in the scene file
            del g.attrib["material"]
    ET.SubElement(root, "worldbody").append(body)
    tmp = f"/tmp/orcahand_render_{side}.xml"
    ET.ElementTree(root).write(tmp)
    model = pin.buildModelFromMJCF(tmp)
    geom = pin.buildGeomFromMJCF(model, tmp, pin.GeometryType.VISUAL)
    items = []
    for g in geom.geometryObjects:
        mesh_name = g.name.rsplit("Geom_", 1)[0]
        # geom names are '<body>Geom_<n>'; the mesh ref is recoverable from the
        # geometry object itself when pin kept the path
        path = getattr(g, "meshPath", "") or ""
        scale = dscale
        if mesh_name in meshfile:
            path, scale = meshfile[mesh_name]
        elif path:
            for name, (p, s) in meshfile.items():
                if Path(p).name == Path(path).name:
                    scale = s
                    break
        if not path or not Path(path).exists():
            continue
        skin = "skin" in Path(path).name.lower()
        key = f"orca_{side}_{Path(path).stem}"
        items.append({"tris": simplify(load_stl(path) * scale, max_tris_per_geom, cache_key=key),
                      "jid": g.parentJoint,
                      "place": g.placement.homogeneous,
                      "rgb": (0.91, 0.88, 0.83) if skin else (0.23, 0.24, 0.27)})
    return model, model.createData(), items


def orca_q_from_degrees(model, joints_deg: dict, side: str) -> np.ndarray:
    """Map this repo's 17 ORCA joint angles (degrees, hardware names) onto the
    description model's joint vector (radians, '<side>_<short>' names)."""
    from ..hands.joint_map import orca_to_sim_short
    q = np.zeros(model.nq)
    for name, deg in joints_deg.items():
        short = orca_to_sim_short(name)
        jname = f"{side}_{short}"
        if model.existJointName(jname):
            j = model.joints[model.getJointId(jname)]
            lo = model.lowerPositionLimit[j.idx_q]
            hi = model.upperPositionLimit[j.idx_q]
            q[j.idx_q] = float(np.clip(np.radians(float(deg)), lo, hi))
    return q
