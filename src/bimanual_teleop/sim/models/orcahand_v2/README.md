# Vendored ORCA hand model (simplified)

Render-grade copy of the official ORCA hand v2 description —
<https://github.com/orcahand/orcahand_description> (MIT, see LICENSE here) —
with every mesh quadric-simplified to ≤600 triangles for the
dashboard/GIF renderers. Kinematics (joints, placements) are unmodified.

If the full-resolution description is cloned as a sibling repo
(`../orcahand_description`), the loaders prefer it automatically.

Regenerate with: `uv run python scripts/vendor_orcahand.py`
