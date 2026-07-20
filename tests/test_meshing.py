"""Unit test for lingbot_plus.meshing — synthetic colored box room -> Poisson mesh GLB.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_meshing.py
Plain asserts (no pytest dependency).
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_box_room(n_per_face=6000, rng=None):
    """Points on the 5 interior faces of a 4x3x2.5 room (no ceiling), color per face."""
    rng = rng or np.random.default_rng(0)
    W, L, H = 4.0, 3.0, 2.5
    faces = [
        # (fixed axis, value, color)
        (2, 0.0, (0.6, 0.4, 0.2)),   # floor  z=0  brown
        (0, 0.0, (0.9, 0.1, 0.1)),   # wall x=0    red
        (0, W,   (0.1, 0.9, 0.1)),   # wall x=W    green
        (1, 0.0, (0.1, 0.1, 0.9)),   # wall y=0    blue
        (1, L,   (0.9, 0.9, 0.1)),   # wall y=L    yellow
    ]
    pts, cols = [], []
    for axis, val, c in faces:
        p = rng.uniform(0, 1, (n_per_face, 3)) * np.array([W, L, H])
        p[:, axis] = val
        pts.append(p)
        cols.append(np.tile(c, (n_per_face, 1)))
    return np.concatenate(pts), np.concatenate(cols)


def main():
    from lingbot_plus.meshing import mesh_from_points

    pts, cols = make_box_room()
    # slight sensor noise
    pts = pts + np.random.default_rng(1).normal(0, 0.008, pts.shape)
    cam_centers = np.array([[2.0, 1.5, 1.4]])          # camera in room center
    frame_idx = np.zeros(len(pts), dtype=np.int32)     # all points from frame 0

    with tempfile.TemporaryDirectory() as td:
        out_glb = os.path.join(td, "mesh.glb")
        verts, faces, vcols = mesh_from_points(
            pts, cols, cam_centers, frame_idx, out_glb=out_glb)

        # mesh exists and is substantial
        assert len(faces) > 1000, f"too few triangles: {len(faces)}"
        assert len(verts) == len(vcols), "vertex/color count mismatch"

        # bbox stays near the input bbox (Poisson balloon must be cropped)
        lo, hi = verts.min(0), verts.max(0)
        assert np.all(lo > np.array([-0.3, -0.3, -0.3])), f"bbox low {lo}"
        assert np.all(hi < np.array([4.3, 3.3, 2.8])), f"bbox high {hi}"

        # colors interpolated: floor vertices (z near 0, interior) should be brownish
        floor = verts[:, 2] < 0.05
        interior = (verts[:, 0] > 0.5) & (verts[:, 0] < 3.5) & \
                   (verts[:, 1] > 0.5) & (verts[:, 1] < 2.5)
        fc = vcols[floor & interior]
        assert len(fc) > 50, "no floor vertices found"
        mean = fc.mean(0)
        assert mean[0] > mean[2], f"floor not brownish (R>B expected): {mean}"

        # GLB written and loadable with same topology
        assert os.path.exists(out_glb), "mesh.glb not written"
        import trimesh
        m = trimesh.load(out_glb, force="mesh")
        assert len(m.faces) == len(faces), "GLB face count mismatch"
        assert m.visual.kind is not None, "GLB lost vertex colors"

    print(f"OK — {len(verts)} verts, {len(faces)} tris, floor color {mean.round(2)}")


if __name__ == "__main__":
    main()
