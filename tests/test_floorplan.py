"""Unit test for floor detection robustness under scale drift / fragmentation.

The failure this locks down (observed on real-room-hq): RANSAC finds a dense
plane the camera sits right next to (a wall / near-camera clutter) with MORE
inliers than the true floor, and the old "largest plane wins" picks it —
camera height above the chosen plane was ~2% of scene extent (cameras on it).

Fix under test: a plane whose cameras sit essentially ON it (camh/extent below
a small ratio) is demoted, so the true floor below the cameras wins even with
fewer inliers. Scale-invariant (ratio), so it holds without metric calibration.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_floorplan.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _plane_points(axis, val, span, n, rng, other_lo=0.0):
    """n random points on the plane {coord[axis]==val}, spread over `span` in
    the other two axes (starting at other_lo)."""
    p = rng.uniform(0, 1, (n, 3))
    p = other_lo + p * span
    p[:, axis] = val
    return p


def make_scene(rng):
    """A room whose biggest, densest plane is a near-camera wall clutter, not
    the floor — the exact trap that broke real-room-hq. Cameras hover at z≈1.5.
    """
    span = np.array([4.0, 3.0, 2.5])
    floor = _plane_points(2, 0.0, span, 12000, rng)                 # true floor z=0
    wall = _plane_points(0, 0.0, span, 14000, rng)                  # wall x=0
    # dense clutter plane at camera height (a shelf/curtain the camera skims)
    clutter = _plane_points(2, 1.5, np.array([4.0, 3.0, 0.0]), 26000, rng)
    pts = np.concatenate([floor, wall, clutter])
    pts += rng.normal(0, 0.006, pts.shape)

    # cameras: upright, roving at z≈1.5 (right at the clutter plane)
    n_cam = 60
    C = np.stack([rng.uniform(0.5, 3.5, n_cam),
                  rng.uniform(0.5, 2.5, n_cam),
                  rng.uniform(1.4, 1.6, n_cam)], axis=1)
    # world->cam extrinsic with identity-ish rotation (looking down-ish)
    E = np.tile(np.eye(4), (n_cam, 1, 1)).astype(np.float64)
    for i in range(n_cam):
        R = np.eye(3)
        E[i, :3, :3] = R
        E[i, :3, 3] = -R @ C[i]
    return pts, C, E


def main():
    import open3d as o3d
    from lingbot_plus.floorplan import find_floor

    rng = np.random.default_rng(0)
    pts, C, E = make_scene(rng)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))

    floor = find_floor(pcd, C, o3d, unit=1.0, scaled=False, extent=extent)
    n = floor["n"] / np.linalg.norm(floor["n"])

    # chosen plane must be horizontal (normal ≈ ±Z), i.e. the floor or clutter,
    # not the vertical wall
    assert abs(n[2]) > 0.9, f"floor normal not vertical: {n}"

    # and it must be the LOW floor (cameras well above), not the clutter at cam height
    z0 = float(np.median(floor["pts"][:, 2]))
    cam_z = float(np.median(C[:, 2]))
    assert z0 < 0.5, f"picked plane at z={z0:.2f}, expected floor near 0 (clutter trap at 1.5)"
    assert cam_z - z0 > 1.0, f"cameras only {cam_z - z0:.2f} above floor — near-camera plane picked"

    print(f"OK — floor z0={z0:.3f}, cam_z={cam_z:.2f}, "
          f"cam_height={cam_z - z0:.2f}, normal={n.round(2)}")


if __name__ == "__main__":
    main()
