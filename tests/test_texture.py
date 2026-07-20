"""M2 — UV texture baking: xatlas unwrap + project posed images onto an atlas.

Synthetic check: a flat quad viewed head-on by a camera whose image is a solid
colour must bake an albedo of that colour on the covered charts, with UVs in
[0,1] and the face count preserved.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_texture.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def look_at_extrinsic(eye, target, up=(0, 1, 0)):
    """World->camera 3x4 (OpenCV: +Z forward, +Y down)."""
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)                       # forward (+Z)
    r = np.cross(f, np.array(up, float)); r /= np.linalg.norm(r)   # +X
    d = np.cross(f, r)                            # +Y (down)
    R = np.stack([r, d, f], 0)                    # world->cam rows
    t = -R @ np.array(eye, float)
    return np.hstack([R, t[:, None]])


def main():
    from lingbot_plus.meshing import texture_mesh

    # a 2x2 quad on z=0, split into triangles, densified so xatlas has charts
    xs, ys = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))
    zs = np.zeros_like(xs)
    verts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], 1).astype(np.float32)
    faces = []
    n = 12
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1
            faces += [[a, b, c], [b, d, c]]
    faces = np.array(faces, np.uint32)

    # one camera above the quad looking down -Z, image = solid green
    H, W = 240, 320
    fx = fy = 300.0
    K = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], float)
    E = look_at_extrinsic(eye=(0, 0, -3), target=(0, 0, 0), up=(0, 1, 0))
    green = np.zeros((H, W, 3), np.uint8); green[:] = (40, 200, 60)
    images = green[None]                          # (1,H,W,3)

    NV, F, uv, albedo = texture_mesh(
        verts, faces, images, K[None], E[None], tex_size=256, img_stride=1)

    assert len(F) == len(faces), f"face count changed {len(F)} vs {len(faces)}"
    assert uv.min() >= -1e-4 and uv.max() <= 1 + 1e-4, f"uv out of range [{uv.min()},{uv.max()}]"
    assert albedo.shape == (256, 256, 3), f"albedo shape {albedo.shape}"

    covered = albedo.reshape(-1, 3)[albedo.reshape(-1, 3).sum(1) > 0]
    assert len(covered) > 200, f"almost nothing baked ({len(covered)} texels)"
    mean = covered.mean(0)
    assert mean[1] > mean[0] and mean[1] > mean[2], f"albedo not green: {mean.round(1)}"
    print(f"OK — textured quad, {len(F)} tris, albedo covered-mean {mean.round(0)} (green)")


if __name__ == "__main__":
    main()
