"""Chunk stitching: recover a known similarity between overlapping clouds.

Two failures the dense/robust fit fixes vs the old camera-centre Umeyama:
  1. camera centres over a straight walk are near-collinear -> scale is
     ill-conditioned (tiny residual on a wrong scale). Dense, well-distributed
     overlap points constrain scale in all 3 axes.
  2. some overlap points disagree between chunks (occlusion/bad depth) ->
     outliers. A trimmed refit ignores them; plain Umeyama is dragged off.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_stitch.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rot(axis, deg):
    a = np.deg2rad(deg)
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s,   c + y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s,   z*y*(1-c)+x*s, c + z*z*(1-c)],
    ])


def test_robust_recovers_scale_with_outliers():
    from lingbot_plus.chunked import robust_umeyama

    rng = np.random.default_rng(0)
    src = rng.uniform(-1, 1, (4000, 3))          # well-distributed 3D
    s_true, R_true = 2.06, rot(np.array([0.2, 1.0, 0.3]), 27.0)
    t_true = np.array([1.5, -0.4, 0.9])
    dst = s_true * src @ R_true.T + t_true
    dst += rng.normal(0, 0.002, dst.shape)       # mild noise
    # 25% gross outliers
    n_out = len(dst) // 4
    dst[:n_out] += rng.normal(0, 5.0, (n_out, 3))

    s, R, t = robust_umeyama(src, dst, trim=0.35, iters=3)
    assert abs(s - s_true) / s_true < 0.03, f"scale {s:.4f} vs {s_true} (>3% off)"
    print(f"  robust w/ 25% outliers: scale {s:.4f} (true {s_true}) OK")


def test_collinear_cameras_vs_dense():
    """Near-collinear points give an unreliable scale; the same transform from
    well-distributed points is accurate. Documents WHY we feed dense overlap
    points, not camera centres."""
    from lingbot_plus.chunked import robust_umeyama

    rng = np.random.default_rng(1)
    s_true, R_true, t_true = 2.06, rot(np.array([0, 0, 1.0]), 15.0), np.array([0.3, 0.2, 0.1])

    line = np.zeros((12, 3))
    line[:, 0] = np.linspace(0, 1, 12)
    line += rng.normal(0, 0.01, line.shape)      # camera walk: ~1D
    dst_line = s_true * line @ R_true.T + t_true
    s_line, _, _ = robust_umeyama(line, dst_line, trim=0.0, iters=1)

    dense = rng.uniform(-1, 1, (4000, 3))
    dst_dense = s_true * dense @ R_true.T + t_true + rng.normal(0, 0.002, (4000, 3))
    s_dense, _, _ = robust_umeyama(dense, dst_dense, trim=0.0, iters=1)

    assert abs(s_dense - s_true) / s_true < 0.02, f"dense scale {s_dense:.4f} inaccurate"
    print(f"  collinear scale={s_line:.4f} (noisy) · dense scale={s_dense:.4f} "
          f"(true {s_true}) — dense accurate OK")


def main():
    test_robust_recovers_scale_with_outliers()
    test_collinear_cameras_vs_dense()
    print("OK — robust dense stitch similarity")


if __name__ == "__main__":
    main()
