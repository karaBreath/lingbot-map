"""Chunked runner: unlimited-length sequences on backends with fragile allocators.

Splits the (strided) frame list into overlapping chunks, runs worker.py in a
fresh subprocess per chunk (fresh DirectML heap each time), then stitches the
chunks into one coordinate frame with a similarity transform (Umeyama) fitted
on the overlapping frames' camera centers — monocular scale differs per chunk,
so rotation+translation alone is not enough.

Usage:
    python -m lingbot_plus.chunked --model_path W.pt --image_folder DIR \
        --stride 3 --chunk_size 120 --overlap 10 --out_dir scans/room1 [--port 8099]
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_plus.frames import list_images  # noqa: E402


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) with dst ≈ s * R @ src + t. Nx3 each."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_umeyama(src, dst, trim=0.3, iters=3):
    """Umeyama similarity with iterative outlier trimming.

    Fit (s, R, t), drop the `trim` fraction of correspondences with the largest
    residual, refit — `iters` times. Resists depth-disagreement outliers between
    two chunks' overlap and the ill-conditioning of near-collinear inputs (the
    dense caller feeds well-distributed points so scale is constrained in 3D).
    """
    s, R, t = umeyama(src, dst)
    if trim <= 0 or len(src) < 8:
        return s, R, t
    for _ in range(iters):
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        keep = res <= np.quantile(res, 1.0 - trim)
        if keep.sum() < 8:
            break
        s, R, t = umeyama(src[keep], dst[keep])
    return s, R, t


def extr_to_centers_rots(extr: np.ndarray):
    """extr: [N,3,4] in the pipeline's stored convention (w2c, OpenCV cam-from-world
    — same thing demo.py saves and the viewer consumes). Returns camera centers
    C = -R^T t and cam->world rotations R_c2w = R^T."""
    R_wc = extr[:, :3, :3]
    t_wc = extr[:, :3, 3]
    R_c2w = np.transpose(R_wc, (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_c2w, t_wc)
    return C, R_c2w


def overlap_similarity(prev, cur, ov, unproject, conf_thr=3.0, max_pts=20000, seed=0):
    """Similarity mapping cur (native frame) -> prev (already-global frame),
    fit on DENSE corresponding overlap points instead of the ~ov camera centres.

    The overlap frames are the same video frames in both chunks, so matching by
    (frame, pixel) gives direct 3D correspondences. Camera centres over a
    straight walk are near-collinear (scale ill-conditioned); dense well-spread
    points fix that. Falls back to camera centres if the overlap is too sparse.
    Returns (s, R, t, residual, method).
    """
    prev_pts = unproject(prev["depth"][-ov:], prev["extrinsic"][-ov:], prev["intrinsic"][-ov:])
    cur_pts = unproject(cur["depth"][:ov], cur["extrinsic"][:ov], cur["intrinsic"][:ov])
    dst = prev_pts.reshape(-1, 3)
    src = cur_pts.reshape(-1, 3)
    keep = (prev["depth_conf"][-ov:].reshape(-1) > conf_thr) & \
           (cur["depth_conf"][:ov].reshape(-1) > conf_thr)
    src, dst = src[keep], dst[keep]
    if len(src) >= 200:
        if len(src) > max_pts:
            sel = np.random.default_rng(seed).choice(len(src), max_pts, replace=False)
            src, dst = src[sel], dst[sel]
        s, R, t = robust_umeyama(src, dst, trim=0.3, iters=3)
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        keep2 = res <= np.quantile(res, 0.7)
        return s, R, t, float(res[keep2].mean()), f"dense({keep2.sum()})"
    # fallback: camera centres
    sc, _ = extr_to_centers_rots(cur["extrinsic"][:ov])
    dc, _ = extr_to_centers_rots(prev["extrinsic"][-ov:])
    s, R, t = umeyama(sc, dc)
    return s, R, t, float(np.linalg.norm(s * sc @ R.T + t - dc, axis=1).mean()), "cam-centre"


def sim3_extr(extr: np.ndarray, s: float, Rg: np.ndarray, tg: np.ndarray):
    """Apply world-frame similarity (s, Rg, tg) to stored w2c extrinsics."""
    C, R_c2w = extr_to_centers_rots(extr)
    C2 = s * np.einsum("ij,nj->ni", Rg, C) + tg
    R_c2w2 = np.einsum("ij,njk->nik", Rg, R_c2w)
    R_wc2 = np.transpose(R_c2w2, (0, 2, 1))
    t_wc2 = -np.einsum("nij,nj->ni", R_wc2, C2)
    out = extr.copy()
    out[:, :3, :3] = R_wc2
    out[:, :3, 3] = t_wc2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=16)
    ap.add_argument("--image_ext", default=".jpg,.png,.JPG")
    ap.add_argument("--port", type=int, default=0, help="if >0, open viser viewer on merged result")
    ap.add_argument("--conf_threshold", type=float, default=1.5)
    args = ap.parse_args()

    n = len(list_images(args.image_folder, args.image_ext)[:: args.stride])
    if n == 0:
        raise SystemExit("no frames found")

    step = args.chunk_size - args.overlap
    chunks = []
    start = 0
    while start < n:
        end = min(start + args.chunk_size, n)
        chunks.append((start, end))
        if end >= n:
            break
        start += step
    print(f"[chunked] {n} strided frames -> {len(chunks)} chunks {chunks}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    npzs = []
    t0 = time.time()
    for k, (s_, e_) in enumerate(chunks):
        out = os.path.join(args.out_dir, f"chunk_{k:03d}.npz")
        npzs.append(out)
        if os.path.exists(out):
            print(f"[chunked] chunk {k} exists, skip", flush=True)
            continue
        cmd = [
            sys.executable, "-m", "lingbot_plus.worker",
            "--model_path", args.model_path, "--image_folder", args.image_folder,
            "--start", str(s_), "--end", str(e_), "--stride", str(args.stride),
            "--out", out, "--backend", args.backend,
            "--image_size", str(args.image_size),
            "--kv_cache_sliding_window", str(args.kv_cache_sliding_window),
            "--image_ext", args.image_ext,
        ]
        print(f"[chunked] chunk {k}/{len(chunks)-1}: frames {s_}..{e_}", flush=True)
        # Give the driver a beat to reclaim the previous worker's heap — chunk
        # N can OOM if launched the instant chunk N-1 exits (observed on
        # DirectML: identical chunk fails right after a success, passes on
        # retry). One automatic retry with a longer pause covers it.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for attempt in (1, 2):
            time.sleep(5 if attempt == 1 else 15)
            r = subprocess.run(cmd, cwd=repo_root)
            if r.returncode == 0:
                break
            print(f"[chunked] chunk {k} attempt {attempt} failed (exit {r.returncode})"
                  + ("; retrying after pause" if attempt == 1 else ""), flush=True)
        if r.returncode != 0:
            raise SystemExit(f"[chunked] chunk {k} FAILED after retry — see output above")
    print(f"[chunked] all chunks done in {time.time()-t0:.0f}s", flush=True)

    # ── Stitch ───────────────────────────────────────────────────────────────
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

    merged = {"world_points": [], "world_points_conf": [], "images": [], "extrinsic": [], "intrinsic": []}
    prev = None  # previous chunk data already in global frame
    for k, f in enumerate(npzs):
        d = dict(np.load(f))
        if prev is not None:
            ov = args.overlap
            # overlap frames: last `ov` of prev == first `ov` of this chunk.
            # Fit on DENSE corresponding overlap points (camera centres over a
            # straight walk are near-collinear -> scale ill-conditioned).
            s, R, t, resid, method = overlap_similarity(
                prev, d, ov, unproject_depth_map_to_point_map,
                conf_thr=args.conf_threshold)
            print(f"[stitch] chunk {k}: scale={s:.4f} residual={resid:.4f} [{method}]", flush=True)
            d["extrinsic"] = sim3_extr(d["extrinsic"], s, R, t)
            d["depth"] = d["depth"] * s  # metric content scales with the map
            # drop duplicated overlap frames from this chunk
            for key in ("depth", "depth_conf", "extrinsic", "intrinsic", "images"):
                d[key] = d[key][ov:]
        pts = unproject_depth_map_to_point_map(d["depth"], d["extrinsic"], d["intrinsic"])
        merged["world_points"].append(pts.astype(np.float32))
        merged["world_points_conf"].append(d["depth_conf"])
        merged["images"].append(d["images"].astype(np.float32) / 255.0)
        merged["extrinsic"].append(d["extrinsic"])
        merged["intrinsic"].append(d["intrinsic"])
        prev = d

    pred = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    out_npz = os.path.join(args.out_dir, "merged.npz")
    np.savez_compressed(out_npz, **pred)
    print(f"[chunked] merged {pred['world_points'].shape[0]} frames -> {out_npz}", flush=True)

    if args.port > 0:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=pred, port=args.port, vis_threshold=args.conf_threshold,
            downsample_factor=10, point_size=0.00001, use_point_map=True,
        )
        print(f"[chunked] viewer at http://localhost:{args.port}", flush=True)
        viewer.run()


if __name__ == "__main__":
    main()
