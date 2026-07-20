"""R2 — top-down floor plan, room segmentation, and room dimensions.

Pipeline (CPU only, Open3D + numpy — never touches the GPU backend):
  merged.npz -> confidence-filtered cloud -> RANSAC floor plane -> rotate so
  floor = z=0 -> slice wall-height band -> 2D occupancy grid -> walls mask ->
  flood-fill enclosed free space -> rooms (count, WxL, area) -> plan.png/.svg
  + plan.json in the scan folder.

Honesty rules: areas are estimate-grade; unscanned walls stay as gaps (we
never invent geometry the camera did not see). If meta.json has no scale
(R1 not run), dimensions are reported in model units and labeled as such.

Usage:
    python -m lingbot_plus.floorplan --scan_dir scans/tum_room
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scan(scan_dir, conf_threshold):
    npz = os.path.join(scan_dir, "merged.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"not found: {npz} — run lingbot_plus.chunked first")
    d = np.load(npz)
    pts = d["world_points"].reshape(-1, 3)
    conf = d["world_points_conf"].reshape(-1)
    keep = conf > conf_threshold
    pts = pts[keep].astype(np.float64)

    scale = None
    meta_p = os.path.join(scan_dir, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            scale = json.load(f).get("scale_m_per_unit")
    cam_centers = None
    if "extrinsic" in d:
        E = d["extrinsic"]
        R_wc, t_wc = E[:, :3, :3], E[:, :3, 3]
        cam_centers = -np.einsum("nji,nj->ni", R_wc, t_wc)  # -R^T t
    return pts, scale, cam_centers


def find_floor(pcd, cam_centers, o3d, unit=1.0, scaled=False, extent=None):
    """RANSAC out up to 6 planes; floor = the plane most consistent with
    'cameras are above it, moving flat across it'.

    Two traps this resists, both seen on real scans:

    - **Desk trap** (TUM fr1): the biggest plane is a desk surface, not the
      floor. When the scan is metric-calibrated, a plausible handheld camera
      height (0.8-2.2 m) beats raw inlier count.
    - **Wall / near-camera trap** (real-room-hq, strong scale drift): the
      densest plane is a wall or clutter the camera skims, so 'largest plane'
      picks it (camera ~2% of scene extent above the chosen plane). Two
      scale-invariant signals fix this without metric calibration:
        1. cameras must sit meaningfully ABOVE the plane (camh/extent), not on
           it — kills near-camera clutter/ceiling;
        2. cameras spread little ALONG the floor normal (they walk at roughly
           constant height) but far across it — a wall has cameras spread wide
           along its normal, so penalise high spread-along-normal.

    `extent` = scene diagonal (model units); computed from the cloud if omitted.
    """
    pts_all = np.asarray(pcd.points)
    if extent is None:
        extent = float(np.linalg.norm(pts_all.max(0) - pts_all.min(0)))
    thr = 0.01 if scaled else extent * 0.004          # extent-relative when uncalibrated
    cam_spread = float(np.linalg.norm(cam_centers.std(0))) if cam_centers is not None else 0.0

    rest = pcd
    cands = []
    for _ in range(6):
        if len(rest.points) < 500:
            break
        model, inliers = rest.segment_plane(
            distance_threshold=thr, ransac_n=3, num_iterations=1000)
        plane_pts = rest.select_by_index(inliers)
        n = np.asarray(model[:3])
        d0 = model[3]
        cam_h = 0.0
        spread_n = 0.0
        if cam_centers is not None:
            sd = cam_centers @ n + d0
            if np.median(sd) < 0:
                n, d0, sd = -n, -d0, -sd
            cam_h = float(np.median(sd))
            if cam_spread > 1e-9:
                spread_n = float((cam_centers @ n).std()) / cam_spread
        cands.append({"n": n, "d": d0, "inliers": len(inliers), "cam_h": cam_h,
                      "camh_ratio": cam_h / extent if extent else 0.0,
                      "spread_n": spread_n, "pts": np.asarray(plane_pts.points)})
        rest = rest.select_by_index(inliers, invert=True)
    if not cands:
        raise SystemExit("no plane found — cloud too sparse?")

    def score(c):
        s = float(c["inliers"])
        if cam_centers is not None:
            if c["camh_ratio"] < 0.02:      # cameras sit ON it -> clutter/ceiling/wall-base
                s *= 0.15
            # walls: cameras roam far along the plane normal; floor: they stay flat
            s *= max(0.15, 1.0 - c["spread_n"])
        if scaled:
            h_m = c["cam_h"] * unit
            if 0.8 <= h_m <= 2.2:           # plausible handheld height above FLOOR
                s *= 3.0
            elif h_m < 0.5:                 # desk/table surface
                s *= 0.2
        return s

    return max(cands, key=score)


def segment_rooms(floor_mask, wall_mask, per_cell, scaled, min_room_m2, min_cells,
                  bridge_m=0.35, wall_gap_m=0.25, min_wall_m=0.5):
    """Rooms from occupancy masks, robust to fragmented walls/floor.

    `per_cell` = metres per grid cell when scaled, else model-units per cell
    (distances below are taken as ratios of it, so the morphology scales with
    the grid in both cases). Three robustness steps:
      1. close floor gaps (bridge_m) — a patchy single room reconnects instead
         of shattering into sub-min fragments;
      2. keep only genuine walls — bridge small wall gaps (wall_gap_m) then drop
         components shorter than min_wall_m, so stray furniture specks and dashed
         reconstruction noise stop punching false dividers;
      3. free space = bridged floor minus genuine walls -> connected components.

    Returns (labels, rooms) with rooms sorted large->small and ids assigned.
    """
    from scipy import ndimage

    def cells(d):
        return max(1, int(round(d / per_cell)))

    st = np.ones((3, 3))
    floor_c = ndimage.binary_closing(floor_mask, structure=st, iterations=cells(bridge_m))
    wall_c = ndimage.binary_closing(wall_mask, structure=st, iterations=cells(wall_gap_m))

    # keep wall components whose longest extent reaches min_wall_m (real walls),
    # drop short/isolated specks (furniture edges, drift noise)
    wl, wn = ndimage.label(wall_c)
    strong = np.zeros_like(wall_c)
    minlen = cells(min_wall_m)
    for i in range(1, wn + 1):
        comp = wl == i
        ys, xs = np.nonzero(comp)
        span = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        if span >= minlen:
            strong |= comp

    free = floor_c & ~strong
    free = ndimage.binary_opening(free, structure=st)

    labels, n_lab = ndimage.label(free)
    cell_area = per_cell ** 2
    rooms = []
    for i in range(1, n_lab + 1):
        m = labels == i
        area = float(m.sum() * cell_area)
        if scaled and area < min_room_m2:
            continue
        if not scaled and m.sum() < min_cells:
            continue
        ys, xs = np.nonzero(m)
        w = float((xs.max() - xs.min() + 1) * per_cell)
        h = float((ys.max() - ys.min() + 1) * per_cell)
        rooms.append({"area": round(area, 2), "width": round(w, 2), "length": round(h, 2),
                      "bbox_cells": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                      "mask_label": i})
    rooms.sort(key=lambda r: -r["area"])
    for j, r in enumerate(rooms):
        r["id"] = j + 1
    return labels, rooms


def rotation_to_z(n):
    """Rotation matrix sending unit vector n -> +Z."""
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = float(n @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--conf_threshold", type=float, default=3.0,
                    help="depth-confidence filter; higher = cleaner cloud (default stricter than viewer)")
    ap.add_argument("--cell_m", type=float, default=0.04, help="grid cell size (meters, if scale known)")
    ap.add_argument("--wall_band", type=float, nargs=2, default=[0.3, 2.0],
                    help="height band (m) treated as wall evidence")
    ap.add_argument("--min_room_m2", type=float, default=1.5)
    args = ap.parse_args()

    import open3d as o3d
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Tahoma"   # Thai glyphs everywhere (legend too)
    import matplotlib.pyplot as plt
    from scipy import ndimage

    pts, scale, cam_centers = load_scan(args.scan_dir, args.conf_threshold)
    unit = scale if scale else 1.0
    unit_name = "m" if scale else "หน่วยโมเดล"
    print(f"[floorplan] {len(pts):,} points after conf>{args.conf_threshold}"
          f" · scale={'%.4f m/unit' % scale if scale else 'NONE (run R1 measure to calibrate)'}",
          flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.02 / unit if scale else
                                float(np.linalg.norm(pts.max(0) - pts.min(0))) / 500)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[floorplan] {len(pcd.points):,} points after downsample+outlier removal", flush=True)

    extent = float(np.linalg.norm(np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0)))
    floor = find_floor(pcd, cam_centers, o3d, unit=unit, scaled=bool(scale), extent=extent)
    print(f"[floorplan] floor: {floor['inliers']:,} inliers, camera height ≈ "
          f"{floor['cam_h'] * unit:.2f} {unit_name}", flush=True)
    floor_warning = None
    if scale and floor["cam_h"] * unit < 0.6:
        floor_warning = ("ระนาบอ้างอิงอยู่ห่างกล้องแค่ "
                         f"{floor['cam_h'] * unit:.2f} m — น่าจะเป็นผิวโต๊ะ ไม่ใช่พื้นจริง "
                         "(พื้นแท้โผล่ในภาพน้อย) ขนาด/พื้นที่อาจคลาดมากกว่าปกติ")
        print(f"[floorplan] ⚠ {floor_warning}", flush=True)

    # rotate cloud: floor normal -> +Z, floor plane -> z=0
    R = rotation_to_z(floor["n"])
    P = np.asarray(pcd.points) @ R.T
    z0 = np.median(floor["pts"] @ R.T[:, 2])
    P[:, 2] -= z0
    if cam_centers is not None:
        cams2 = cam_centers @ R.T
        cams2[:, 2] -= z0

    # wall-evidence slice
    lo, hi = args.wall_band
    lo_u, hi_u = lo / unit, hi / unit
    band = P[(P[:, 2] > lo_u) & (P[:, 2] < hi_u)]
    floor_band = P[np.abs(P[:, 2]) < 0.1 / unit]
    print(f"[floorplan] wall-band points: {len(band):,} · floor points: {len(floor_band):,}", flush=True)

    # 2D occupancy grid
    cell = args.cell_m / unit
    all2d = P[:, :2]
    mn = all2d.min(0) - 2 * cell
    mx = all2d.max(0) + 2 * cell
    shape = np.ceil((mx - mn) / cell).astype(int)[::-1]  # rows=y, cols=x

    def grid_of(xy):
        idx = np.floor((xy - mn) / cell).astype(int)
        g = np.zeros(shape, dtype=np.int32)
        np.add.at(g, (idx[:, 1], idx[:, 0]), 1)
        return g

    g_wall = grid_of(band[:, :2])
    g_floor = grid_of(floor_band[:, :2])

    wall_mask = g_wall >= 3
    wall_mask = ndimage.binary_closing(wall_mask, structure=np.ones((3, 3)), iterations=2)
    floor_mask = ndimage.binary_closing(g_floor >= 2, structure=np.ones((3, 3)), iterations=2)

    # rooms — robust to fragmented walls/floor (bridge gaps, drop wall specks)
    per_cell = args.cell_m if scale else cell
    labels, rooms = segment_rooms(
        floor_mask, wall_mask, per_cell=per_cell, scaled=bool(scale),
        min_room_m2=args.min_room_m2, min_cells=400)
    print(f"[floorplan] rooms detected: {len(rooms)}", flush=True)
    for r in rooms:
        print(f"  ห้อง {r['id']}: {r['width']:.2f} x {r['length']:.2f} {unit_name}"
              f" · พื้นที่ ~{r['area']:.2f} {unit_name}²", flush=True)

    # coverage: fraction of the free-space bbox actually scanned
    coverage = float(floor_mask.sum() / max(1, (floor_mask | wall_mask).sum()))

    # ── render ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10 * shape[0] / max(1, shape[1])))
    ax.imshow(np.zeros(shape), cmap="gray", vmin=0, vmax=1, origin="lower")
    ax.imshow(np.ma.masked_where(~floor_mask, floor_mask), cmap="Greys", alpha=0.15, origin="lower")
    cmap = plt.get_cmap("tab10")
    for r in rooms:
        m = labels == r["mask_label"]
        ax.imshow(np.ma.masked_where(~m, m), cmap=matplotlib.colors.ListedColormap([cmap(r["id"] % 10)]),
                  alpha=0.45, origin="lower")
        cy, cx = ndimage.center_of_mass(m)
        ax.text(cx, cy,
                f"ห้อง {r['id']}\n{r['width']:.2f}x{r['length']:.2f} {unit_name}\n~{r['area']:.1f} {unit_name}²",
                ha="center", va="center", fontsize=11, fontfamily="Tahoma",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    wy, wx = np.nonzero(wall_mask)
    ax.scatter(wx, wy, s=0.5, c="black", marker="s")
    if cam_centers is not None:
        cidx = np.floor((cams2[:, :2] - mn) / cell).astype(int)
        ax.plot(cidx[:, 0], cidx[:, 1], "-", color="tab:green", linewidth=1.2, alpha=0.8, label="เส้นทางกล้อง")
        ax.legend(loc="lower right")
    title = f"แปลนมุมบน (estimate-grade ±5-10%) · ครอบคลุม {coverage*100:.0f}%"
    if not scale:
        title += " · ยังไม่ calibrate สเกล — หน่วยเป็นหน่วยโมเดล"
    ax.set_title(title, fontfamily="Tahoma")
    ax.set_xticks([]); ax.set_yticks([])
    for ext in ("png", "svg"):
        out = os.path.join(args.scan_dir, f"plan.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # intermediate arrays for downstream stages (R3 furniture pins, R4 report):
    # everything needed to map a world-frame 3D point onto this plan's grid.
    np.savez_compressed(
        os.path.join(args.scan_dir, "plan_data.npz"),
        wall_mask=wall_mask, floor_mask=floor_mask, room_labels=labels,
        R=R, z0=z0, mn=mn, cell=cell, shape=np.array(shape),
        room_ids=np.array([r["mask_label"] for r in rooms]),
    )

    with open(os.path.join(args.scan_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump({
            "rooms": [{k: v for k, v in r.items() if k != "mask_label"} for r in rooms],
            "unit": unit_name, "scaled": bool(scale), "coverage": round(coverage, 3),
            "grade": "estimate ±5-10%",
            "warning": floor_warning,
        }, f, ensure_ascii=False, indent=2)
    print(f"[floorplan] wrote plan.png / plan.svg / plan.json -> {args.scan_dir}", flush=True)


if __name__ == "__main__":
    main()
