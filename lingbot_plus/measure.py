"""R1 — measurement + metric-scale calibration on top of PointCloudViewer.

Monocular reconstructions have consistent shape but unknown units. The user
measures ONE real thing (e.g. a door edge) with a tape measure, clicks its two
endpoints in the viewer, enters the real length, and every later measurement is
reported in meters. The scale factor is persisted to <scan_dir>/meta.json.

Usage:
    python -m lingbot_plus.measure --scan_dir scans/tum_room --port 8099
    (expects merged.npz from lingbot_plus.chunked in --scan_dir)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.vis.point_cloud_viewer import PointCloudViewer


class MeasureViewer(PointCloudViewer):
    def __init__(self, *args, scan_dir=None, **kwargs):
        self.scan_dir = scan_dir
        self.scale_m_per_unit = None
        self._measure_pts = []          # up to 2 clicked 3D points
        self._last_model_dist = None
        self._marker_handles = []
        self._all_pts_cache = None
        self._load_meta()
        # super().__init__ calls _setup_gui(), which we override to put the
        # measure panel at the TOP of the sidebar (it's the headline feature —
        # burying it under the tuning sliders makes it unreachable).
        super().__init__(*args, **kwargs)

    def _setup_gui(self):
        self._setup_measure_gui()
        super()._setup_gui()

    # ── meta.json persistence ────────────────────────────────────────────────
    def _meta_path(self):
        return os.path.join(self.scan_dir, "meta.json") if self.scan_dir else None

    def _load_meta(self):
        p = self._meta_path()
        if p and os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    self.scale_m_per_unit = json.load(f).get("scale_m_per_unit")
            except (json.JSONDecodeError, OSError):
                pass

    def _save_meta(self, model_dist, real_m):
        p = self._meta_path()
        if not p:
            return
        meta = {}
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                meta = {}
        meta.update({
            "scale_m_per_unit": self.scale_m_per_unit,
            "calib_model_dist": float(model_dist),
            "calib_real_m": float(real_m),
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ── geometry ─────────────────────────────────────────────────────────────
    def _all_points(self):
        if self._all_pts_cache is None:
            if not self.vis_pts_list:
                return None
            pts = [p for p in self.vis_pts_list if len(p) > 0]
            if not pts:
                return None
            self._all_pts_cache = np.concatenate(pts, axis=0).astype(np.float32)
        return self._all_pts_cache

    def _pick_point(self, origin, direction):
        """First surface the click ray hits, or None if the click missed.

        Global nearest-to-ray is wrong here: one dense/outlier point close to
        many rays swallows every click (two distinct clicks returned the same
        3D point -> 0.000 m). Instead: take points within a thin cone around
        the ray, then the closest one ALONG the ray (first hit), refined to the
        median of its local neighborhood for noise robustness.
        """
        pts = self._all_points()
        if pts is None:
            return None
        o = np.asarray(origin, dtype=np.float32)
        u = np.asarray(direction, dtype=np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = pts - o
        t = v @ u                       # distance along ray
        perp = np.linalg.norm(v - np.outer(t, u), axis=1)
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        MIN_CLUSTER = 12                # a real surface has many points; floaters don't
        band = 0.015 * extent
        for frac in (0.004, 0.008, 0.015):
            cand = (t > 0) & (perp < frac * extent)
            if cand.sum() < MIN_CLUSTER:
                continue
            t_c = t[cand]
            idx_c = np.flatnonzero(cand)
            order = np.argsort(t_c)
            t_sorted = t_c[order]
            # first DENSE cluster along the ray: sliding window over sorted t —
            # skips isolated near-camera noise specks that a plain first-hit
            # (or global nearest-to-ray) would latch onto.
            n = len(t_sorted)
            i_end = np.searchsorted(t_sorted, t_sorted + band, side="right")
            counts = i_end - np.arange(n)
            dense = np.flatnonzero(counts >= MIN_CLUSTER)
            if len(dense) == 0:
                continue
            i0 = int(dense[0])
            sel = idx_c[order[i0 : i_end[i0]]]
            return pts[sel].mean(axis=0)
        return None

    def _scene_extent(self):
        pts = self._all_points()
        return float(np.linalg.norm(pts.max(0) - pts.min(0))) if pts is not None else 1.0

    # ── GUI ──────────────────────────────────────────────────────────────────
    def _fmt_dist(self, d):
        if self.scale_m_per_unit:
            return f"{d * self.scale_m_per_unit:.3f} m"
        return f"{d:.4f} หน่วยโมเดล (ยังไม่ตั้งสเกล)"

    def _setup_measure_gui(self):
        with self.server.gui.add_folder("วัดระยะ (Measure)"):
            self._status = self.server.gui.add_markdown("คลิกปุ่มแล้วคลิกจุด 2 จุดบน point cloud")
            btn_measure = self.server.gui.add_button("เริ่มวัด (2 คลิก)")
            btn_clear = self.server.gui.add_button("ล้างหมุด")
            self._real_len = self.server.gui.add_number(
                "ความยาวจริง (เมตร)", initial_value=1.00, min=0.01, max=100.0, step=0.01)
            btn_calib = self.server.gui.add_button("ตั้งสเกลจากการวัดล่าสุด")
            scale_txt = (f"สเกลปัจจุบัน: {self.scale_m_per_unit:.4f} m/หน่วย"
                         if self.scale_m_per_unit else "สเกลปัจจุบัน: ยังไม่ตั้ง")
            self._scale_status = self.server.gui.add_markdown(scale_txt)

        @btn_measure.on_click
        def _(_):
            self._measure_pts = []
            self._clear_markers()
            self._status.content = "คลิกจุดที่ 1 บน point cloud..."
            self._arm_click()

        @btn_clear.on_click
        def _(_):
            self._measure_pts = []
            self._clear_markers()
            self._status.content = "ล้างแล้ว — กด 'เริ่มวัด' เพื่อวัดใหม่"

        @btn_calib.on_click
        def _(_):
            if self._last_model_dist is None or self._last_model_dist <= 0:
                self._status.content = "ยังไม่มีการวัด — วัดระยะบนของที่รู้ความยาวจริงก่อน"
                return
            real = float(self._real_len.value)
            self.scale_m_per_unit = real / self._last_model_dist
            self._save_meta(self._last_model_dist, real)
            self._scale_status.content = f"สเกลปัจจุบัน: {self.scale_m_per_unit:.4f} m/หน่วย (บันทึกแล้ว)"
            self._status.content = (
                f"ตั้งสเกลแล้ว: {self._last_model_dist:.4f} หน่วย = {real:.2f} m — "
                "การวัดต่อไปแสดงเป็นเมตร")

    def _arm_click(self):
        @self.server.scene.on_click()
        def _(event):
            p = self._pick_point(event.ray_origin, event.ray_direction)
            if p is None:
                self._status.content = "คลิกไม่โดนจุด — ลองคลิกบริเวณที่มีจุดหนาแน่น"
                return
            self._measure_pts.append(p)
            r = self._scene_extent() / 250.0
            color = (255, 60, 60) if len(self._measure_pts) == 1 else (60, 220, 60)
            self._marker_handles.append(self.server.scene.add_icosphere(
                f"/measure/pt{len(self._measure_pts)}", radius=r, color=color,
                position=tuple(float(x) for x in p)))
            if len(self._measure_pts) == 1:
                self._status.content = "ได้จุดที่ 1 — คลิกจุดที่ 2..."
            else:
                a, b = self._measure_pts
                d = float(np.linalg.norm(a - b))
                self._last_model_dist = d
                seg = np.array([[a, b]], dtype=np.float32)
                self._marker_handles.append(self.server.scene.add_line_segments(
                    "/measure/line", points=seg,
                    colors=np.array([[[255, 220, 0], [255, 220, 0]]], dtype=np.uint8),
                    line_width=3.0))
                mid = tuple(float(x) for x in (a + b) / 2)
                self._marker_handles.append(self.server.scene.add_label(
                    "/measure/dist", text=self._fmt_dist(d), position=mid))
                self._status.content = f"ระยะ = {self._fmt_dist(d)}"
                self._measure_pts = []

    def _clear_markers(self):
        for h in self._marker_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._marker_handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True, help="folder containing merged.npz (from lingbot_plus.chunked)")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--conf_threshold", type=float, default=1.5)
    ap.add_argument("--downsample_factor", type=int, default=10)
    args = ap.parse_args()

    npz = os.path.join(args.scan_dir, "merged.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"not found: {npz} — run lingbot_plus.chunked first")
    pred = dict(np.load(npz))
    print(f"[measure] loaded {npz}: {pred['world_points'].shape[0]} frames", flush=True)

    viewer = MeasureViewer(
        pred_dict=pred, port=args.port, vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor, point_size=0.00001,
        use_point_map=True, scan_dir=args.scan_dir,
    )
    print(f"[measure] viewer at http://localhost:{args.port}", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
