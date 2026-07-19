"""R3 — furniture detection with 3D positions and room assignment.

LingBot-Map already gives per-pixel world coordinates (world_points) for every
frame, so furniture localization is just: detect on 2D keyframes (YOLO) ->
read the 3D position straight out of world_points under the box -> cluster
detections of the same physical object across frames -> pin onto the R2 plan.

Outputs into the scan folder:
    furniture.json        — [{type, type_th, position, room, confidence, sightings}]
    plan_furniture.png    — R2 plan with furniture pins

Usage:
    python -X utf8 -m lingbot_plus.furniture --scan_dir scans/tum_room
    (needs merged.npz from chunked.py and plan_data.npz from floorplan.py)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# COCO classes that count as furniture/appliances for a room report, with Thai names.
FURNITURE = {
    "chair": "เก้าอี้", "couch": "โซฟา", "bed": "เตียง", "dining table": "โต๊ะ",
    "tv": "ทีวี/จอ", "laptop": "โน้ตบุ๊ก", "keyboard": "คีย์บอร์ด",
    "refrigerator": "ตู้เย็น", "microwave": "ไมโครเวฟ", "oven": "เตาอบ",
    "sink": "ซิงก์", "toilet": "สุขภัณฑ์", "potted plant": "ต้นไม้",
    "clock": "นาฬิกา", "vase": "แจกัน", "book": "หนังสือ",
}
MERGE_RADIUS_M = 0.7   # detections of the same class within this radius = same object
MIN_SIGHTINGS = 2      # object must be seen in >=2 keyframes to count (kills flukes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--frame_stride", type=int, default=5, help="detect on every Nth frame")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    merged_p = os.path.join(args.scan_dir, "merged.npz")
    plan_p = os.path.join(args.scan_dir, "plan_data.npz")
    for p, need in ((merged_p, "lingbot_plus.chunked"), (plan_p, "lingbot_plus.floorplan")):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p} — run {need} first")

    d = np.load(merged_p)
    plan = np.load(plan_p)
    scale = None
    meta_p = os.path.join(args.scan_dir, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            scale = json.load(f).get("scale_m_per_unit")
    unit = scale or 1.0
    unit_name = "m" if scale else "หน่วยโมเดล"

    images = d["images"]                 # (S,3,H,W) float 0..1
    wpts = d["world_points"]             # (S,H,W,3)
    wconf = d["world_points_conf"]       # (S,H,W)
    S, _, H, W = images.shape
    frames = list(range(0, S, args.frame_stride))
    print(f"[furniture] {S} frames -> detecting on {len(frames)} keyframes (stride {args.frame_stride})", flush=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    detections = []                      # {cls, conf, pos(3), frame}
    for fi in frames:
        img = (images[fi].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        res = model.predict(img, conf=args.conf, verbose=False)[0]
        for box in res.boxes:
            cls_name = model.names[int(box.cls)]
            if cls_name not in FURNITURE:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            # core third of the bbox: robust against background bleeding at edges
            cx1, cx2 = x1 + (x2 - x1) // 3, x2 - (x2 - x1) // 3
            cy1, cy2 = y1 + (y2 - y1) // 3, y2 - (y2 - y1) // 3
            core_w = wpts[fi, cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)].reshape(-1, 3)
            core_c = wconf[fi, cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)].reshape(-1)
            good = core_c > 2.0
            if good.sum() < 10:
                continue
            pos = np.median(core_w[good], axis=0)
            detections.append({"cls": cls_name, "conf": float(box.conf), "pos": pos, "frame": fi})
    print(f"[furniture] raw detections: {len(detections)}", flush=True)

    # ── cluster per class (greedy, highest confidence first) ─────────────────
    objects = []
    for cls_name in sorted({dt["cls"] for dt in detections}):
        ds = sorted([dt for dt in detections if dt["cls"] == cls_name],
                    key=lambda x: -x["conf"])
        used = [False] * len(ds)
        for i, di in enumerate(ds):
            if used[i]:
                continue
            group = [di]
            used[i] = True
            for j in range(i + 1, len(ds)):
                if used[j]:
                    continue
                if np.linalg.norm(ds[j]["pos"] - di["pos"]) * unit < MERGE_RADIUS_M:
                    group.append(ds[j])
                    used[j] = True
            if len(group) < MIN_SIGHTINGS:
                continue
            objects.append({
                "type": cls_name,
                "type_th": FURNITURE[cls_name],
                "position": np.median([g["pos"] for g in group], axis=0),
                "confidence": round(float(np.mean([g["conf"] for g in group])), 3),
                "sightings": len(group),
            })

    # ── room assignment via the R2 plan grid ─────────────────────────────────
    R, z0 = plan["R"], float(plan["z0"])
    mn, cell = plan["mn"], float(plan["cell"])
    room_labels = plan["room_labels"]
    room_of_label = {int(lab): idx + 1 for idx, lab in enumerate(plan["room_ids"])}
    for o in objects:
        p2 = o["position"] @ R.T
        gx, gy = int((p2[0] - mn[0]) / cell), int((p2[1] - mn[1]) / cell)
        room = None
        if 0 <= gy < room_labels.shape[0] and 0 <= gx < room_labels.shape[1]:
            lab = int(room_labels[gy, gx])
            room = room_of_label.get(lab)
        o["room"] = f"ห้อง {room}" if room else "นอกพื้นที่ห้องที่ระบุได้"
        o["grid"] = (gx, gy)

    objects.sort(key=lambda o: (o["room"], -o["sightings"]))
    print(f"[furniture] objects after clustering: {len(objects)}", flush=True)
    for o in objects:
        print(f"  {o['type_th']} ({o['type']}) · {o['room']} · เห็น {o['sightings']} เฟรม"
              f" · conf {o['confidence']}", flush=True)

    with open(os.path.join(args.scan_dir, "furniture.json"), "w", encoding="utf-8") as f:
        json.dump([{**o, "position": [round(float(x), 4) for x in o["position"]],
                    "grid": list(o["grid"])} for o in objects],
                  f, ensure_ascii=False, indent=2)

    # ── render pins on the plan ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Tahoma"
    import matplotlib.pyplot as plt
    from scipy import ndimage

    wall_mask = plan["wall_mask"]
    floor_mask = plan["floor_mask"]
    shape = tuple(plan["shape"])
    fig, ax = plt.subplots(figsize=(10, 10 * shape[0] / max(1, shape[1])))
    ax.imshow(np.zeros(shape), cmap="gray", vmin=0, vmax=1, origin="lower")
    ax.imshow(np.ma.masked_where(~floor_mask, floor_mask), cmap="Oranges",
              alpha=0.35, origin="lower")
    wy, wx = np.nonzero(wall_mask)
    ax.scatter(wx, wy, s=0.5, c="black", marker="s")
    cmap = plt.get_cmap("tab10")
    seen_types = sorted({o["type"] for o in objects})
    type_color = {t: cmap(i % 10) for i, t in enumerate(seen_types)}
    for o in objects:
        gx, gy = o["grid"]
        ax.scatter([gx], [gy], s=120, color=type_color[o["type"]],
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(o["type_th"], (gx, gy), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color="white",
                    bbox=dict(facecolor=type_color[o["type"]], alpha=0.85, edgecolor="none"))
    ax.set_title(f"เฟอร์นิเจอร์ {len(objects)} ชิ้น (estimate-grade)")
    ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(args.scan_dir, "plan_furniture.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[furniture] wrote furniture.json + plan_furniture.png -> {args.scan_dir}", flush=True)


if __name__ == "__main__":
    main()
