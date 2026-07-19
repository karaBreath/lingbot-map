"""R4 — single-file HTML room report: 3D viewer + floor plan + tables.

Combines everything the pipeline produced into ONE self-contained report.html
(Three.js inlined, point cloud + plan images embedded as base64 — opens from
a file share or phone browser with no server and no internet).

Usage:
    python -X utf8 -m lingbot_plus.report --scan_dir scans/tum_room
    (needs merged.npz; plan/furniture artifacts are included when present)
"""

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webviewer", "vendor")


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--conf_threshold", type=float, default=3.0)
    ap.add_argument("--max_points", type=int, default=250_000)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    sd = args.scan_dir
    merged_p = os.path.join(sd, "merged.npz")
    if not os.path.exists(merged_p):
        raise SystemExit(f"not found: {merged_p} — run lingbot_plus.chunked first")

    d = np.load(merged_p)
    meta = {}
    if os.path.exists(os.path.join(sd, "meta.json")):
        with open(os.path.join(sd, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    plan_json = None
    if os.path.exists(os.path.join(sd, "plan.json")):
        with open(os.path.join(sd, "plan.json"), encoding="utf-8") as f:
            plan_json = json.load(f)
    furniture = None
    if os.path.exists(os.path.join(sd, "furniture.json")):
        with open(os.path.join(sd, "furniture.json"), encoding="utf-8") as f:
            furniture = json.load(f)

    # ── point cloud -> compact binary ────────────────────────────────────────
    pts = d["world_points"].reshape(-1, 3).astype(np.float32)
    conf = d["world_points_conf"].reshape(-1)
    cols = (d["images"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    keep = conf > args.conf_threshold
    pts, cols = pts[keep], cols[keep]
    if len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[sel], cols[sel]

    # orient like the plan: floor -> z=0, then three.js y-up (x, z, -y)
    if os.path.exists(os.path.join(sd, "plan_data.npz")):
        pl = np.load(os.path.join(sd, "plan_data.npz"))
        pts = pts @ pl["R"].T
        pts[:, 2] -= float(pl["z0"])
    pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1).astype(np.float32)

    center = pts.mean(0)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    pos_b64 = base64.b64encode(pts.tobytes()).decode()
    col_b64 = base64.b64encode(cols.tobytes()).decode()
    print(f"[report] embedding {len(pts):,} points ({len(pos_b64)//1024//1024} MB base64)", flush=True)

    scale = meta.get("scale_m_per_unit")
    unit_name = "ม." if scale else "หน่วยโมเดล"

    # ── sections ─────────────────────────────────────────────────────────────
    title = args.title or f"รายงานสแกนห้อง — {os.path.basename(os.path.abspath(sd))}"
    date_str = time.strftime("%d/%m/%Y %H:%M")

    warn_html = ""
    if plan_json and plan_json.get("warning"):
        warn_html = f'<div class="warn">⚠ {plan_json["warning"]}</div>'
    if not scale:
        warn_html += '<div class="warn">⚠ ยังไม่ calibrate สเกล — ตัวเลขเป็นหน่วยโมเดล ไม่ใช่เมตร (ใช้เครื่องมือวัดใน measure viewer)</div>'

    rooms_html = ""
    if plan_json:
        rows = "".join(
            f"<tr><td>ห้อง {r['id']}</td><td>{r['width']:.2f} × {r['length']:.2f} {unit_name}</td>"
            f"<td>~{r['area']:.1f} {unit_name}²</td></tr>"
            for r in plan_json.get("rooms", []))
        cov = plan_json.get("coverage")
        rooms_html = f"""
        <h2>ห้อง ({len(plan_json.get('rooms', []))})</h2>
        <table><tr><th>ห้อง</th><th>กว้าง × ยาว</th><th>พื้นที่</th></tr>{rows}</table>
        <p class="mut">ครอบคลุมพื้นที่สแกน ~{cov*100:.0f}% · ผนังที่กล้องไม่เห็นแสดงเป็นช่องว่าง (ไม่แต่งเติม)</p>"""

    furn_html = ""
    if furniture:
        by_room = {}
        for o in furniture:
            by_room.setdefault(o["room"], []).append(o)
        blocks = ""
        for room, items in sorted(by_room.items()):
            count = {}
            for o in items:
                count[o["type_th"]] = count.get(o["type_th"], 0) + 1
            rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in
                           sorted(count.items(), key=lambda kv: -kv[1]))
            blocks += f"<h3>{room} — {len(items)} ชิ้น</h3><table><tr><th>ชนิด</th><th>จำนวน</th></tr>{rows}</table>"
        furn_html = f"<h2>เฟอร์นิเจอร์ ({len(furniture)} ชิ้น)</h2>{blocks}"

    imgs_html = ""
    for fname, cap in (("plan.png", "แปลนมุมบน"), ("plan_furniture.png", "ตำแหน่งเฟอร์นิเจอร์")):
        p = os.path.join(sd, fname)
        if os.path.exists(p):
            imgs_html += (f'<h2>{cap}</h2><img src="data:image/png;base64,{b64_file(p)}" '
                          f'alt="{cap}" loading="lazy">')

    three_js = open(os.path.join(VENDOR, "three.min.js"), encoding="utf-8").read()
    orbit_js = open(os.path.join(VENDOR, "OrbitControls.js"), encoding="utf-8").read()

    html = f"""<!DOCTYPE html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 body{{font-family:'Segoe UI',Tahoma,sans-serif;margin:0;background:#111;color:#eee}}
 header{{padding:16px 20px;background:#1b1b1b;border-bottom:2px solid #333}}
 h1{{margin:0;font-size:1.3rem}} h2{{margin:24px 0 8px;font-size:1.1rem;color:#8ecbff}}
 h3{{margin:14px 0 6px;font-size:1rem;color:#ffd08e}}
 .mut{{color:#999;font-size:.85rem}}
 .warn{{background:#4a3200;color:#ffd08e;padding:8px 12px;border-radius:8px;margin:8px 0;font-size:.9rem}}
 main{{padding:12px 20px 60px;max-width:900px;margin:0 auto}}
 #view{{width:100%;height:70vh;background:#000;border-radius:10px;touch-action:none}}
 table{{border-collapse:collapse;width:100%;max-width:460px}}
 td,th{{border:1px solid #333;padding:6px 10px;text-align:left;font-size:.9rem}}
 th{{background:#222}} img{{max-width:100%;border-radius:10px}}
 .grade{{display:inline-block;background:#233;padding:2px 10px;border-radius:20px;font-size:.8rem;color:#9fd}}
</style></head><body>
<header><h1>{title}</h1>
<div class="mut">{date_str} · <span class="grade">estimate-grade ±5-10%</span></div></header>
<main>
{warn_html}
<h2>โมเดล 3D (ลาก=หมุน · สองนิ้ว/ล้อ=ซูม)</h2>
<div id="view"></div>
{rooms_html}
{furn_html}
{imgs_html}
<p class="mut">สร้างโดย LingBot-Map Plus · จุด {len(pts):,} จุด · ทุกตัวเลขเป็นการประเมินจากภาพ ไม่ใช่การรังวัด</p>
</main>
<script>{three_js}</script>
<script>{orbit_js}</script>
<script>
const POS_B64="{pos_b64}";
const COL_B64="{col_b64}";
function dec(b64,T){{const s=atob(b64);const a=new Uint8Array(s.length);
 for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new T(a.buffer);}}
const pos=dec(POS_B64,Float32Array), col8=dec(COL_B64,Uint8Array);
const col=new Float32Array(col8.length); for(let i=0;i<col8.length;i++)col[i]=col8[i]/255;
const el=document.getElementById('view');
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x000000);
const cam=new THREE.PerspectiveCamera(60, el.clientWidth/el.clientHeight, 0.01, 1000);
const ren=new THREE.WebGLRenderer({{antialias:true}});
ren.setSize(el.clientWidth, el.clientHeight); ren.setPixelRatio(window.devicePixelRatio);
el.appendChild(ren.domElement);
const g=new THREE.BufferGeometry();
g.setAttribute('position', new THREE.BufferAttribute(pos,3));
g.setAttribute('color', new THREE.BufferAttribute(col,3));
const mat=new THREE.PointsMaterial({{size:{extent}*0.0022, vertexColors:true}});
scene.add(new THREE.Points(g, mat));
const C=[{center[0]:.4f},{center[1]:.4f},{center[2]:.4f}], E={extent:.4f};
cam.position.set(C[0]+E*0.45, C[1]+E*0.35, C[2]+E*0.45);
const ctl=new THREE.OrbitControls(cam, ren.domElement);
ctl.target.set(C[0],C[1],C[2]); ctl.enableDamping=true; ctl.update();
window.addEventListener('resize',()=>{{cam.aspect=el.clientWidth/el.clientHeight;
 cam.updateProjectionMatrix(); ren.setSize(el.clientWidth, el.clientHeight);}});
(function loop(){{requestAnimationFrame(loop); ctl.update(); ren.render(scene,cam);}})();
</script></body></html>"""

    out = os.path.join(sd, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] wrote {out} ({os.path.getsize(out)//1024//1024} MB)", flush=True)


if __name__ == "__main__":
    main()
