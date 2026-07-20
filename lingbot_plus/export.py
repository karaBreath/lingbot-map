"""Save a finished scan's deliverables to a permanent local folder.

Replaces cloud/LINE delivery: report.html is fully self-contained (3D + plan +
tables inlined), so a copied folder on the user's machine IS the shareable
result — double-click report.html, no server, no internet. Heavy intermediates
(merged.npz, chunk_*.npz, frames/) are never copied.

Usage:
    python -X utf8 -m lingbot_plus.export --scan_dir scans/pexels-house --open
Destination: --dest, else $LBP_EXPORT_DIR, else ~/Documents/LingBot-Scans.
"""

import argparse
import os
import shutil
import sys

DELIVERABLES = [
    "report.html",              # self-contained viewer (the main deliverable)
    "mesh.glb", "mesh_textured.glb", "albedo.png",
    "plan.png", "plan.svg", "plan.json",
    "plan_furniture.png", "furniture.json",
]

README = ("ดับเบิลคลิก report.html เพื่อเปิดรายงานสแกนห้อง 3D (ไม่ต้องต่อเน็ต ไม่ต้องลงโปรแกรม)\n"
          "mesh_textured.glb = โมเดลผิว+ภาพจริง เปิดในโปรแกรม 3D อื่นได้\n"
          "plan.png = แปลนมุมบน · furniture.json = รายการเฟอร์นิเจอร์\n")


def default_dest():
    docs = os.path.expanduser(os.path.join("~", "Documents"))
    root = docs if os.path.isdir(docs) else os.path.expanduser("~")
    return os.path.join(root, "LingBot-Scans")


def export_scan(scan_dir, dest_root=None, name=None):
    """Copy deliverables from scan_dir into dest_root/<name>/. Returns (out_dir, [copied])."""
    scan_dir = os.path.abspath(scan_dir)
    if not os.path.isdir(scan_dir):
        raise SystemExit(f"not found: {scan_dir}")
    name = name or os.path.basename(scan_dir.rstrip("/\\")) or "scan"
    dest_root = dest_root or os.environ.get("LBP_EXPORT_DIR") or default_dest()
    out = os.path.join(dest_root, name)
    os.makedirs(out, exist_ok=True)

    copied = []
    for f in DELIVERABLES:
        src = os.path.join(scan_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, f))
            copied.append(f)
    with open(os.path.join(out, "เปิดรายงาน.txt"), "w", encoding="utf-8") as fh:
        fh.write(README)
    return out, copied


def reveal(path):
    """Open the folder in the OS file manager (best-effort, local machine only)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606 — local, user-invoked
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])
    except Exception as e:  # noqa: BLE001 — reveal is a convenience, never fatal
        print(f"[export] could not open folder: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--dest", default=None, help="destination root (default $LBP_EXPORT_DIR or ~/Documents/LingBot-Scans)")
    ap.add_argument("--name", default=None, help="folder name (default = scan folder name)")
    ap.add_argument("--open", action="store_true", help="reveal the folder afterwards")
    args = ap.parse_args()
    out, copied = export_scan(args.scan_dir, args.dest, args.name)
    if "report.html" not in copied:
        print("[export] ⚠ report.html not found — scan may be incomplete", flush=True)
    print(f"[export] {len(copied)} files -> {out}", flush=True)
    if args.open:
        reveal(out)


if __name__ == "__main__":
    main()
