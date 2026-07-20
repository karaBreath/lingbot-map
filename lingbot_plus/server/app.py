"""Phase 1 GUI — web upload for room scans (grows into the Phase-4 LINE server).

One FastAPI app, one worker thread (the GPU handles one job at a time anyway):
    upload video -> extract frames (auto 16:9 center-crop) -> chunked
    reconstruction -> floor plan -> furniture -> report.html

Run:
    .venv-amd\\Scripts\\python -X utf8 -m uvicorn lingbot_plus.server.app:app --port 8500
Config via .env (see .env.example) — no hardcoded paths/ports.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_env():
    p = os.path.join(REPO, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
MODEL_PATH = os.path.join(REPO, os.environ.get("LBP_MODEL_PATH", "weights/lingbot-map.pt"))
SCANS_DIR = os.path.join(REPO, os.environ.get("LBP_SCANS_DIR", "scans"))
UPLOAD_DIR = os.path.join(REPO, os.environ.get("LBP_UPLOAD_DIR", "uploads"))
BACKEND = os.environ.get("LBP_DEFAULT_BACKEND", "auto")
os.makedirs(SCANS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

PRESETS = {
    # fps: frames sampled per second of video; chunked stride always 1 (fps does the thinning)
    "fast":     {"fps": 3, "chunk_size": 100, "overlap": 8,  "label": "เร็ว"},
    "balanced": {"fps": 5, "chunk_size": 120, "overlap": 10, "label": "สมดุล"},
    "quality":  {"fps": 8, "chunk_size": 120, "overlap": 12, "label": "ละเอียด"},
}
STAGES = ["extract", "reconstruct", "mesh", "floorplan", "furniture", "report"]
STAGE_TH = {"extract": "แตกเฟรมวิดีโอ", "reconstruct": "สร้างโมเดล 3D",
            "mesh": "ขึ้นผิวโมเดล", "floorplan": "ทำแปลนห้อง", "furniture": "หาเฟอร์นิเจอร์",
            "report": "รวมรายงาน", "done": "เสร็จ", "failed": "ล้มเหลว", "queued": "รอคิว"}
OUTPUTS = ("points", "mesh", "both")   # Track M — 3D result mode

app = FastAPI(title="LingBot-Map Plus")
app.mount("/scans", StaticFiles(directory=SCANS_DIR, html=True), name="scans")

_jobs: dict = {}
_q: "queue.Queue[str]" = queue.Queue()


def _extract_frames(video_path, out_dir, fps):
    """Video -> numbered 16:9 frames. Center-crops any aspect to 16:9
    (model geometry proven at 518x294; taller frames OOM DirectML)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("เปิดไฟล์วิดีโอไม่ได้ — รองรับ mp4/mov/avi ที่ OpenCV อ่านได้")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, round(src_fps / fps))
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            h, w = frame.shape[:2]
            target_h = int(w * 9 / 16)
            if target_h < h:                     # too tall (4:3 / vertical) -> crop rows
                top = (h - target_h) // 2
                frame = frame[top:top + target_h]
            elif target_h > h:                   # ultra-wide -> crop columns
                target_w = int(h * 16 / 9)
                left = (w - target_w) // 2
                frame = frame[:, left:left + target_w]
            cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    if saved < 12:
        raise RuntimeError(f"ได้เฟรมแค่ {saved} — วิดีโอสั้นเกินไปหรืออ่านไม่ได้")
    return saved


def _run_stage(job, args_list):
    """Run a pipeline module as a subprocess, teeing output into the job log."""
    cmd = [sys.executable, "-X", "utf8", "-m"] + args_list
    job["log"].append("$ " + " ".join(args_list))
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            job["log"] = job["log"][-60:] + [line]
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{args_list[0]} exit {proc.returncode}")


def _worker():
    while True:
        name = _q.get()
        job = _jobs[name]
        scan_dir = os.path.join(SCANS_DIR, name)
        frames_dir = os.path.join(scan_dir, "frames")
        preset = PRESETS[job["preset"]]
        try:
            job["status"] = "extract"
            n = _extract_frames(job["video"], frames_dir, preset["fps"])
            job["log"].append(f"ได้ {n} เฟรม (fps={preset['fps']})")

            job["status"] = "reconstruct"
            _run_stage(job, ["lingbot_plus.chunked", "--model_path", MODEL_PATH,
                             "--image_folder", frames_dir, "--stride", "1",
                             "--chunk_size", str(preset["chunk_size"]),
                             "--overlap", str(preset["overlap"]),
                             "--out_dir", scan_dir, "--backend", job["backend"]])

            if job.get("output", "points") in ("mesh", "both"):
                job["status"] = "mesh"
                # mesh mode is the "pretty" output -> also bake the UV texture (M2)
                _run_stage(job, ["lingbot_plus.meshing", "--scan_dir", scan_dir, "--texture"])

            job["status"] = "floorplan"
            _run_stage(job, ["lingbot_plus.floorplan", "--scan_dir", scan_dir])

            job["status"] = "furniture"
            _run_stage(job, ["lingbot_plus.furniture", "--scan_dir", scan_dir])

            job["status"] = "report"
            _run_stage(job, ["lingbot_plus.report", "--scan_dir", scan_dir])

            job["status"] = "done"
            job["report"] = f"/scans/{name}/report.html"
        except Exception as e:  # noqa: BLE001 — surface any stage failure to the UI
            job["status"] = "failed"
            job["error"] = str(e)
        finally:
            job["finished_at"] = time.time()
            with open(os.path.join(scan_dir, "job.json"), "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in job.items() if k != "log"} |
                          {"log_tail": job["log"][-20:]}, f, ensure_ascii=False, indent=2)


threading.Thread(target=_worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/api/scan")
async def create_scan(video: UploadFile = File(...), name: str = Form(""),
                      preset: str = Form("balanced"), backend: str = Form(""),
                      output: str = Form("points")):
    if preset not in PRESETS:
        raise HTTPException(400, f"preset ต้องเป็นหนึ่งใน {list(PRESETS)}")
    if output not in OUTPUTS:
        raise HTTPException(400, f"output ต้องเป็นหนึ่งใน {list(OUTPUTS)}")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", name) or time.strftime("scan-%Y%m%d-%H%M%S")
    if safe in _jobs and _jobs[safe]["status"] not in ("done", "failed"):
        raise HTTPException(409, f"งานชื่อ '{safe}' กำลังทำอยู่")
    scan_dir = os.path.join(SCANS_DIR, safe)
    os.makedirs(scan_dir, exist_ok=True)
    vid_path = os.path.join(UPLOAD_DIR, f"{safe}{os.path.splitext(video.filename or '.mp4')[1]}")
    with open(vid_path, "wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    _jobs[safe] = {"name": safe, "status": "queued", "preset": preset,
                   "backend": backend or BACKEND, "output": output, "video": vid_path,
                   "created_at": time.time(), "log": [], "error": None, "report": None}
    _q.put(safe)
    return {"name": safe, "status": "queued"}


@app.get("/api/jobs")
def jobs():
    out = []
    for j in sorted(_jobs.values(), key=lambda x: -x["created_at"]):
        out.append({"name": j["name"], "status": j["status"],
                    "status_th": STAGE_TH.get(j["status"], j["status"]),
                    "stage_index": STAGES.index(j["status"]) if j["status"] in STAGES else
                                   (len(STAGES) if j["status"] == "done" else -1),
                    "stages_total": len(STAGES),
                    "preset": j["preset"], "backend": j["backend"],
                    "report": j["report"], "error": j["error"],
                    "last_log": j["log"][-1] if j["log"] else ""})
    return JSONResponse(out)
