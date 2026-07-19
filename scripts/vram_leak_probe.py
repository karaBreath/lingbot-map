import os
import subprocess
import sys

import torch
import torch_directml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingbot_map.models.gct_stream import GCTStream

pid = os.getpid()


def gpu_mb():
    ps = (
        f"(Get-Counter '\\GPU Process Memory(*)\\Local Usage' -ErrorAction SilentlyContinue)."
        f"CounterSamples | Where-Object {{$_.Path -match '_{pid}_'}} | "
        f"Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    val = out.stdout.strip()
    try:
        return float(val) / 1024 / 1024
    except ValueError:
        return -1.0


device = torch_directml.device()
print(f"PID={pid}  GPU mem at start: {gpu_mb():.1f} MB", flush=True)

model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                   kv_cache_sliding_window=64, kv_cache_scale_frames=2,
                   kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                   use_sdpa=True, camera_num_iterations=4)
ckpt = torch.load("weights/lingbot-map.pt", map_location="cpu", weights_only=False)
sd = ckpt.get("model", ckpt)
model.load_state_dict(sd, strict=False)
print(f"GPU mem after load_state_dict (still CPU): {gpu_mb():.1f} MB", flush=True)
model = model.to(device).eval()
print(f"GPU mem after model.to(device): {gpu_mb():.1f} MB", flush=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
imgs = torch.rand(N, 3, 518, 518, dtype=torch.float32, device=device)
print(f"GPU mem after allocating {N}-frame input tensor: {gpu_mb():.1f} MB", flush=True)

with torch.no_grad():
    out = model.inference_streaming(
        imgs, num_scale_frames=2, keyframe_interval=1,
        output_device=torch.device("cpu"),
    )
print(f"GPU mem after inference_streaming({N} frames, output_device=cpu): {gpu_mb():.1f} MB", flush=True)

print("DONE - no crash", flush=True)
