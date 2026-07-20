"""Single-chunk inference worker.

Runs LingBot-Map streaming inference on a slice of an image folder and writes
the predictions to an .npz. Meant to be launched as a fresh subprocess per
chunk by chunked.py — a new process means a fresh DirectML heap, which is the
whole point (the allocator has no cache/defrag, so long runs die in one
process but chains of short runs are stable).

Usage:
    python -m lingbot_plus.worker --model_path W.pt --image_folder DIR \
        --start 0 --end 120 --stride 3 --out chunk_000.npz [--backend auto]
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_plus.device import resolve_backend
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--start", type=int, required=True, help="index into sorted frame list (after stride)")
    ap.add_argument("--end", type=int, required=True, help="exclusive")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--num_scale_frames", type=int, default=2)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=16)
    ap.add_argument("--image_ext", default=".jpg,.png,.JPG")
    args = ap.parse_args()

    backend = resolve_backend(args.backend)
    print(f"[worker] {backend.describe()}", flush=True)

    paths = []
    for ext in args.image_ext.split(","):
        paths.extend(glob.glob(os.path.join(args.image_folder, f"*{ext}")))
    paths = sorted(paths)[:: args.stride]
    chunk_paths = paths[args.start : args.end]
    if not chunk_paths:
        raise SystemExit(f"[worker] empty chunk {args.start}:{args.end} (have {len(paths)} strided frames)")
    print(f"[worker] frames {args.start}..{args.end} of {len(paths)} (chunk={len(chunk_paths)})", flush=True)

    images = load_and_preprocess_images(chunk_paths, mode="crop", image_size=args.image_size, patch_size=14)

    from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(
        img_size=args.image_size, patch_size=14, enable_3d_rope=True,
        max_frame_num=1024, kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        use_sdpa=backend.force_sdpa, camera_num_iterations=4,
    )
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = model.to(backend.device).eval()

    # images stay on CPU: inference_streaming moves each frame itself.
    with torch.no_grad(), backend.autocast():
        pred = model.inference_streaming(
            images, num_scale_frames=args.num_scale_frames, keyframe_interval=1,
            output_device=torch.device("cpu"),
        )

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    e44 = torch.zeros((*extrinsic.shape[:-2], 4, 4), dtype=extrinsic.dtype)
    e44[..., :3, :4] = extrinsic.cpu()
    e44[..., 3, 3] = 1.0
    e44 = closed_form_inverse_se3_general(e44)  # w2c -> c2w

    def _np(t):
        t = t.cpu() if isinstance(t, torch.Tensor) else torch.as_tensor(t)
        if t.ndim > 0 and t.shape[0] == 1:
            t = t[0]
        return t.numpy()

    np.savez_compressed(
        args.out,
        depth=_np(pred["depth"]).astype(np.float32),
        depth_conf=_np(pred["depth_conf"]).astype(np.float32),
        extrinsic=_np(e44[..., :3, :4]).astype(np.float32),
        intrinsic=_np(intrinsic).astype(np.float32),
        images=(_np(images.unsqueeze(0)) * 255).clip(0, 255).astype(np.uint8),
        start=args.start, end=args.end,
    )
    print(f"[worker] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
