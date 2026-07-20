"""Track M (M1) — point cloud -> continuous surface mesh (Poisson) -> mesh.glb.

CPU-only (Open3D + trimesh), safe to run while the GPU is busy. Quality note:
LingBot-Map produces SLAM-grade point clouds, so the mesh is "early Polycam"
grade — good continuous surfaces where the camera covered well, holes where it
did not. We never invent geometry (no hole filling beyond Poisson smoothing).

Usage:
    python -X utf8 -m lingbot_plus.meshing --scan_dir scans/tum_room
Reads  merged.npz  (from lingbot_plus.chunked)
Writes mesh.glb (shareable) + mesh_data.npz (for report.html embedding)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_plus.chunked import extr_to_centers_rots  # noqa: E402


def mesh_from_points(pts, cols, cam_centers, frame_idx,
                     out_glb=None, depth=9, density_quantile=0.06):
    """Poisson-reconstruct a colored mesh from a colored point cloud.

    pts (N,3) float; cols (N,3) float 0..1; cam_centers (F,3) camera positions;
    frame_idx (N,) which camera saw each point (normals get oriented toward it).
    Returns (vertices f32, faces u32, vertex_colors u8).
    """
    import open3d as o3d

    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    voxel = extent / 400.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1).astype(np.float64))
    # keep camera assignment through downsampling via nearest original point
    down = pcd.voxel_down_sample(voxel)
    down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=40))
    # orient each normal toward the camera that observed it (nearest-original lookup)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dpts = np.asarray(down.points)
    dnorm = np.asarray(down.normals)
    for i in range(len(dpts)):
        _, idx, _ = tree.search_knn_vector_3d(dpts[i], 1)
        cam = cam_centers[frame_idx[idx[0]]]
        if np.dot(dnorm[i], cam - dpts[i]) < 0:
            dnorm[i] = -dnorm[i]
    down.normals = o3d.utility.Vector3dVector(dnorm)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(down, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))

    # crop the Poisson balloon back to the observed volume (+3%)
    lo, hi = pts.min(0), pts.max(0)
    pad = (hi - lo) * 0.03
    bbox = o3d.geometry.AxisAlignedBoundingBox(lo - pad, hi + pad)
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.uint32)
    vcols = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)

    if out_glb:
        import trimesh
        tm = trimesh.Trimesh(vertices=verts, faces=faces,
                             vertex_colors=vcols, process=False)
        tm.export(out_glb)

    return verts, faces, vcols


def texture_mesh(verts, faces, images, intrinsics, extrinsics,
                 tex_size=1024, img_stride=4):
    """M2 — bake a UV texture atlas from the posed keyframes.

    xatlas unwraps the mesh into UV charts (robust to the open boundaries of a
    cropped Poisson mesh, which Open3D's own compute_uvatlas rejects), then
    Open3D project_images_to_albedo back-projects the images onto the atlas and
    blends overlaps — image-resolution surface colour instead of per-vertex.

    images (N,H,W,3) uint8 · intrinsics (N,3,3) · extrinsics (N,3,4) w2c.
    Returns (vertices f32, faces u32, uv f32 per-vertex, albedo (T,T,3) u8).
    """
    import xatlas

    vmap, idx, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    NV = verts[vmap].astype(np.float32)

    import open3d as o3d
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex["positions"] = o3d.core.Tensor(NV, o3d.core.float32)
    tm.triangle["indices"] = o3d.core.Tensor(idx.astype(np.int64), o3d.core.int64)
    tm.triangle["texture_uvs"] = o3d.core.Tensor(uvs[idx].astype(np.float32), o3d.core.float32)

    sel = range(0, len(images), img_stride)
    imgs = [o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(images[i]))) for i in sel]
    Ks = [o3d.core.Tensor(intrinsics[i].astype(np.float64)) for i in sel]
    Es = [o3d.core.Tensor(np.vstack([extrinsics[i], [0, 0, 0, 1]]).astype(np.float64)) for i in sel]
    alb = tm.project_images_to_albedo(imgs, Ks, Es, tex_size=tex_size, update_material=True)
    albedo = alb.as_tensor().numpy().astype(np.uint8)

    # fill small uncovered texels (occlusion/blend gaps show as black speckle)
    import cv2
    gap = (albedo.sum(2) == 0).astype(np.uint8)
    if 0 < gap.mean() < 0.9:
        albedo = cv2.inpaint(np.ascontiguousarray(albedo), gap, 3, cv2.INPAINT_TELEA)
    return NV, idx.astype(np.uint32), uvs.astype(np.float32), albedo


def decimate(verts, faces, vcols, target_tris):
    """Reduce triangle count for lightweight embedding (full-res GLB kept separately)."""
    import open3d as o3d
    if len(faces) <= target_tris:
        return verts, faces, vcols
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)))
    m.vertex_colors = o3d.utility.Vector3dVector(vcols.astype(np.float64) / 255.0)
    m = m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    return (np.asarray(m.vertices).astype(np.float32),
            np.asarray(m.triangles).astype(np.uint32),
            (np.asarray(m.vertex_colors) * 255).clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--conf_threshold", type=float, default=3.0)
    ap.add_argument("--max_points", type=int, default=800_000)
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth (9=default, 10=finer)")
    ap.add_argument("--embed_tris", type=int, default=180_000,
                    help="triangle budget for report.html embed (GLB stays full-res)")
    ap.add_argument("--texture", action="store_true",
                    help="M2: bake a UV texture atlas from keyframes (image-res surface colour)")
    ap.add_argument("--tex_size", type=int, default=1024)
    ap.add_argument("--tex_img_stride", type=int, default=4, help="use every Nth keyframe when baking")
    args = ap.parse_args()

    sd = args.scan_dir
    merged_p = os.path.join(sd, "merged.npz")
    if not os.path.exists(merged_p):
        raise SystemExit(f"not found: {merged_p} — run lingbot_plus.chunked first")

    d = np.load(merged_p)
    F, H, W = d["world_points"].shape[:3]
    pts = d["world_points"].reshape(-1, 3).astype(np.float32)
    conf = d["world_points_conf"].reshape(-1)
    cols = d["images"].transpose(0, 2, 3, 1).reshape(-1, 3)
    frame_idx = np.repeat(np.arange(F, dtype=np.int32), H * W)
    cam_centers, _ = extr_to_centers_rots(d["extrinsic"])

    keep = conf > args.conf_threshold
    pts, cols, frame_idx = pts[keep], cols[keep], frame_idx[keep]
    if len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols, frame_idx = pts[sel], cols[sel], frame_idx[sel]
    print(f"[mesh] {len(pts):,} points from {F} frames -> Poisson depth={args.depth}", flush=True)

    out_glb = os.path.join(sd, "mesh.glb")
    verts, faces, vcols = mesh_from_points(
        pts, cols, cam_centers, frame_idx, out_glb=out_glb, depth=args.depth)
    ev, ef, ec = decimate(verts, faces, vcols, args.embed_tris)
    np.savez_compressed(os.path.join(sd, "mesh_data.npz"),
                        vertices=ev, faces=ef, colors=ec)
    print(f"[mesh] wrote {out_glb} ({os.path.getsize(out_glb)//1024//1024} MB, "
          f"{len(verts):,} verts, {len(faces):,} tris; embed {len(ef):,} tris)", flush=True)

    if args.texture:
        import cv2
        images = (d["images"].transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
        print(f"[mesh] texturing on the {len(ef):,}-tri mesh from "
              f"{len(range(0, F, args.tex_img_stride))} keyframes...", flush=True)
        tv, tf, tuv, albedo = texture_mesh(
            ev, ef, images, d["intrinsic"], d["extrinsic"],
            tex_size=args.tex_size, img_stride=args.tex_img_stride)
        cv2.imwrite(os.path.join(sd, "albedo.png"),
                    cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
        np.savez_compressed(os.path.join(sd, "mesh_tex.npz"),
                            vertices=tv, faces=tf, uv=tuv)
        # textured GLB (image baked in) for external viewers
        import trimesh
        from PIL import Image
        vis = trimesh.visual.TextureVisuals(uv=tuv, image=Image.fromarray(albedo))
        trimesh.Trimesh(vertices=tv, faces=tf, visual=vis, process=False).export(
            os.path.join(sd, "mesh_textured.glb"))
        cover = 100 * (albedo.reshape(-1, 3).sum(1) > 0).mean()
        print(f"[mesh] wrote mesh_textured.glb + albedo.png ({args.tex_size}px, "
              f"{cover:.0f}% covered)", flush=True)


if __name__ == "__main__":
    main()
