#!/usr/bin/env python3
"""
generate_dense_pointcloud.py — Dense Point Cloud Generation via Depth Estimation

Generates a dense, uniform initial point cloud by running a monocular depth
network (Depth Anything V2) over training images and back-projecting the depth
maps using existing COLMAP camera poses.

This replaces COLMAP's sparse points and ensures large, textureless areas
(like floors and walls) are blanketed with splats from iteration zero.

Usage:
    python scripts/generate_dense_pointcloud.py \
        --transforms data/transforms.json \
        --sparse-ply data/point_clouds/sparse_pc.ply \
        --images-dir data/images \
        --output-dir data/point_clouds \
        --depth-model-path <path-to-depth_anything_v2_vitl.pth> \
        --depth-repo-path <path-to-Depth-Anything-V2> \
        --sample-every 4 \
        --voxel-size 0.005

Requirements:
    pip install torch torchvision opencv-python numpy open3d scikit-learn plyfile
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dense point cloud from depth estimation + COLMAP poses"
    )
    parser.add_argument("--transforms", required=True, help="Path to transforms.json")
    parser.add_argument("--sparse-ply", required=True, help="Path to sparse_pc.ply")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for dense PLY files")
    parser.add_argument("--depth-model-path", required=True, help="Path to depth_anything_v2_vitl.pth")
    parser.add_argument("--depth-repo-path", required=True, help="Path to Depth-Anything-V2 repo clone")
    parser.add_argument("--sample-every", type=int, default=4, help="Sample every N-th frame (default: 4)")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size for downsampling (default: 0.005)")
    parser.add_argument("--pixel-stride", type=int, default=2, help="Pixel stride for back-projection (default: 2)")
    return parser.parse_args()


def load_transforms(path):
    """Load camera transforms from transforms.json."""
    with open(path) as f:
        data = json.load(f)
    return data


def load_sparse_points(path):
    """Load sparse COLMAP points from PLY file."""
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def load_depth_model(model_path, repo_path):
    """Load Depth Anything V2 model."""
    sys.path.insert(0, str(repo_path))
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
    }
    model = DepthAnythingV2(**model_configs["vitl"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to("cuda").eval()
    return model


def world_to_cam(point_world, c2w):
    """Transform a world point into camera coordinates using a c2w matrix."""
    w2c = np.linalg.inv(c2w)
    p_hom = np.append(point_world, 1.0)
    p_cam = w2c @ p_hom
    return p_cam[:3]


def align_depth_to_colmap(depth_pred, c2w, fx, fy, cx, cy, w, h, sparse_pts, max_proj=5000):
    """Align predicted relative depth to COLMAP scale using RANSAC regression."""
    w2c = np.linalg.inv(c2w)
    pts_h = np.hstack([sparse_pts, np.ones((len(sparse_pts), 1))])
    pts_cam = (w2c @ pts_h.T).T[:, :3]

    # Keep points in front of camera
    mask = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[mask]
    if len(pts_cam) < 10:
        return None, None

    # Project to pixel coordinates
    u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(int)
    v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(int)

    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, pts_cam = u[valid], v[valid], pts_cam[valid]
    if len(u) < 10:
        return None, None

    # Subsample if too many
    if len(u) > max_proj:
        idx = np.random.choice(len(u), max_proj, replace=False)
        u, v, pts_cam = u[idx], v[idx], pts_cam[idx]

    d_pred = depth_pred[v, u]
    d_colmap = pts_cam[:, 2]

    # RANSAC fit: d_colmap = scale * d_pred + shift
    reg = RANSACRegressor(residual_threshold=0.5, max_trials=1000)
    reg.fit(d_pred.reshape(-1, 1), d_colmap)
    scale = reg.estimator_.coef_[0]
    shift = reg.estimator_.intercept_

    return scale, shift


def backproject_depth(depth_aligned, c2w, fx, fy, cx, cy, image_rgb, stride=2):
    """Back-project aligned depth map into 3D world coordinates."""
    h, w = depth_aligned.shape
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.flatten(), vv.flatten()

    d = depth_aligned[vv, uu]
    valid = d > 0.01
    uu, vv, d = uu[valid], vv[valid], d[valid]

    # Pixel to camera coordinates
    x_cam = (uu - cx) * d / fx
    y_cam = (vv - cy) * d / fy
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]

    colors = image_rgb[vv, uu] / 255.0

    return pts_world, colors


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Dense Point Cloud Generation Pipeline")
    print("=" * 60)

    # Load transforms
    print("\n[1/5] Loading camera transforms...")
    transforms = load_transforms(args.transforms)
    frames = transforms["frames"]
    print(f"  Found {len(frames)} frames")

    # Load sparse points
    print("[2/5] Loading sparse COLMAP points...")
    sparse_pts = load_sparse_points(args.sparse_ply)
    print(f"  Loaded {len(sparse_pts)} sparse points")

    # Load depth model
    print("[3/5] Loading Depth Anything V2 (ViT-Large)...")
    model = load_depth_model(args.depth_model_path, args.depth_repo_path)
    print("  Model loaded on GPU")

    # Sample frames
    sampled_frames = frames[:: args.sample_every]
    print(f"\n[4/5] Processing {len(sampled_frames)} keyframes (every {args.sample_every}th)...")

    all_points = []
    all_colors = []

    for i, frame in enumerate(sampled_frames):
        img_path = os.path.join(args.images_dir, os.path.basename(frame["file_path"]))
        if not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        c2w = np.array(frame["transform_matrix"])
        fx = transforms.get("fl_x", transforms.get("camera_angle_x", 1000))
        fy = transforms.get("fl_y", fx)
        cx = transforms.get("cx", w / 2)
        cy = transforms.get("cy", h / 2)

        # Predict depth
        with torch.no_grad():
            depth_pred = model.infer_image(img_bgr)

        # Align to COLMAP scale
        scale, shift = align_depth_to_colmap(depth_pred, c2w, fx, fy, cx, cy, w, h, sparse_pts)
        if scale is None:
            continue

        depth_aligned = scale * depth_pred + shift
        depth_aligned = np.clip(depth_aligned, 0.01, None)

        # Back-project
        pts, cols = backproject_depth(depth_aligned, c2w, fx, fy, cx, cy, img_rgb, stride=args.pixel_stride)
        all_points.append(pts)
        all_colors.append(cols)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i + 1}/{len(sampled_frames)}: {len(pts)} points")

    # Combine all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    print(f"\n  Total raw points: {len(all_points):,}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Statistical outlier removal
    print("\n[5/5] Cleaning and downsampling...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal: {len(pcd.points):,}")

    # Voxel downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"  After voxel downsample (voxel={args.voxel_size}): {len(pcd_down.points):,}")

    # Save
    output_path = os.path.join(args.output_dir, "dense_pc.ply")
    o3d.io.write_point_cloud(output_path, pcd_down)
    print(f"\n✅ Saved dense point cloud: {output_path}")
    print(f"   Points: {len(pcd_down.points):,}")


if __name__ == "__main__":
    main()
