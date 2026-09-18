#!/usr/bin/env python3
"""
create_collision_mesh.py — Generate a physics collision mesh for NVIDIA Isaac Sim.

Extracts splat center positions from a Gaussian Splatting PLY file, estimates
surface normals, runs Poisson Surface Reconstruction to create a watertight
triangle mesh, filters hallucinated outer geometry, and optionally decimates
to a target triangle count for real-time physics performance.

Usage:
    python scripts/create_collision_mesh.py \
        --input data/point_clouds/final_splat.ply \
        --output-dir outputs/collision \
        --depth 11 \
        --target-triangles 500000

Requirements:
    pip install open3d numpy plyfile
"""

import argparse
import time
import numpy as np
import open3d as o3d
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate collision mesh from Gaussian Splatting PLY"
    )
    parser.add_argument("--input", required=True, help="Input Gaussian Splatting PLY file")
    parser.add_argument("--output-dir", required=True, help="Output directory for collision mesh")
    parser.add_argument("--depth", type=int, default=11,
                        help="Poisson reconstruction depth (default: 11, higher = more detail)")
    parser.add_argument("--target-triangles", type=int, default=500000,
                        help="Target triangle count for decimation (default: 500000)")
    parser.add_argument("--density-quantile", type=float, default=0.05,
                        help="Bottom quantile of density to remove (default: 0.05)")
    parser.add_argument("--knn", type=int, default=30,
                        help="K-nearest neighbors for normal estimation (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    output_obj = os.path.join(args.output_dir, "collision_map.obj")
    output_ply = os.path.join(args.output_dir, "collision_map.ply")

    print("=" * 60)
    print("Collision Mesh Generation Pipeline")
    print("=" * 60)

    # Load PLY
    print(f"\n[1/5] Loading splat PLY: {args.input}")
    t0 = time.time()
    plydata = PlyData.read(args.input)
    vertex_data = plydata.elements[0].data

    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    print(f"  Loaded {len(points):,} points in {time.time() - t0:.2f}s")

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate Normals
    print(f"\n[2/5] Estimating normals (KNN={args.knn})...")
    t0 = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.knn))
    pcd.orient_normals_consistent_tangent_plane(100)
    print(f"  Normals estimated in {time.time() - t0:.2f}s")

    # Poisson Surface Reconstruction
    print(f"\n[3/5] Poisson Surface Reconstruction (depth={args.depth})...")
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth
    )
    print(f"  Completed in {time.time() - t0:.2f}s — Triangles: {len(mesh.triangles):,}")

    # Filter low-density artifacts
    print(f"\n[4/5] Filtering low density artifacts (quantile={args.density_quantile})...")
    vertices_to_remove = densities < np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  Post-filter triangles: {len(mesh.triangles):,}")

    # Decimate if necessary
    if len(mesh.triangles) > args.target_triangles:
        print(f"\n[5/5] Decimating from {len(mesh.triangles):,} to {args.target_triangles:,} triangles...")
        t0 = time.time()
        mesh = mesh.simplify_quadric_decimation(args.target_triangles)
        print(f"  Decimation completed in {time.time() - t0:.2f}s")
    else:
        print(f"\n[5/5] No decimation needed ({len(mesh.triangles):,} <= {args.target_triangles:,})")

    # Compute vertex normals for smooth shading
    mesh.compute_vertex_normals()

    # Export
    print(f"\nSaving to {output_obj} and {output_ply}...")
    o3d.io.write_triangle_mesh(output_obj, mesh)
    o3d.io.write_triangle_mesh(output_ply, mesh)
    print(f"\n✅ Collision mesh is ready!")
    print(f"   OBJ: {output_obj}")
    print(f"   PLY: {output_ply}")
    print(f"   Triangles: {len(mesh.triangles):,}")


if __name__ == "__main__":
    main()
