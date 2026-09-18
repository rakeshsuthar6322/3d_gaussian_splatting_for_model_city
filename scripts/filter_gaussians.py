#!/usr/bin/env python3
"""
filter_gaussians.py — Remove low-opacity and oversized Gaussian blobs from a PLY export.

Filters out "floater" artifacts from trained Gaussian Splatting models by
removing splats that are either too transparent (low opacity) or too large
(oversized scale), which typically correspond to hazy clouds floating in
empty space.

Usage:
    python scripts/filter_gaussians.py \
        --input data/point_clouds/splat.ply \
        --output data/point_clouds/splat_filtered.ply \
        --min-opacity 0.05 \
        --max-scale 0.15

Requirements:
    pip install numpy plyfile
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove low-opacity and oversized Gaussian blobs from a PLY export"
    )
    parser.add_argument("--input", required=True, help="Input PLY file path")
    parser.add_argument("--output", required=True, help="Output filtered PLY file path")
    parser.add_argument("--min-opacity", type=float, default=0.05,
                        help="Minimum opacity threshold (default: 0.05)")
    parser.add_argument("--max-scale", type=float, default=0.15,
                        help="Maximum scale threshold (default: 0.15)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input}...")
    ply = PlyData.read(args.input)
    vertices = ply["vertex"].data

    # Calculate opacity (inverse sigmoid)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"]))

    # Calculate scale (exponentiated)
    scales = np.exp(
        np.column_stack((vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]))
    )

    # Filter: keep splats with sufficient opacity and reasonable scale
    keep = (opacity >= args.min_opacity) & (scales.max(axis=1) <= args.max_scale)
    filtered = vertices[keep]

    # Write output
    PlyData(
        [PlyElement.describe(filtered, "vertex")],
        text=False,
        byte_order=ply.byte_order,
        comments=ply.comments,
    ).write(args.output)

    print(f"✅ Kept {len(filtered):,} / {len(vertices):,} Gaussians")
    print(f"   Removed {len(vertices) - len(filtered):,} low-opacity or oversized Gaussians")
    print(f"   Saved to: {args.output}")


if __name__ == "__main__":
    main()
