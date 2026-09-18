#!/usr/bin/env python3
"""
remove_floaters_smart.py — Intelligent floater removal using scale + opacity analysis.

A more aggressive floater removal strategy that combines scale thresholds
with opacity filtering to surgically remove "hazy cloud" artifacts from
trained Gaussian Splatting exports without destroying surface geometry.

Usage:
    python scripts/remove_floaters_smart.py \
        --input data/point_clouds/splat.ply \
        --output data/point_clouds/splat_clean.ply \
        --max-scale 0.02 \
        --min-opacity 0.05

Requirements:
    pip install numpy plyfile
"""

import argparse
import time
import numpy as np
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart floater removal using scale + opacity thresholds"
    )
    parser.add_argument("--input", required=True, help="Input PLY file path")
    parser.add_argument("--output", required=True, help="Output filtered PLY file path")
    parser.add_argument("--max-scale", type=float, default=0.02,
                        help="Max scale threshold (default: 0.02)")
    parser.add_argument("--min-opacity", type=float, default=0.05,
                        help="Min opacity threshold (default: 0.05)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input}...")
    start = time.time()
    plydata = PlyData.read(args.input)
    print(f"Loaded PLY in {time.time() - start:.2f}s")

    vertex_data = plydata.elements[0].data

    # Calculate Scale (max of scale_0, scale_1, scale_2)
    scales = np.exp(np.c_[vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']])
    max_scales = scales.max(axis=1)

    # Calculate Opacity (inverse sigmoid)
    opacities = 1 / (1 + np.exp(-vertex_data['opacity']))

    # Create Mask: Keep points with small/normal scale AND high opacity
    mask = (max_scales < args.max_scale) & (opacities > args.min_opacity)

    num_kept = np.sum(mask)
    num_removed = len(vertex_data) - num_kept
    print(f"Total Gaussians: {len(vertex_data):,}")
    print(f"Kept: {num_kept:,} | Removed (floaters): {num_removed:,}")

    # Filter and write
    filtered_vertex_data = vertex_data[mask]
    filtered_element = PlyElement.describe(filtered_vertex_data, 'vertex')
    filtered_plydata = PlyData([filtered_element])

    print(f"Saving to {args.output}...")
    start = time.time()
    filtered_plydata.write(args.output)
    print(f"✅ Saved in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
