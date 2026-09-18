#!/usr/bin/env python3
"""
create_transforms.py — Create a custom transforms.json pointing to a different PLY file.

Useful when swapping between sparse, dense, or filtered point clouds
as the initial geometry for Gaussian Splatting training.

Usage:
    python scripts/create_transforms.py \
        --input data/transforms.json \
        --output data/transforms_dense.json \
        --ply-path dense_pc.ply

Requirements:
    Python 3.8+ (json is stdlib)
"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a transforms.json variant pointing to a different PLY"
    )
    parser.add_argument("--input", required=True, help="Original transforms.json path")
    parser.add_argument("--output", required=True, help="Output transforms.json path")
    parser.add_argument("--ply-path", required=True,
                        help="PLY file path to set in ply_file_path field")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    data["ply_file_path"] = args.ply_path

    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Created {args.output}")
    print(f"   ply_file_path = {args.ply_path}")


if __name__ == "__main__":
    main()
