#!/usr/bin/env python3
"""
blur_detector.py — Auto-detect and remove blurry frames from extracted video frames.

Uses the Laplacian variance method to calculate a sharpness score for each
frame. Frames below the threshold are discarded, ensuring only high-quality
images are used for COLMAP and Gaussian Splatting.

Usage:
    python scripts/blur_detector.py \
        --input-dir data/frames_raw \
        --output-dir data/images \
        --threshold 15

Requirements:
    pip install opencv-python
"""

import argparse
import os
import shutil
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-detect and remove blurry frames"
    )
    parser.add_argument("--input-dir", required=True, help="Directory of raw frames")
    parser.add_argument("--output-dir", required=True, help="Directory for sharp frames")
    parser.add_argument("--threshold", type=float, default=15,
                        help="Laplacian variance threshold (default: 15)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kept, removed = 0, 0
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filepath = os.path.join(args.input_dir, fname)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var >= args.threshold:
            shutil.copy(filepath, os.path.join(args.output_dir, fname))
            print(f"✅ KEEP  {fname} (sharpness: {laplacian_var:.1f})")
            kept += 1
        else:
            print(f"❌ SKIP  {fname} (sharpness: {laplacian_var:.1f})")
            removed += 1

    print(f"\n📊 Results: {kept} kept, {removed} removed out of {kept + removed} total")


if __name__ == "__main__":
    main()
