#!/usr/bin/env python3
"""
export_and_convert.py — Export trained Gaussian Splatting model and convert to USDZ.

Wraps the ns-export and usd-convert-gsplat commands into a single script
for streamlined model export to Isaac Sim-compatible USDZ format.

Usage:
    python scripts/export_and_convert.py \
        --config outputs/splatfacto/<timestamp>/config.yml \
        --output-dir outputs/splat \
        --convert-usdz

Requirements:
    - nerfstudio (ns-export command)
    - usd-convert-gsplat (from usd_convert conda env)
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained 3DGS model and optionally convert to USDZ"
    )
    parser.add_argument("--config", required=True,
                        help="Path to nerfstudio config.yml from training")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save exported PLY and USDZ")
    parser.add_argument("--convert-usdz", action="store_true",
                        help="Also convert the exported PLY to USDZ format")
    parser.add_argument("--gs-conda-env", default="gs_pipeline",
                        help="Conda environment for nerfstudio (default: gs_pipeline)")
    parser.add_argument("--usd-conda-env", default="usd_convert",
                        help="Conda environment for usd-convert-gsplat (default: usd_convert)")
    return parser.parse_args()


def run_cmd(cmd, env_vars=None):
    """Run a command and stream output."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ply_path = os.path.join(args.output_dir, "splat.ply")
    usdz_path = os.path.join(args.output_dir, "splat.usdz")

    # Step 1: Export Gaussian Splat PLY
    print("=" * 60)
    print("[1/2] Exporting Gaussian Splat to PLY...")
    print("=" * 60)

    export_cmd = (
        f"conda run --no-capture-output -n {args.gs_conda_env} "
        f"ns-export gaussian-splat "
        f"--load-config {args.config} "
        f"--output-dir {args.output_dir}"
    )
    run_cmd(export_cmd, env_vars={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    print(f"✅ PLY exported: {ply_path}")

    # Step 2: Convert to USDZ (optional)
    if args.convert_usdz:
        print(f"\n{'=' * 60}")
        print("[2/2] Converting PLY to USDZ...")
        print("=" * 60)

        convert_cmd = (
            f"conda run -n {args.usd_conda_env} "
            f"usd-convert-gsplat "
            f"-i {ply_path} "
            f"-o {usdz_path}"
        )
        run_cmd(convert_cmd)
        print(f"✅ USDZ saved: {usdz_path}")
    else:
        print("\n⏭️  Skipping USDZ conversion (use --convert-usdz to enable)")

    print(f"\n🎉 Export complete!")
    print(f"   PLY:  {ply_path}")
    if args.convert_usdz:
        print(f"   USDZ: {usdz_path}")


if __name__ == "__main__":
    main()
