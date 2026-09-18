#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_pipeline.sh — Full end-to-end 3DGS pipeline
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh <path-to-video>
#
# Example:
#   ./run_pipeline.sh data/video/my_scene.MOV
#
# This script runs the ENTIRE pipeline from video to Isaac Sim-ready outputs:
#   1. Extract frames from video (FFmpeg)
#   2. Remove blurry frames (blur_detector.py)
#   3. Run COLMAP via nerfstudio (ns-process-data)
#   4. Train 3D Gaussian Splatting (ns-train splatfacto)
#   5. Export to PLY (ns-export)
#   6. Convert to USDZ (usd-convert-gsplat)
#   7. Generate collision mesh (create_collision_mesh.py)
#
# Prerequisites:
#   - Run setup.sh first to install all dependencies
#   - export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"

# Training parameters (Refined preset — best quality)
MAX_ITERATIONS=30000
DENSIFY_GRAD_THRESH=0.0002
STOP_SPLIT_AT=15000
CULL_SCALE_THRESH=0.2

# Frame extraction
FRAME_RATE=2              # Frames per second to extract
BLUR_THRESHOLD=15         # Laplacian variance threshold

# Collision mesh
POISSON_DEPTH=11          # Reconstruction depth (higher = more detail)
TARGET_TRIANGLES=500000   # Max triangles for physics performance

# Conda environments
GS_ENV="gs_pipeline"
USD_ENV="usd_convert"

# Export environment variable
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ── Input validation ─────────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-video>"
    echo "Example: $0 data/video/my_scene.MOV"
    exit 1
fi

VIDEO_PATH="$1"
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  3DGS Isaac Sim Pipeline — Full Run"
echo "═══════════════════════════════════════════════════════════════"
echo "  Video:      $VIDEO_PATH"
echo "  Iterations: $MAX_ITERATIONS"
echo "  Preset:     Refined"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Extract Frames ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/7] Extracting frames at ${FRAME_RATE} FPS..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$DATA_DIR/frames"
ffmpeg -i "$VIDEO_PATH" -vf "fps=$FRAME_RATE" -q:v 1 \
    "$DATA_DIR/frames/frame_%05d.jpg" \
    -y 2>&1 | tail -5

FRAME_COUNT=$(ls "$DATA_DIR/frames"/*.jpg 2>/dev/null | wc -l)
echo "✅ Extracted $FRAME_COUNT frames"
echo ""

# ── Step 2: Remove Blurry Frames ────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/7] Removing blurry frames (threshold=$BLUR_THRESHOLD)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run --no-capture-output -n $GS_ENV python "$SCRIPTS_DIR/blur_detector.py" \
    --input-dir "$DATA_DIR/frames" \
    --output-dir "$DATA_DIR/frames_clean" \
    --threshold $BLUR_THRESHOLD

CLEAN_COUNT=$(ls "$DATA_DIR/frames_clean"/*.jpg 2>/dev/null | wc -l)
echo "✅ $CLEAN_COUNT clean frames ready"
echo ""

# ── Step 3: COLMAP Processing ────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/7] Running COLMAP via nerfstudio..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run --no-capture-output -n $GS_ENV ns-process-data images \
    --data "$DATA_DIR/frames_clean" \
    --output-dir "$DATA_DIR/processed" \
    --skip-image-processing \
    2>&1 | tee "$DATA_DIR/logs/colmap.log"

echo "✅ COLMAP processing complete"
echo ""

# ── Step 4: Train Gaussian Splatting ─────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/7] Training 3D Gaussian Splatting ($MAX_ITERATIONS iterations)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TRAINING_OUTPUT="$DATA_DIR/processed/training_output"

conda run --no-capture-output -n $GS_ENV ns-train splatfacto \
    --data "$DATA_DIR/processed" \
    --output-dir "$TRAINING_OUTPUT" \
    --pipeline.model.densify-grad-thresh $DENSIFY_GRAD_THRESH \
    --pipeline.model.stop-split-at $STOP_SPLIT_AT \
    --pipeline.model.cull-scale-thresh $CULL_SCALE_THRESH \
    --max-num-iterations $MAX_ITERATIONS \
    --steps-per-save 1000 \
    --viewer.quit-on-train-completion True \
    2>&1 | tee "$DATA_DIR/logs/training.log"

echo "✅ Training complete"
echo ""

# ── Step 5: Find config and Export PLY ───────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/7] Exporting trained model to PLY..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Auto-detect the training config path
CONFIG_PATH=$(find "$TRAINING_OUTPUT" -name "config.yml" -type f | head -1)
if [ -z "$CONFIG_PATH" ]; then
    echo "❌ Could not find config.yml in training output"
    exit 1
fi
echo "  Found config: $CONFIG_PATH"

mkdir -p "$OUTPUTS_DIR/splat"

conda run --no-capture-output -n $GS_ENV ns-export gaussian-splat \
    --load-config "$CONFIG_PATH" \
    --output-dir "$OUTPUTS_DIR/splat" \
    2>&1 | tee "$DATA_DIR/logs/export.log"

echo "✅ PLY exported: $OUTPUTS_DIR/splat/splat.ply"
echo ""

# ── Step 6: Convert to USDZ ─────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/7] Converting PLY to USDZ for Isaac Sim..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run --no-capture-output -n $USD_ENV usd-convert-gsplat \
    -i "$OUTPUTS_DIR/splat/splat.ply" \
    -o "$OUTPUTS_DIR/splat/splat.usdz" \
    2>&1 | tee "$DATA_DIR/logs/usd_convert.log"

echo "✅ USDZ saved: $OUTPUTS_DIR/splat/splat.usdz"
echo ""

# ── Step 7: Generate Collision Mesh ──────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[7/7] Generating collision mesh for Isaac Sim physics..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$OUTPUTS_DIR/collision"

conda run --no-capture-output -n $GS_ENV python "$SCRIPTS_DIR/create_collision_mesh.py" \
    --input "$OUTPUTS_DIR/splat/splat.ply" \
    --output-dir "$OUTPUTS_DIR/collision" \
    --depth $POISSON_DEPTH \
    --target-triangles $TARGET_TRIANGLES \
    2>&1 | tee "$DATA_DIR/logs/collision.log"

echo "✅ Collision mesh saved"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  🎉 Pipeline Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  📁 Final Outputs:"
echo "  ├── outputs/splat/splat.ply          — Gaussian Splat (view in SuperSplat)"
echo "  ├── outputs/splat/splat.usdz         — Isaac Sim visual asset"
echo "  ├── outputs/collision/collision_map.obj — Physics collision mesh"
echo "  └── outputs/collision/collision_map.ply — Physics collision mesh (PLY)"
echo ""
echo "  🤖 Isaac Sim Import:"
echo "  1. Drag splat.usdz into your Isaac Sim stage"
echo "  2. Drag collision_map.obj into the same stage"
echo "  3. Align both at (0, 0, 0)"
echo "  4. Set collision mesh to Invisible"
echo "  5. Add Physics → Collider to the collision mesh"
echo ""
echo "  📋 Logs saved to: data/logs/"
echo ""
