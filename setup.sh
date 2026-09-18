#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# setup.sh — One-time environment setup for 3DGS Isaac Sim Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script:
#   1. Creates the gs_pipeline conda environment with all dependencies
#   2. Creates the usd_convert conda environment for USDZ conversion
#   3. Clones Depth-Anything-V2 into third_party/
#   4. Downloads the ViT-Large depth model weights
#
# Prerequisites:
#   - conda (Miniconda or Anaconda) must be installed
#   - NVIDIA GPU driver must be installed (check with: nvidia-smi)
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"

echo "═══════════════════════════════════════════════════════════════"
echo "  3DGS Isaac Sim Pipeline — Environment Setup"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Check prerequisites ──────────────────────────────────────────────
echo ""
echo "[1/5] Checking prerequisites..."

if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "  ✅ conda found: $(conda --version)"

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. NVIDIA GPU driver is required."
    exit 1
fi
echo "  ✅ GPU driver found"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/  ✅ GPU: /'

# ── Step 2: Create gs_pipeline conda environment ─────────────────────────────
echo ""
echo "[2/5] Creating gs_pipeline conda environment..."

if conda env list | grep -q "gs_pipeline"; then
    echo "  ⏭️  gs_pipeline already exists, skipping creation"
else
    conda create -n gs_pipeline python=3.10 -y
    echo "  ✅ Created gs_pipeline environment"
fi

echo "  Installing dependencies..."
conda run -n gs_pipeline pip install -q \
    nerfstudio \
    open3d \
    plyfile \
    scikit-learn \
    opencv-python \
    numpy \
    torch \
    torchvision
echo "  ✅ gs_pipeline dependencies installed"

# ── Step 3: Create usd_convert conda environment ─────────────────────────────
echo ""
echo "[3/5] Creating usd_convert conda environment..."

if conda env list | grep -q "usd_convert"; then
    echo "  ⏭️  usd_convert already exists, skipping creation"
else
    conda create -n usd_convert python=3.10 -y
    echo "  ✅ Created usd_convert environment"
fi

echo "  Installing usd-convert-gsplat..."
conda run -n usd_convert pip install -q usd-convert-gsplat
echo "  ✅ usd_convert dependencies installed"

# ── Step 4: Clone Depth-Anything-V2 ──────────────────────────────────────────
echo ""
echo "[4/5] Setting up Depth Anything V2..."

mkdir -p "$THIRD_PARTY_DIR"

if [ -d "$THIRD_PARTY_DIR/Depth-Anything-V2" ]; then
    echo "  ⏭️  Depth-Anything-V2 already cloned"
else
    echo "  Cloning Depth-Anything-V2..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git \
        "$THIRD_PARTY_DIR/Depth-Anything-V2"
    echo "  ✅ Cloned to third_party/Depth-Anything-V2/"
fi

# ── Step 5: Download depth model weights ─────────────────────────────────────
echo ""
echo "[5/5] Downloading depth model weights..."

WEIGHTS_DIR="$THIRD_PARTY_DIR/Depth-Anything-V2"
WEIGHTS_FILE="$WEIGHTS_DIR/depth_anything_v2_vitl.pth"

if [ -f "$WEIGHTS_FILE" ]; then
    echo "  ⏭️  Model weights already downloaded"
else
    echo "  Downloading depth_anything_v2_vitl.pth (1.3 GB)..."
    wget -q --show-progress -O "$WEIGHTS_FILE" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    echo "  ✅ Downloaded model weights"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ Setup Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "    1. Place your video file in: data/video/"
echo "    2. Run the full pipeline:    ./run_pipeline.sh data/video/your_video.MOV"
echo ""
echo "  Environment variable (add to ~/.bashrc):"
echo "    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
echo ""
