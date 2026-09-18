# 3D Gaussian Splatting Pipeline for NVIDIA Isaac Sim

End-to-end pipeline for creating photorealistic 3D Gaussian Splatting models from video footage and importing them into NVIDIA Isaac Sim with physics-ready collision meshes.

**Video → Frames → COLMAP → 3DGS Training → PLY → USDZ → Isaac Sim**

---

## 📁 Project Structure

```
project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.sh                           # ⚡ One-time setup (installs everything)
├── run_pipeline.sh                    # ⚡ Full end-to-end pipeline
├── .gitignore                         # Git ignore rules
│
├── scripts/                           # All pipeline scripts
│   ├── blur_detector.py               # Step 2: Remove blurry frames
│   ├── generate_dense_pointcloud.py   # Optional: Dense point cloud via depth estimation
│   ├── filter_gaussians.py            # Post-processing: Basic floater removal
│   ├── remove_floaters_smart.py       # Post-processing: Aggressive floater removal
│   ├── create_transforms.py           # Utility: Swap PLY references in transforms.json
│   ├── create_collision_mesh.py       # Step 7: Poisson collision mesh for physics
│   └── export_and_convert.py          # Steps 5-6: Export PLY → USDZ
│
├── configs/                           # Configuration files
│   ├── training_presets.ini           # Splatfacto training parameter presets
│   └── environment_setup.txt          # Manual environment setup reference
│
├── data/                              # Working data (not committed to Git)
│   ├── video/                         # 📱 PUT YOUR VIDEO HERE
│   ├── frames/                        # Extracted raw frames
│   ├── frames_clean/                  # Blur-filtered frames
│   ├── processed/                     # COLMAP output + transforms.json
│   ├── point_clouds/                  # PLY point clouds (sparse, dense, filtered)
│   ├── exports/                       # Intermediate exports
│   └── logs/                          # Pipeline logs
│
├── outputs/                           # Final outputs (not committed to Git)
│   ├── splat/                         # splat.ply, splat.usdz
│   └── collision/                     # collision_map.obj, collision_map.ply
│
├── third_party/                       # Cloned dependencies (created by setup.sh)
│   └── Depth-Anything-V2/            # Depth estimation model (auto-cloned)
│
└── docs/                              # Documentation
    └── full_pipeline_guide.md         # Complete 1,100-line step-by-step guide
```

---

## 🏁 Quick Start (3 Commands)

### Step 1: Clone and Setup

```bash
# Clone this repository
https://github.com/rakeshsuthar6322/3d_gaussian_splatting_for_model_city.git
cd 3d_gaussian_splatting_for_model_city

# Run one-time setup (creates conda environments, clones Depth-Anything-V2,
# downloads model weights — takes ~10 minutes)
chmod +x setup.sh
./setup.sh
```

> **What `setup.sh` does automatically:**
> 1. Creates `gs_pipeline` conda environment with nerfstudio, Open3D, PyTorch, etc.
> 2. Creates `usd_convert` conda environment with usd-convert-gsplat
> 3. Clones [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) into `third_party/`
> 4. Downloads the ViT-Large model weights (`depth_anything_v2_vitl.pth`, 1.3 GB)

### Step 2: Place Your Video

```bash
# Copy your video into the data/video/ directory
cp /path/to/your/scene_video.MOV data/video/
```

### Step 3: Run the Pipeline

```bash
# Set required environment variable
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Run the full end-to-end pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh data/video/scene_video.MOV
```

That's it! ☕ Go grab a coffee. When it finishes (~30-45 minutes depending on video length and GPU), you'll find:

| Output File | Purpose |
|:---|:---|
| `outputs/splat/splat.ply` | Gaussian Splat (view in [SuperSplat](https://playcanvas.com/supersplat/editor)) |
| `outputs/splat/splat.usdz` | Isaac Sim visual asset |
| `outputs/collision/collision_map.obj` | Physics collision mesh |
| `outputs/collision/collision_map.ply` | Physics collision mesh (PLY) |

---

## ⚙️ Prerequisites

| Requirement | Why | How to Check |
|:---|:---|:---|
| **NVIDIA GPU** (≥8 GB VRAM) | 3DGS training | `nvidia-smi` |
| **NVIDIA GPU Driver** | CUDA support | `nvidia-smi` |
| **Conda** (Miniconda or Anaconda) | Environment management | `conda --version` |
| **FFmpeg** | Video frame extraction | `ffmpeg -version` |
| **Git** | Cloning dependencies | `git --version` |

> [!NOTE]
> No `sudo` access is required. Everything installs into your home directory via Conda.

---

## 🔧 What Gets Cloned / Downloaded

The `setup.sh` script automatically handles all of this:

| Dependency | What | Where it's placed | Size |
|:---|:---|:---|:---|
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | Monocular depth estimation repo | `third_party/Depth-Anything-V2/` | ~50 MB |
| `depth_anything_v2_vitl.pth` | ViT-Large model weights | `third_party/Depth-Anything-V2/` | ~1.3 GB |

**Conda packages installed automatically:**

| Environment | Packages |
|:---|:---|
| `gs_pipeline` | nerfstudio, open3d, plyfile, scikit-learn, opencv-python, torch, torchvision |
| `usd_convert` | usd-convert-gsplat |

---

## 🧪 Running Individual Steps

If you want to run steps individually instead of the full pipeline:

### Extract Frames
```bash
ffmpeg -i data/video/scene.MOV -vf fps=2 -q:v 1 data/frames/frame_%05d.jpg
```

### Remove Blurry Frames
```bash
conda run -n gs_pipeline python scripts/blur_detector.py \
    --input-dir data/frames \
    --output-dir data/frames_clean \
    --threshold 15
```

### COLMAP Processing
```bash
conda run -n gs_pipeline ns-process-data images \
    --data data/frames_clean \
    --output-dir data/processed \
    --skip-image-processing
```

### Train (with Refined preset)
```bash
conda run --no-capture-output -n gs_pipeline ns-train splatfacto \
    --data data/processed \
    --output-dir data/processed/training_output \
    --pipeline.model.densify-grad-thresh 0.0002 \
    --pipeline.model.stop-split-at 15000 \
    --pipeline.model.cull-scale-thresh 0.2 \
    --max-num-iterations 30000 \
    --steps-per-save 1000 \
    --viewer.quit-on-train-completion True
```

### Export PLY + USDZ
```bash
python scripts/export_and_convert.py \
    --config data/processed/training_output/splatfacto/<timestamp>/config.yml \
    --output-dir outputs/splat \
    --convert-usdz
```

### Generate Collision Mesh
```bash
conda run -n gs_pipeline python scripts/create_collision_mesh.py \
    --input outputs/splat/splat.ply \
    --output-dir outputs/collision \
    --depth 11 \
    --target-triangles 500000
```

### Optional: Remove Floaters
```bash
conda run -n gs_pipeline python scripts/filter_gaussians.py \
    --input outputs/splat/splat.ply \
    --output outputs/splat/splat_filtered.ply \
    --min-opacity 0.05 \
    --max-scale 0.15
```

### Optional: Dense Point Cloud (for scenes with large holes)
```bash
conda run -n gs_pipeline python scripts/generate_dense_pointcloud.py \
    --transforms data/processed/transforms.json \
    --sparse-ply data/processed/sparse_pc.ply \
    --images-dir data/processed/images \
    --output-dir data/point_clouds \
    --depth-model-path third_party/Depth-Anything-V2/depth_anything_v2_vitl.pth \
    --depth-repo-path third_party/Depth-Anything-V2
```

---

## 📊 Training Presets

| Preset | `densify_grad_thresh` | `stop_split_at` | `cull_scale_thresh` | Best For |
|:---|:---|:---|:---|:---|
| **Default** | 0.0008 | 15000 | 0.5 | General scenes |
| **Aggressive** | 0.0001 | 25000 | 0.5 | Scenes with large holes |
| **Refined** ⭐ | 0.0002 | 15000 | 0.2 | Best quality (default in pipeline) |

See [`configs/training_presets.ini`](configs/training_presets.ini) for details.

---

## 🤖 Importing into Isaac Sim

1. Drag `outputs/splat/splat.usdz` into your Isaac Sim stage
2. Drag `outputs/collision/collision_map.obj` into the same stage
3. Align both at `(0, 0, 0)`
4. Set the collision mesh to **Invisible** (Property → Visibility)
5. Right-click collision mesh → Add → Physics → **Collider**
6. Your robots will collide with the invisible mesh while seeing the photorealistic splats!

---

## 📋 Key Lessons Learned

1. **Aggressive densification** fills holes but may create floaters — follow up with filter scripts
2. **`cull_scale_thresh = 0.2`** prevents large clumpy splats in detailed areas
3. **Stopping densification early** (15k/30k) gives 15k iterations for pure color optimization
4. **Smart floater removal** (scale < 0.02 AND opacity > 0.05) preserves surfaces while removing haze
5. **Poisson depth 11** gives the best detail-to-performance ratio for collision meshes
6. **Blur detection** before COLMAP dramatically improves reconstruction quality

---

## 📄 License

This project is for research and educational purposes.

---

## 🙏 Acknowledgments

- [Nerfstudio](https://nerf.studio/) — Gaussian Splatting training framework
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Monocular depth estimation
- [Open3D](http://www.open3d.org/) — Point cloud processing and mesh reconstruction
- [SuperSplat](https://playcanvas.com/supersplat/editor) — Gaussian Splat viewer and editor
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) — Robotics simulation platform
