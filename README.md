# mri_transfer

Two-stage pipeline for MRI tongue segmentation → 3D mesh retargeting (Stage 1) and ArtiSynth FEM forward simulation (Stage 2).

## Dataset

**The dataset is not included in this repository.** It is hosted separately on Hugging Face (gated — request access, then download after approval):

**https://huggingface.co/datasets/SeunghoEum/mri-tongue-dataset**

```bash
# After your access request is approved:
hf download SeunghoEum/mri-tongue-dataset --local-dir ./data
```

| Asset | Path in the HF dataset | Description |
|-------|------------------------|-------------|
| Rest tongue mesh | `tongue_model/tongue_rest_m.obj` | 3D reference mesh for registration |
| MRI masks | `datasets/GT_Segmentations/Subject3/mask_*.mat` | 2D segmentation masks (`.mat`) |
| SSFP cine (optional) | `datasets/MRI_SSFP_10fps/Subject*/image_*.dcm` | Raw DICOM frames |

When running from a standalone clone, use absolute paths or mount the downloaded folder as `/data` (see [Docker](#docker-recommended)).

Generated outputs (`_test_out/`, `*.obj`, `*.png`, `*.csv`, etc.) are also excluded via `.gitignore`.

---

## Requirements

- **Python 3.10+**
- **JDK 21** (for Stage 2 / ArtiSynth via JPype)
- **ArtiSynth** built with `artisynth_models` (HexTongueDemo), or use Docker (ArtiSynth is built inside the image)
- Headless rendering (optional): `xvfb`, `libxrender1`

```bash
pip install -r requirements.txt
```

---

## Settings

Configuration uses [Hydra](https://hydra.cc/). Defaults live under `configs/`.

### Main config — `configs/configs.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `stage` | `all` | Run mode: `retarget`, `fem`, or `all` |
| `paths.tongue_obj` | `tongue_model/tongue_rest_m.obj` | Rest 3D mesh (OBJ) |
| `paths.mask_dir` | `datasets/GT_Segmentations/Subject3` | Folder of `mask_*.mat` files |
| `paths.rest_mask` | `${paths.mask_dir}/mask_1.mat` | Rest-frame 2D mask for registration |
| `paths.target_mask` | `${paths.mask_dir}/mask_51.mat` | Target mask (file = single frame; folder = full sequence) |
| `paths.out_dir` | `null` → `_test_out/` | Output directory |
| `render.upper_degree` | `45` | Camera elevation (deg) |
| `render.right_degree` | `90` | Camera azimuth (deg, sagittal view) |
| `render.size` | `[640, 640]` | Render resolution |

Relative paths are resolved from the **parent directory of this repo** (the full project root that contains `datasets/` and `tongue_model/`). Use absolute paths when running a standalone clone.

### Stage 1 — `configs/retarget/default.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `mm_per_px` | `1.164` | Pixel size (mm) for register / lift / retarget |
| `lift.nz` | `15` | Lift dome depth samples |
| `lift.half_w` | `30.0` | Half-width of lift dome (mm) |
| `retarget.nctrl` | `13` | RBF control points |
| `retarget.rbf_len` | `18.0` | RBF length scale (mm) |
| `contour.n_markers` | `25` | Dorsal contour markers |

### Stage 2 — `configs/artisynth/default.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `tongue_model` | `artisynth.models.tongue3d.HexTongueDemo` | ArtiSynth model class |
| `settle_t` | `0.4` | Hold time at target activation (s) |
| `maxstep` | `0.0005` | FEM integrator step (s) |
| `nramp` | `20` | Activation ramp steps |
| `incomp` | `AUTO` | Incompressibility: `OFF`, `AUTO`, `ELEMENT`, `NODAL` |
| `jvm_xmx` | `4g` | JVM heap size |
| `activations.*` | see yaml | 11D muscle activations (0–1) |

**Muscle order (11D):** `GGP`, `GGM`, `GGA`, `STY`, `GH`, `MH`, `HG`, `VERT`, `TRANS`, `IL`, `SL`

### Environment variables

| Variable | Description |
|----------|-------------|
| `ARTISYNTH_HOME` | Path to compiled ArtiSynth tree (`classes/` + `lib/*.jar`) |
| `TONGUE_MODEL` | Override ArtiSynth model class name |
| `OUT_DIR` | Override default output directory (`_test_out/`) |

### CLI overrides (Hydra)

Any setting can be overridden on the command line:

```bash
python main.py stage=fem artisynth.activations.HG=0.3 retarget.mm_per_px=1.2
python main.py paths.out_dir=/tmp/run1 render.size=[512,512]
```

---

## Stage 1 — Retargeting

Maps 2D MRI tongue masks to a deformed 3D surface mesh.

**Steps:** `register` → `lift` → `retarget`

### Run

```bash
# Full retarget stage (default paths)
python main.py stage=retarget

# Single target frame (default: mask_51.mat)
python main.py stage=retarget paths.target_mask=/path/to/mask_51.mat

# Entire sequence (all mask_*.mat in folder)
python main.py stage=retarget paths.target_mask=/path/to/GT_Segmentations/Subject3
```

### Outputs (`_test_out/` by default)

| File | Description |
|------|-------------|
| `registration.csv` | 2D↔3D anchor mapping |
| `tongue_lift_3d.npy` | Lifted 3D mask sequence |
| `retargeted.obj` | Single-frame retargeted mesh |
| `retargeted_objs/frame_*.obj` | Per-frame meshes (folder target) |

---

## Stage 2 — FEM (ArtiSynth forward)

Loads the HexTongueDemo FEM model, applies 11D muscle activations, and runs forward simulation to equilibrium.

### Run

```bash
# FEM only (no MRI data required if rest OBJ exists or synthetic fallback is used)
python main.py stage=fem

# Custom activations
python main.py stage=fem artisynth.activations.GGP=0.3 artisynth.activations.HG=0.2

# Tune solver
python main.py stage=fem artisynth.nramp=40 artisynth.settle_t=0.6
```

Set `ARTISYNTH_HOME` before running (unless using Docker):

```bash
export ARTISYNTH_HOME=/opt/artisynth/artisynth_core
python main.py stage=fem
```

### Outputs (`_test_out/` by default)

| File | Description |
|------|-------------|
| `rest.png` / `rest.obj` | Rest pose render and mesh |
| `fem.png` / `fem.obj` | Deformed pose after forward solve |

### Headless rendering

```bash
xvfb-run -a python main.py stage=fem
```

---

## Run both stages

```bash
python main.py                    # stage=all
python main.py stage=all paths.mask_dir=/data/datasets/GT_Segmentations/Subject3
```

---

## Docker (recommended)

Docker bundles JDK, ArtiSynth, JPype, and Python dependencies. **Data must still be mounted separately.**

```bash
cp docker.env.example .env   # set DATA_DIR to your local dataset root

# Expected layout under DATA_DIR:
#   tongue_model/tongue_rest_m.obj
#   datasets/GT_Segmentations/Subject3/mask_*.mat

docker compose build
docker compose run --rm shell

# Stage 1 + 2 (default paths under /data)
docker compose run --rm pipeline

# Stage 2 only
docker compose run --rm -e PIPELINE_ARGS="stage=fem" pipeline

# Stage 1 only
docker compose run --rm -e PIPELINE_ARGS="stage=retarget" pipeline
```

See `docker.env.example` for path overrides (`TONGUE_OBJ`, `MRI_MASK_DIR`, etc.).

---

## Project layout

```
main.py              Entry point (Hydra)
configs/             YAML settings
retarget/            Stage 1: register, lift, retarget
artisynth/           Stage 2: JPype + ArtiSynth FEM
modules/             I/O, visualization, paths
```
