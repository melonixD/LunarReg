# LunarReg

LunarReg is a working lunar image-registration proof of concept for aligning Chandrayaan optical imagery with lunar reference imagery such as LRO NAC. It combines a deployable research interface, a classical geometric registration baseline, a supplied crater-detection model, downloadable datasets, and a separate training kit for the future learned registration model.

This repository is ready for a **single-service Render deployment**. The same Render URL serves the website and the Python inference API.

> **Current scientific status:** the learned registration network has not yet been trained on a validated Chandrayaan–LRO paired dataset. The deployed SIFT + MAGSAC registration is real and functional, but the project does not claim competition-grade sub-pixel performance until the trained model is evaluated independently.

---

## 1. What works right now

| Component | Status | What it does |
|---|---|---|
| Web interface | Working | Uploads fixed/moving images, shows outputs and measured metrics |
| Browser baseline | Working | Estimates translation locally without sending images to a server |
| Render registration API | Working | Uses SIFT, Lowe-ratio filtering, USAC_MAGSAC and homography warping |
| Crater detector | Working | Runs the supplied YOLOv8n weights and returns boxes plus an annotated image |
| Sample pair | Included | Loads a prepared LRO demonstration pair with one click |
| Training handbook | Included | 21-page PDF covering the mathematics and training plan |
| Registration training kit | Included | PyTorch code, Perlin augmentation, configs, tests and inference scripts |
| Learned registration checkpoint | Not trained | Must be trained with genuine, geographically separated lunar pairs |

The interface deliberately does **not** invent mineral analysis, landing-safety scores or sub-pixel accuracy.

---

## 2. Repository map

```text
LunarReg/
├── index.html                       Website
├── assets/                          CSS and browser registration code
├── samples/                         Included demonstration pair
├── docs/
│   └── LunarReg_Model_Training_Handoff.pdf
├── downloads/
│   ├── LunarReg-Training-Kit-v1.zip
│   ├── training_tiles.zip
│   ├── lunar_crater_detector.zip
│   └── CH3_region_69S_32E_dataset.zip
├── inference-service/
│   ├── app.py                       FastAPI registration + crater API
│   ├── models/crater_detector.pt
│   └── requirements.render.txt
├── Dockerfile.render                Complete website + API image
├── render.yaml                      One-click Render Blueprint
├── api/                              Optional Vercel proxy functions
└── vercel.json                       Optional frontend-only Vercel config
```

---

## 3. Deploy the complete project on Render

### Recommended: Render Blueprint

The repository must contain `render.yaml` at its top level—not inside another folder.

1. Extract the provided project ZIP.
2. Open the extracted `LunarReg-Vercel` folder.
3. Copy **the contents inside that folder** into the root of your GitHub repository.
4. Commit and push the changes.
5. Open [Render](https://dashboard.render.com/).
6. Select **New → Blueprint**.
7. Connect the GitHub repository containing LunarReg.
8. Render will find `render.yaml` and show one service named `lunarreg`.
9. Approve the Blueprint and start deployment.
10. Wait for the first Docker build to finish. The initial build is slower because PyTorch, OpenCV and Ultralytics must be installed.

The service health check is:

```text
/api/health
```

After deployment, open the generated URL:

```text
https://lunarreg-xxxx.onrender.com
```

### Manual Render setup if Blueprint is unavailable

Create a **Web Service** with these settings:

| Setting | Value |
|---|---|
| Source | Your LunarReg GitHub repository |
| Runtime | Docker |
| Branch | `main` |
| Root directory | Leave blank |
| Dockerfile path | `./Dockerfile.render` |
| Docker build context | `.` |
| Health check path | `/api/health` |

Optional environment variables:

```text
MAX_IMAGE_SIDE=2400
MAX_UPLOAD_BYTES=20971520
CORS_ORIGINS=*
```

### Important Render note

The Blueprint starts on Render's free plan so the website and classical registration can be tested without changing the code. The first request after inactivity may be slow. YOLO/PyTorch inference uses substantially more memory than the static website; if crater inference is terminated for memory usage, move the service to a plan with more RAM.

---

## 4. Run the complete project locally

Install Docker Desktop, then from the repository root run:

```bash
docker build -f Dockerfile.render -t lunarreg .
docker run --rm -p 10000:10000 lunarreg
```

Open:

```text
http://localhost:10000
```

Health endpoint:

```text
http://localhost:10000/api/health
```

Interactive FastAPI documentation:

```text
http://localhost:10000/docs
```

---

## 5. How to use the website

### Image registration

1. Open **Registration console**.
2. Upload the reference image under **Fixed image**.
3. Upload the image that must be transformed under **Moving image**.
4. Select an engine:
   - **Auto:** uses the Render SIFT service when available.
   - **Browser baseline:** keeps computation in the browser and estimates translation only.
   - **SIFT + MAGSAC service:** estimates a projective homography on the server.
5. Click **Run registration**.
6. Inspect RMSE, inlier count, inlier ratio, median error, maximum error and spatial coverage.
7. Download the registered overlay as PNG.

Use **Load included sample** for a quick deployment test.

### Crater detection

1. Scroll to **Explore the supplied crater detector**.
2. Select a PNG, JPEG or WebP lunar image.
3. Click **Detect craters**.
4. The server returns an annotated image, candidate count, confidence values and bounding boxes.

The supplied detector was trained on LRO NAC tiles. Its output on OHRC, TMC-2 or IIRS imagery must be treated as exploratory because of sensor-domain differences.

---

## 6. API usage

### Health

```bash
curl https://YOUR-SERVICE.onrender.com/api/health
```

### Register two images

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/api/register \
  -F "fixed=@reference.png" \
  -F "moving=@source.png" \
  -o registration-result.json
```

The response contains:

- registered overlay as a base64 PNG;
- 3×3 homography matrix;
- accepted correspondence coordinates;
- RMSE and median/max reprojection error;
- inlier count and inlier ratio;
- convex-hull spatial coverage.

### Detect craters

```bash
curl -X POST "https://YOUR-SERVICE.onrender.com/api/detect-craters?confidence=0.25" \
  -F "image=@lunar-image.png" \
  -o crater-result.json
```

---

# MODEL TRAINING GUIDE

The downloadable `LunarReg-Training-Kit-v1.zip` is for training the **image-correspondence model**. It is separate from `crater_detector.pt`, which is already a trained one-class crater detector. Render hosts the demonstration and inference service; use a GPU machine or RunPod for training rather than attempting a long training run on the Render web service.

## 7. Training objective

The registration model learns correspondences between a source image (I_s) and a reference image (I_r). The complete system is hybrid:

1. A neural network proposes dense or semi-dense feature correspondences.
2. Confidence filtering removes weak candidates.
3. USAC_MAGSAC estimates a geometrically consistent homography.
4. Local gradient normalized cross-correlation refines coordinates to fractional-pixel positions.
5. Evaluation measures reprojection accuracy and spatial distribution.

Perlin noise is used only to create smooth synthetic illumination fields. It helps simulate changing Sun angle and broad shadows; it is never used as geometric ground truth.

---

## 8. Hardware and storage

Recommended training environment:

- Linux or RunPod PyTorch CUDA template;
- NVIDIA GPU with at least 16 GB VRAM;
- 50–100 GB persistent storage;
- Python 3.10 or 3.11;
- recent CUDA-compatible PyTorch;
- stable storage for checkpoints and logs.

The included configurations are:

| GPU/configuration | Config file | Batch | Accumulation |
|---|---|---:|---:|
| RTX 5090 | `configs/train_5090.yaml` | 4 | 8 |
| RTX PRO 5000 | `configs/train_pro5000.yaml` | 6 | 6 |
| Installation test | `configs/smoke.yaml` | 4 | 1 |

Actual time depends on tile count, storage speed, PyTorch version and validation frequency. Always run the smoke configuration before paying for a long GPU session.

---

## 9. Extract and install the training kit

Download `LunarReg-Training-Kit-v1.zip` from the website or use the copy in `downloads/`.

```bash
unzip LunarReg-Training-Kit-v1.zip
cd LunarReg-Training-Kit
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Do not start the main run until this prints `True` for CUDA on a GPU machine.

---

## 10. Prepare the image data

Place legally obtained images into these directories:

```text
data/raw/ohrc/
data/raw/tmc2/
data/raw/iirs/
data/raw/lro/
data/raw/selene/
```

Keep original PDS/GeoTIFF products and metadata in a separate archive. The training code accepts PNG, JPEG, TIFF and OpenCV-readable GeoTIFF images.

Start with OHRC and LRO. Do not mix every sensor on the first run.

Create 512×512 tiles:

```bash
python scripts/prepare_tiles.py \
  --input data/raw/ohrc \
  --output data/tiles/ohrc \
  --size 512 \
  --overlap 64 \
  --min-std 0.025

python scripts/prepare_tiles.py \
  --input data/raw/lro \
  --output data/tiles/lro \
  --size 512 \
  --overlap 64 \
  --min-std 0.025
```

`--min-std` rejects nearly blank tiles. Review the output manually and remove corrupted, labelled, border-only or non-overlapping products.

### Prevent data leakage

Split by lunar geography or parent mosaic—not randomly by tiles. Tiles cut from the same crater or mosaic must never appear in both training and testing. Otherwise the reported accuracy will be misleading.

Recommended split:

- 70% geographic regions for training;
- 15% separate regions for validation;
- 15% completely unseen regions for final testing.

---

## 11. Run the smoke test

The smoke run verifies installation, data loading, forward/backward passes and checkpoint saving. It does not create a useful scientific model.

```bash
python scripts/make_demo_data.py
python train.py --config configs/smoke.yaml
```

A successful run creates:

```text
runs/smoke/best.pt
runs/smoke/last.pt
runs/smoke/history.csv
```

Test inference:

```bash
python infer.py \
  --checkpoint runs/smoke/best.pt \
  --source data/demo/source.png \
  --reference data/demo/reference.png \
  --output runs/demo_inference
```

Expected inference outputs:

```text
registered.png
matches.csv
matches_preview.png
metrics.json
homography.json
```

---

## 12. Train using synthetic geometric supervision

The dataset class generates a fresh homography and photometric transformation for every sample. Photometric augmentation includes smooth Perlin illumination, gamma/contrast changes, blur and noise while geometric labels remain known.

RTX 5090:

```bash
python train.py --config configs/train_5090.yaml
```

RTX PRO 5000:

```bash
python train.py --config configs/train_pro5000.yaml
```

Monitor training:

```bash
tensorboard --logdir runs --bind_all
```

Resume after interruption:

```bash
python train.py \
  --config configs/train_5090.yaml \
  --resume runs/lunar_5090/last.pt
```

Never delete `last.pt` while the run is active. Copy `best.pt`, the YAML config and `history.csv` to persistent storage after every important experiment.

---

## 13. Create real OHRC–LRO pseudo-pairs

Synthetic homographies teach geometry, but cross-sensor performance needs real images covering the same lunar location.

Create:

```text
data/manifests/candidates.csv
```

Example:

```csv
pair_id,source_path,reference_path,split
pair_0001,data/tiles/ohrc/ohrc_0001.png,data/tiles/lro/lro_0001.png,train
pair_0002,data/tiles/ohrc/ohrc_0002.png,data/tiles/lro/lro_0002.png,val
```

Generate pseudo-labels with pretrained LoFTR:

```bash
python scripts/pseudo_label_loftr.py \
  --manifest data/manifests/candidates.csv \
  --output data/manifests/pseudo_pairs.csv \
  --preview-dir data/pseudo_previews
```

Open every preview and reject incorrect pairs. Important checks:

- matches cover several parts of the image;
- matches lie on the same physical lunar features;
- homography does not collapse or mirror the image;
- at least 30 MAGSAC inliers survive;
- the pair is not duplicated across splits.

The supplied main configs mix approximately 70% synthetic samples with 30% accepted real pairs. Pseudo-labelled samples receive lower loss weight because their labels are less reliable.

---

## 14. Recommended training sequence

1. Run the installation checker.
2. Complete the smoke test.
3. Overfit approximately 100 synthetic pairs to confirm that the model can learn.
4. Train OHRC-only synthetic pairs for 10–15 epochs.
5. Train LRO-only synthetic pairs for 10–15 epochs.
6. Generate OHRC–LRO pseudo-labels and inspect every preview.
7. Mix 70% synthetic and 30% accepted real pairs for 5–10 additional epochs.
8. Tune the MAGSAC threshold on validation data only.
9. Add TMC-2 after OHRC–LRO results become stable.
10. Add IIRS last because its spectral appearance differs substantially.
11. Freeze settings and evaluate exactly once on geographically independent test scenes.

---

## 15. Acceptance criteria

Before claiming a useful registration model, verify:

- training and validation loss decrease without strong divergence;
- the network can overfit a controlled 100-pair subset;
- held-out synthetic PCK@3 exceeds 90%;
- real pairs retain at least 30 geometric inliers;
- accepted matches are spatially distributed instead of concentrated in one crater;
- median reprojection error is below 1 pixel after local refinement;
- results remain stable across illumination and scale changes;
- no parent scene or geographic region is shared across train and test;
- a naive random-tile split is **not** used—the split must be lunar-geographic.

Sub-pixel accuracy must be calculated from independently verified control points or accurately georeferenced products. A low training loss alone is not evidence of sub-pixel performance.

---

## 16. Training outputs and experiment records

Every serious experiment should preserve:

```text
runs/<experiment>/
├── best.pt
├── last.pt
├── config.yaml
├── history.csv
└── tensorboard/
```

Also record:

- Git commit or ZIP version;
- GPU name and CUDA/PyTorch versions;
- training regions and excluded test regions;
- dataset checksums;
- random seed;
- number of accepted/rejected pseudo-pairs;
- final threshold values;
- validation and test metrics by sensor pair.

This makes results reproducible and defensible during judging.

---

## 17. Adding a trained checkpoint to the website

The current deployed registration endpoint uses SIFT + MAGSAC. After training:

1. Select the best checkpoint using geographic validation—not training loss.
2. Export or copy `best.pt` into `inference-service/models/`.
3. Add a cached model loader in `inference-service/app.py`.
4. Add a neural matching stage before MAGSAC.
5. Keep SIFT as a fallback when the network returns too few distributed matches.
6. Return the same API response fields so the frontend needs no major change.
7. Rebuild and redeploy the Docker service.
8. Update the interface status only after independent benchmark results exist.

Do not overwrite `crater_detector.pt`; it performs a different task.

---

## 18. Troubleshooting

### Render says `render.yaml` was not found

The contents were uploaded one folder too deep. `render.yaml`, `index.html` and `Dockerfile.render` must appear at the GitHub repository root.

### Build spends a long time installing packages

This is expected during the first Docker build because Ultralytics installs PyTorch. Later builds may reuse cached layers if `requirements.render.txt` is unchanged.

### The website opens but Auto mode uses browser baseline

Open `/api/model-status`. On the all-in-one Render deployment it should return `"connected": true`. If it does not, confirm that the repository was deployed as a Docker **Web Service**, not a Static Site.

### Registration reports too few matches

- confirm that both images overlap geographically;
- try similar-scale crops first;
- avoid frames dominated by black borders or labels;
- increase local contrast carefully;
- use images with visible crater texture;
- do not expect a single homography to correct severe terrain parallax.

### Crater inference stops the free Render service

The PyTorch process may exceed the available memory. Upgrade the service RAM or host only the YOLO endpoint on a larger CPU/GPU instance. The browser and SIFT baseline can continue without loading YOLO because the model is loaded lazily.

### GitHub rejects browser upload

Use GitHub Desktop instead of the browser uploader. All included individual files are below GitHub's normal 100 MB per-file limit.

---

## 19. Scientific and licensing notes

- Credit ISRO/ISSDC for Chandrayaan mission imagery.
- Credit NASA/GSFC/Arizona State University for LROC imagery.
- The supplied regional archive includes additional paper citations and source notes.
- The supplied crater dataset is described as 248 LRO NAC tiles originating from Fairweather et al., Zenodo 6386198, under CC BY 4.0.
- Confirm applicable terms before commercial redistribution.
- Do not present crater detection as landing-safety certification.
- Do not infer minerals from generic optical images; calibrated IIRS spectra and a separate validated pipeline are required.

---

## 20. Useful project checks

```bash
# JavaScript syntax
npm run check

# Python syntax
python -m py_compile inference-service/app.py

# Verify the website/API container
docker build -f Dockerfile.render -t lunarreg .
docker run --rm -p 10000:10000 lunarreg
curl http://localhost:10000/api/health
```

For the mathematical derivations, loss functions and detailed training rationale, download:

```text
docs/LunarReg_Model_Training_Handoff.pdf
```
