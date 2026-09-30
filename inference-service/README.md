# LunarReg inference service

This optional FastAPI service provides the compute that should not run inside a Vercel function:

- `POST /register`: SIFT descriptors, Lowe ratio filtering, USAC_MAGSAC homography, registered overlay, match coordinates, and measured reprojection metrics.
- `POST /detect-craters`: the supplied YOLOv8n crater detector and structured bounding boxes.
- `GET /health`: readiness and model status.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.render.txt
uvicorn app:app --reload --port 8000
```

Or build the container:

```bash
docker build -t lunarreg-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=https://your-site.vercel.app lunarreg-api
```

Deploy this folder to a Docker-capable GPU/CPU host such as RunPod or Render, then set `INFERENCE_API_URL` in the Vercel project. The API itself works on CPU; GPU acceleration mainly benefits YOLO.

## Crater model limitations

The supplied weights are a one-class YOLOv8n model fine-tuned for 20 CPU epochs at 416 px on 248 labeled LRO NAC tiles. Its supplied README reports precision 0.683, recall 0.641, mAP@0.5 0.673 and mAP@0.5:0.95 0.278 on a random 198/50 split. This split may be optimistic, and performance on Chandrayaan imagery is not established. Do not use crater detections as landing-safety certification.
