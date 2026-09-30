# LunarReg — Vercel-ready working prototype

LunarReg is a lunar image-registration proof of concept for Chandrayaan and LRO imagery. The website is deployable to Vercel with no build step. It performs a real translation registration locally in the browser, reports computed metrics, exports a PNG, and contains a proxy for the bundled SIFT + MAGSAC/YOLO inference service.

> Honest status: no learned registration checkpoint is included because the registration model has not been trained. The site never displays fabricated sub-pixel accuracy. The supplied crater model is included with its documented limitations.

## Fastest deployment

### Vercel CLI (no GitHub required)

1. Unzip this package.
2. Install the CLI: `npm install -g vercel`
3. In this folder run: `vercel`
4. Accept the defaults (Framework Preset: **Other**).
5. For production run: `vercel --prod`

The browser registration baseline works immediately. Static download links also work without any environment variables.

### GitHub + Vercel

1. Create an empty GitHub repository.
2. Add this folder using GitHub Desktop (recommended for the bundled binary archives), commit, and publish.
3. In Vercel choose **Add New → Project**, import the repository, select **Other**, and deploy.

If GitHub rejects the repository because of binary policy or organization limits, remove the three large files in `downloads/` before pushing; the site itself and samples will still work. None of the included files exceeds GitHub's normal 100 MB per-file limit.

## Connect full inference

Vercel serverless functions are not suitable for PyTorch/Ultralytics model hosting. Deploy `inference-service/` to a Docker-capable service (Render, RunPod, Fly.io, a VM, etc.), then add this Vercel environment variable:

```text
INFERENCE_API_URL=https://your-api-host.example.com
```

Redeploy the Vercel project. **Auto** mode will then use SIFT + MAGSAC; the crater detector becomes available in the interface. Keep browser uploads below 4 MB when using the Vercel proxy. For larger scientific products, call the inference service directly or use object storage/signed URLs.

### Run the inference service locally

```bash
cd inference-service
docker build -t lunarreg-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=http://localhost:3000 lunarreg-api
```

Then run `vercel dev` at the project root with `INFERENCE_API_URL=http://localhost:8000` in `.env`.

## Included components

| Path | Purpose |
|---|---|
| `index.html`, `assets/` | Responsive frontend and working browser translation baseline |
| `api/` | Vercel health/status endpoints and streaming inference proxy |
| `inference-service/` | FastAPI, OpenCV SIFT/MAGSAC, YOLO crater inference, Dockerfile |
| `docs/LunarReg_Model_Training_Handoff.pdf` | Training mathematics and teammate handoff handbook |
| `downloads/LunarReg-Training-Kit-v1.zip` | Reproducible registration training code |
| `downloads/training_tiles.zip` | 248 supplied labeled LRO NAC tiles |
| `downloads/lunar_crater_detector.zip` | Supplied detector weights and scripts |
| `downloads/CH3_region_69S_32E_dataset.zip` | Supplied Chandrayaan/LRO regional package |
| `samples/` | Included fixed/moving LRO demo pair |

## API contract

- `GET /api/health` — web readiness.
- `GET /api/model-status` — selected compute mode and training status.
- `POST /api/register` — multipart fields `fixed` and `moving`.
- `POST /api/detect-craters` — multipart field `image`.
- Remote service Swagger docs — `/docs` on the FastAPI host.

Registration returns a base64 PNG, a 3×3 homography, correspondence coordinates, RMSE, inlier count, inlier ratio, median/max reprojection error, and spatial coverage.

## Scientific limitations

- The in-browser engine estimates translation only. Its intensity errors are useful for a demo but are not feature reprojection error.
- The OpenCV backend estimates a projective homography. It is a solid classical baseline, not guaranteed sub-pixel performance across extreme illumination, viewpoint, or scale changes.
- The supplied crater detector README reports precision 0.683, recall 0.641, mAP@0.5 0.673 and mAP@0.5:0.95 0.278 on a random LRO tile split. That split may be optimistic and does not establish performance on Chandrayaan sensors.
- IIRS mineral inference and landing-safety scoring are deliberately not fabricated. They require calibrated spectral/terrain products and separate validation.

## Data credits

Credit ISRO/ISSDC for Chandrayaan mission imagery and NASA/GSFC/Arizona State University for LROC imagery. The supplied regional archive contains additional source notes and paper citations. Confirm the applicable terms before public or commercial redistribution. LRO crater tiles are described in the supplied material as originating from Fairweather et al., Zenodo 6386198 (CC BY 4.0).

