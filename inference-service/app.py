"""LunarReg reference inference service.

Classical SIFT + MAGSAC registration is production-usable as a baseline.
The included YOLO weights are the user-supplied LRO crater detector; see README
for its validation metrics and domain-gap warning.
"""

import base64
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("CRATER_MODEL_PATH", APP_DIR / "models" / "crater_detector.pt"))
MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2400"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

app = FastAPI(title="LunarReg Inference API", version="1.0.0")
origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def decode_upload(upload: UploadFile) -> np.ndarray:
    payload = await upload.read(MAX_UPLOAD + 1)
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD // (1024 * 1024)} MB")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Could not decode {upload.filename or 'image'}")
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def png_data(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    if not ok:
        raise HTTPException(500, "Could not encode output image")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def enhance(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "registration": "SIFT + ratio test + USAC_MAGSAC",
        "crater_model_available": MODEL_PATH.exists(),
        "registration_model": "not-trained",
    }


@app.post("/register")
async def register(fixed: UploadFile = File(...), moving: UploadFile = File(...)):
    fixed_bgr, moving_bgr = await decode_upload(fixed), await decode_upload(moving)
    fixed_gray = enhance(cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2GRAY))
    moving_gray = enhance(cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY))

    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.02, edgeThreshold=14)
    fixed_kp, fixed_desc = sift.detectAndCompute(fixed_gray, None)
    moving_kp, moving_desc = sift.detectAndCompute(moving_gray, None)
    if fixed_desc is None or moving_desc is None:
        raise HTTPException(422, "Not enough texture for SIFT descriptors")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(moving_desc, fixed_desc, k=2)
    ratio_matches = [first for first, second in forward if first.distance < 0.76 * second.distance]
    if len(ratio_matches) < 8:
        raise HTTPException(422, f"Only {len(ratio_matches)} candidate matches; use a more overlapping pair")

    moving_points = np.float32([moving_kp[m.queryIdx].pt for m in ratio_matches]).reshape(-1, 1, 2)
    fixed_points = np.float32([fixed_kp[m.trainIdx].pt for m in ratio_matches]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, mask = cv2.findHomography(moving_points, fixed_points, method, 2.0, maxIters=10000, confidence=0.999)
    if homography is None or mask is None:
        raise HTTPException(422, "A stable homography could not be estimated")

    inliers = mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < 6:
        raise HTTPException(422, f"Only {inlier_count} geometric inliers")
    projected = cv2.perspectiveTransform(moving_points[inliers], homography)
    errors = np.linalg.norm(projected[:, 0, :] - fixed_points[inliers, 0, :], axis=1)

    height, width = fixed_bgr.shape[:2]
    registered = cv2.warpPerspective(moving_bgr, homography, (width, height))
    overlay = cv2.addWeighted(fixed_bgr, 0.50, registered, 0.50, 0)
    hull = cv2.convexHull(fixed_points[inliers]).reshape(-1, 2)
    coverage = float(cv2.contourArea(hull.astype(np.float32)) / (width * height)) if len(hull) >= 3 else 0.0

    correspondences = []
    for match, keep, point_a, point_b in zip(ratio_matches, inliers, moving_points[:, 0], fixed_points[:, 0]):
        if keep and len(correspondences) < 500:
            correspondences.append({
                "moving": [round(float(point_a[0]), 3), round(float(point_a[1]), 3)],
                "fixed": [round(float(point_b[0]), 3), round(float(point_b[1]), 3)],
                "descriptor_distance": round(float(match.distance), 3),
            })

    return {
        "registered_image": png_data(overlay),
        "homography": homography.tolist(),
        "metrics": {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "inliers": inlier_count,
            "inlier_ratio": float(inlier_count / len(ratio_matches)),
            "median_error": float(np.median(errors)),
            "max_error": float(np.max(errors)),
            "coverage": max(0.0, min(1.0, coverage)),
        },
        "matches": correspondences,
        "note": "SIFT feature correspondences filtered by Lowe ratio and USAC_MAGSAC. Validate on geographically separated lunar scenes before reporting benchmark accuracy.",
    }


@lru_cache(maxsize=1)
def crater_model():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Crater detector weights are missing")
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


@app.post("/detect-craters")
async def detect_craters(image: UploadFile = File(...), confidence: float = 0.25):
    frame = await decode_upload(image)
    confidence = max(0.05, min(0.95, confidence))
    result = crater_model().predict(frame, conf=confidence, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            boxes.append({
                "xyxy": [round(float(value), 2) for value in xyxy],
                "confidence": round(float(score), 4),
            })
    return {
        "annotated_image": png_data(result.plot()),
        "count": len(boxes),
        "boxes": boxes,
        "note": "Exploratory YOLOv8n detector trained on 248 LRO NAC tiles. Results may not transfer directly to Chandrayaan sensors.",
    }
