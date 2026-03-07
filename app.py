import sys
import os
import traceback
import pickle
from io import BytesIO

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ── Startup diagnostics ───────────────────────────────────────────────────────
print("=== STARTUP ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Files: {os.listdir('.')}", flush=True)

app = FastAPI(
    title="Embryo Quality Classifier & Ranker",
    description="Upload 1 to N embryo images — get ranked results by viability score",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Threshold (lowered to fix class imbalance: 716 Poor vs 124 Good) ─────────
GOOD_THRESHOLD = 0.15

# ── Lazy model loading ────────────────────────────────────────────────────────
_feature_extractor = None
_fusion_model = None
_scaler = None

def get_models():
    global _feature_extractor, _fusion_model, _scaler

    if _fusion_model is None:
        try:
            from tensorflow.keras.models import load_model, Model as KModel

            print("Loading EfficientNet...", flush=True)
            full_model = load_model("efficientnet_embryo_model.h5")
            _feature_extractor = KModel(
                inputs=full_model.input,
                outputs=full_model.layers[-3].output,
            )
            print("Loading fusion model...", flush=True)
            _fusion_model = load_model("dual_branch_embryo_model.keras")

            print("Loading scaler...", flush=True)
            with open("morph_scaler.pkl", "rb") as f:
                _scaler = pickle.load(f)

            print("✅ All models loaded!", flush=True)

        except Exception as e:
            print(f"❌ Model loading failed: {e}", flush=True)
            traceback.print_exc()
            raise RuntimeError(f"Model loading failed: {e}")

    return _feature_extractor, _fusion_model, _scaler


# ── Image processing helpers ──────────────────────────────────────────────────
def decode_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_efficientnet_features(img: np.ndarray, extractor) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    features = extractor.predict(np.expand_dims(img_resized, axis=0), verbose=0)
    return features.flatten()


def extract_morphological_features(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    symmetry_score = 0.0
    if len(centroids) > 1:
        distances = [
            np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            for i in range(len(centroids))
            for j in range(i + 1, len(centroids))
        ]
        symmetry_score = float(np.mean(distances))

    embryo_area = int(np.sum(thresh == 255))
    fragmented_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) < 500)
    fragmentation_ratio = fragmented_area / embryo_area if embryo_area > 0 else 0.0

    return np.array([symmetry_score, fragmentation_ratio])


def analyze_image(img: np.ndarray) -> dict:
    extractor, fusion, scaler = get_models()

    deep_features = extract_efficientnet_features(img, extractor)
    morph_raw = extract_morphological_features(img)
    morph_scaled = scaler.transform([morph_raw])[0]

    combined = np.expand_dims(np.concatenate([deep_features, morph_scaled]), axis=0)
    prediction = fusion.predict(combined, verbose=0)[0]

    good_prob = float(prediction[1])
    poor_prob = float(prediction[0])
    confidence = float(np.max(prediction))
    score = round(good_prob * 100, 2)

    # Use lowered threshold to fix all-poor prediction from class imbalance
    is_good = good_prob >= GOOD_THRESHOLD

    if score >= 80:
        rec = "Highly recommended for transfer"
    elif score >= 60:
        rec = "Suitable for transfer"
    elif score >= GOOD_THRESHOLD * 100:
        rec = "Marginal quality — use only if no better option"
    else:
        rec = "Poor quality — not recommended for transfer"

    return {
        "label": "Good Quality Embryo" if is_good else "Poor Quality Embryo",
        "confidence": round(confidence, 4),
        "viability_score_percent": score,
        "good_probability": round(good_prob, 4),
        "poor_probability": round(poor_prob, 4),
        "symmetry_score": round(float(morph_raw[0]), 4),
        "fragmentation_ratio": round(float(morph_raw[1]), 6),
        "recommendation": rec,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Embryo Quality Classifier & Ranker",
        "usage": {
            "single": "POST /predict  — upload 1 image file",
            "batch":  "POST /rank     — upload N image files, get ranked results",
        },
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    files = ["efficientnet_embryo_model.h5", "dual_branch_embryo_model.keras", "morph_scaler.pkl"]
    return {
        "threshold": GOOD_THRESHOLD,
        "files": {
            f: {
                "exists": os.path.exists(f),
                "size_mb": round(os.path.getsize(f) / 1024 / 1024, 2) if os.path.exists(f) else 0
            }
            for f in files
        }
    }


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got: {file.content_type}")

    img = decode_upload(await file.read())
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Make sure it's a valid JPG/PNG.")

    result = analyze_image(img)
    result["filename"] = file.filename
    return result


@app.post("/rank")
async def rank_embryos(files: list[UploadFile] = File(...)):
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Upload at least 1 image.")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per request.")

    results = []
    for i, file in enumerate(files):
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")

        img = decode_upload(await file.read())
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode '{file.filename}'.")

        analysis = analyze_image(img)
        results.append({"filename": file.filename, "embryo_id": f"E{i+1}", **analysis})

    results.sort(key=lambda x: x["viability_score_percent"], reverse=True)

    return {
        "total_analyzed": len(results),
        "best_embryo": results[0]["filename"],
        "best_viability_score": results[0]["viability_score_percent"],
        "ranked_embryos": [{"rank": i+1, **r} for i, r in enumerate(results)]
    }
