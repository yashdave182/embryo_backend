import sys
import os
import traceback
import pickle

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

print("=== STARTUP ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Files: {os.listdir('.')}", flush=True)

app = FastAPI(
    title="Embryo Quality Classifier & Ranker",
    description="Dual-branch embryo viability classifier",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH  = "dual_branch_embryo_model.keras"
SCALER_PATH = "morph_scaler.pkl"

# ── Model loading ─────────────────────────────────────────────────────────────
_model  = None
_scaler = None

def get_model():
    global _model, _scaler
    if _model is None:
        try:
            from tensorflow.keras.models import load_model
            print("Loading dual-branch model...", flush=True)
            _model = load_model(MODEL_PATH, compile=False)
            print(f"✅ Model loaded! Input: {_model.input_shape}", flush=True)
        except Exception as e:
            print(f"❌ Model load failed: {e}", flush=True)
            traceback.print_exc()
            raise RuntimeError(f"Model load failed: {e}")
    if _scaler is None:
        try:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
            print("✅ Scaler loaded!", flush=True)
        except Exception as e:
            print(f"⚠️ Scaler load failed (will skip scaling): {e}", flush=True)
            _scaler = None
    return _model, _scaler


# ── Image helpers ─────────────────────────────────────────────────────────────
def decode_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_morphological_features(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    _, thresh_normal = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, thresh_inv    = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours_normal, _ = cv2.findContours(thresh_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_inv, _    = cv2.findContours(thresh_inv,    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours_inv if len(contours_inv) > len(contours_normal) else contours_normal
    thresh   = thresh_inv   if len(contours_inv) > len(contours_normal) else thresh_normal

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

    embryo_area     = int(np.sum(thresh == 255))
    fragmented_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) < 500)
    fragmentation_ratio = fragmented_area / embryo_area if embryo_area > 0 else 0.0
    cell_count = len([c for c in contours if cv2.contourArea(c) > 200])

    return {
        "symmetry_score":      round(symmetry_score, 4),
        "fragmentation_ratio": round(fragmentation_ratio, 6),
        "cell_count":          cell_count,
    }


def analyze_image(img: np.ndarray, filename: str = "") -> dict:
    model, scaler = get_model()

    # Image branch: resize to 224x224, normalize
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    img_batch   = np.expand_dims(img_resized, axis=0)

    # Morphological branch
    morph = extract_morphological_features(img)
    morph_features = np.array([[
        morph["symmetry_score"],
        morph["fragmentation_ratio"],
        morph["cell_count"]
    ]], dtype=np.float32)

    if scaler is not None:
        morph_features = scaler.transform(morph_features).astype(np.float32)

    # Predict
    try:
        prediction = model.predict([img_batch, morph_features], verbose=0)[0]
    except Exception:
        # Fallback: try single input
        prediction = model.predict(img_batch, verbose=0)[0]

    class_id   = int(np.argmax(prediction))
    good_prob  = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
    poor_prob  = 1.0 - good_prob
    confidence = float(np.max(prediction))
    score      = round(good_prob * 100, 2)

    if score >= 80:
        rec = "Highly recommended for transfer"
    elif score >= 60:
        rec = "Suitable for transfer"
    elif score >= 40:
        rec = "Marginal quality — use only if no better option"
    else:
        rec = "Poor quality — not recommended for transfer"

    return {
        "filename":                filename,
        "label":                   "Good Quality Embryo" if class_id == 1 else "Poor Quality Embryo",
        "confidence":              round(confidence, 4),
        "viability_score_percent": score,
        "good_probability":        round(good_prob, 4),
        "poor_probability":        round(poor_prob, 4),
        "symmetry_score":          morph["symmetry_score"],
        "fragmentation_ratio":     morph["fragmentation_ratio"],
        "cell_count":              morph["cell_count"],
        "recommendation":          rec,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Embryo Quality Classifier & Ranker",
        "version": "4.0.0",
        "endpoints": {
            "POST /predict": "Single embryo image",
            "POST /rank":    "N embryo images → ranked",
        },
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    return {
        "model_file":  MODEL_PATH,
        "scaler_file": SCALER_PATH,
        "model_exists":  os.path.exists(MODEL_PATH),
        "scaler_exists": os.path.exists(SCALER_PATH),
        "model_size_mb": round(os.path.getsize(MODEL_PATH)/1024/1024, 2) if os.path.exists(MODEL_PATH) else 0,
    }


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Must be an image, got: {file.content_type}")
    img = decode_upload(await file.read())
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return analyze_image(img, filename=file.filename)


@app.post("/rank")
async def rank_embryos(files: list[UploadFile] = File(...)):
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Upload at least 1 image.")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 images per request.")

    results = []
    for i, file in enumerate(files):
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"'{file.filename}' is not an image.")
        img = decode_upload(await file.read())
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode '{file.filename}'.")
        results.append({"embryo_id": f"E{i+1}", **analyze_image(img, filename=file.filename)})

    results.sort(key=lambda x: x["viability_score_percent"], reverse=True)

    return {
        "total_analyzed":       len(results),
        "best_embryo":          results[0]["filename"],
        "best_viability_score": results[0]["viability_score_percent"],
        "ranked_embryos":       [{"rank": i+1, **r} for i, r in enumerate(results)]
    }
