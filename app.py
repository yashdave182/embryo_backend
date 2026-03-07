import sys
import os
import traceback

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

print("=== STARTUP ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Files: {os.listdir('.')}", flush=True)

app = FastAPI(
    title="Embryo Quality Classifier & Ranker",
    description="Upload 1 to N embryo images — ranked by EfficientNetB3 viability score",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH    = "efficientnet_embryo_model.keras"
GDRIVE_FILE_ID = "1MnT06A1C2KhHMqmtQeyfhf7_xrWhC1Hy"
GDRIVE_URL     = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# ── Download model from Google Drive if not present ───────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"✅ Model already present ({size_mb:.1f} MB), skipping download.", flush=True)
        return

    print("📥 Downloading model from Google Drive...", flush=True)
    try:
        import requests

        # Google Drive large-file download needs a confirmation token
        session = requests.Session()
        response = session.get(GDRIVE_URL, stream=True, timeout=120)

        # Check if Drive is asking for virus-scan confirmation
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            print("🔄 Large file — fetching with confirmation token...", flush=True)
            response = session.get(
                GDRIVE_URL,
                params={"confirm": token},
                stream=True,
                timeout=300
            )

        # Write to disk in chunks
        total = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

        size_mb = total / 1024 / 1024
        print(f"✅ Download complete: {size_mb:.1f} MB saved to {MODEL_PATH}", flush=True)

    except Exception as e:
        print(f"❌ Download failed: {e}", flush=True)
        traceback.print_exc()
        raise RuntimeError(f"Could not download model: {e}")


# Run download immediately at startup
download_model()


# ── Lazy model loading ────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from tensorflow.keras.models import load_model
            print("Loading EfficientNet model...", flush=True)
            _model = load_model(MODEL_PATH)
            print(f"✅ Model loaded! Input: {_model.input_shape}  Output: {_model.output_shape}", flush=True)
        except Exception as e:
            print(f"❌ Model loading failed: {e}", flush=True)
            traceback.print_exc()
            raise RuntimeError(f"Model loading failed: {e}")
    return _model


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
    model = get_model()

    # EfficientNetB3 expects 260×260, raw pixels [0-255] (no rescale)
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (260, 260)).astype(np.float32)
    img_batch   = np.expand_dims(img_resized, axis=0)   # shape: (1, 260, 260, 3)

    prediction = model.predict(img_batch, verbose=0)[0]  # shape: (2,)

    class_id   = int(np.argmax(prediction))
    good_prob  = float(prediction[1])   # class 1 = Good
    poor_prob  = float(prediction[0])   # class 0 = Poor
    confidence = float(np.max(prediction))
    score      = round(good_prob * 100, 2)

    morph = extract_morphological_features(img)

    if score >= 80:
        rec = "Highly recommended for transfer"
    elif score >= 60:
        rec = "Suitable for transfer"
    elif score >= 40:
        rec = "Marginal quality — use only if no better option"
    else:
        rec = "Poor quality — not recommended for transfer"

    return {
        "filename":              filename,
        "label":                 "Good Quality Embryo" if class_id == 1 else "Poor Quality Embryo",
        "confidence":            round(confidence, 4),
        "viability_score_percent": score,
        "good_probability":      round(good_prob, 4),
        "poor_probability":      round(poor_prob, 4),
        "symmetry_score":        morph["symmetry_score"],
        "fragmentation_ratio":   morph["fragmentation_ratio"],
        "cell_count":            morph["cell_count"],
        "recommendation":        rec,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Embryo Quality Classifier & Ranker — EfficientNetB3",
        "version": "5.0.0",
        "endpoints": {
            "POST /predict": "Single embryo image upload",
            "POST /rank":    "N embryo images → ranked by viability",
        },
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    exists = os.path.exists(MODEL_PATH)
    return {
        "model_file":  MODEL_PATH,
        "exists":      exists,
        "size_mb":     round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2) if exists else 0,
        "gdrive_id":   GDRIVE_FILE_ID,
    }


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """Upload a single embryo image → get viability analysis."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Must be an image, got: {file.content_type}")
    img = decode_upload(await file.read())
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPG/PNG.")
    return analyze_image(img, filename=file.filename)


@app.post("/rank")
async def rank_embryos(files: list[UploadFile] = File(...)):
    """Upload N embryo images → ranked list best to worst by viability score."""
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
        "total_analyzed":     len(results),
        "best_embryo":        results[0]["filename"],
        "best_viability_score": results[0]["viability_score_percent"],
        "ranked_embryos":     [{"rank": i+1, **r} for i, r in enumerate(results)]
    }
