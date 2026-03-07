import sys
import os
import traceback

# ── Startup diagnostics — printed before anything else ──────────────────────
print("=== STARTUP DIAGNOSTICS ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)
print(f"Files present: {os.listdir('.')}", flush=True)

# Check each import individually so we know exactly what's missing
for pkg in ["fastapi", "uvicorn", "numpy", "cv2", "tensorflow", "sklearn", "pickle"]:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}", flush=True)
    except ImportError as e:
        print(f"  ❌ {pkg} — {e}", flush=True)

print("=== END DIAGNOSTICS ===", flush=True)

# ── Main app ─────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import requests
import pickle

app = FastAPI(
    title="Embryo Quality Classifier & Ranker API",
    description="Classify and rank multiple embryos by viability score",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ── Helper functions ──────────────────────────────────────────────────────────
def download_image(url: str):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Download error: {e}", flush=True)
        return None


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


def analyze_single_image(img: np.ndarray) -> dict:
    extractor, fusion, scaler = get_models()
    deep_features = extract_efficientnet_features(img, extractor)
    morph_raw = extract_morphological_features(img)
    morph_scaled = scaler.transform([morph_raw])[0]
    combined = np.expand_dims(np.concatenate([deep_features, morph_scaled]), axis=0)
    prediction = fusion.predict(combined, verbose=0)[0]
    class_id = int(np.argmax(prediction))
    good_prob = float(prediction[1])
    poor_prob = float(prediction[0])
    return {
        "class_id": class_id,
        "label": "Good Quality Embryo" if class_id == 1 else "Poor Quality Embryo",
        "confidence": round(float(np.max(prediction)), 4),
        "viability_score_percent": round(good_prob * 100, 2),
        "good_probability": round(good_prob, 4),
        "poor_probability": round(poor_prob, 4),
        "symmetry_score": round(float(morph_raw[0]), 4),
        "fragmentation_ratio": round(float(morph_raw[1]), 6),
    }


# ── Schemas ───────────────────────────────────────────────────────────────────
class SingleRequest(BaseModel):
    image_url: str

class RankRequest(BaseModel):
    embryos: list[dict]

class EmbryoResult(BaseModel):
    rank: int
    id: str
    label: str
    viability_score_percent: float
    confidence: float
    good_probability: float
    poor_probability: float
    symmetry_score: float
    fragmentation_ratio: float
    recommendation: str

class RankResponse(BaseModel):
    total_analyzed: int
    best_embryo_id: str
    ranked_embryos: list[EmbryoResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Embryo Quality Classifier & Ranker",
        "endpoints": {"POST /predict": "Single embryo", "POST /rank": "Rank multiple"},
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    """Check model files and imports."""
    files = ["efficientnet_embryo_model.h5", "dual_branch_embryo_model.keras", "morph_scaler.pkl"]
    file_status = {
        f: {"exists": os.path.exists(f), "size_mb": round(os.path.getsize(f)/1024/1024, 2) if os.path.exists(f) else 0}
        for f in files
    }
    import_status = {}
    for pkg in ["fastapi", "numpy", "cv2", "tensorflow", "sklearn"]:
        try:
            __import__(pkg)
            import_status[pkg] = "ok"
        except ImportError as e:
            import_status[pkg] = str(e)

    return {"files": file_status, "imports": import_status}

@app.post("/predict")
def predict_single(request: SingleRequest):
    img = download_image(request.image_url)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not download or decode image.")
    return analyze_single_image(img)

@app.post("/rank", response_model=RankResponse)
def rank_embryos(request: RankRequest):
    if not 1 <= len(request.embryos) <= 20:
        raise HTTPException(status_code=400, detail="Provide 1–20 embryos.")

    results = []
    for i, embryo in enumerate(request.embryos):
        embryo_id = embryo.get("id") or f"Embryo_{i+1}"
        image_url = embryo.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail=f"Missing image_url for '{embryo_id}'.")
        img = download_image(image_url)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not download image for '{embryo_id}'.")
        results.append({"id": embryo_id, **analyze_single_image(img)})

    results.sort(key=lambda x: x["viability_score_percent"], reverse=True)

    ranked = []
    for rank_pos, r in enumerate(results, start=1):
        score = r["viability_score_percent"]
        if score >= 80:
            rec = "Highly recommended for transfer"
        elif score >= 60:
            rec = "Suitable for transfer"
        elif score >= 40:
            rec = "Marginal quality — use only if no better option"
        else:
            rec = "Poor quality — not recommended for transfer"

        ranked.append(EmbryoResult(
            rank=rank_pos, id=r["id"], label=r["label"],
            viability_score_percent=r["viability_score_percent"],
            confidence=r["confidence"], good_probability=r["good_probability"],
            poor_probability=r["poor_probability"], symmetry_score=r["symmetry_score"],
            fragmentation_ratio=r["fragmentation_ratio"], recommendation=rec,
        ))

    return RankResponse(total_analyzed=len(ranked), best_embryo_id=ranked[0].id, ranked_embryos=ranked)
