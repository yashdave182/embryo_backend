from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import requests
import pickle
from tensorflow.keras.models import load_model, Model

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

# ── Load models on startup ───────────────────────────────────────────────────
print("Loading models...")
full_model = load_model("efficientnet_embryo_model.h5")
efficientnet_feature_extractor = Model(
    inputs=full_model.input,
    outputs=full_model.layers[-3].output,
)
fusion_model = load_model("dual_branch_embryo_model.keras")
with open("morph_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("All models loaded!")


# ── Helper functions ─────────────────────────────────────────────────────────
def download_image(url: str):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Download error: {e}")
        return None


def extract_efficientnet_features(img: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    features = efficientnet_feature_extractor.predict(
        np.expand_dims(img_resized, axis=0), verbose=0
    )
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
    """Run full pipeline on one image, return raw scores."""
    deep_features = extract_efficientnet_features(img)
    morph_raw = extract_morphological_features(img)
    morph_scaled = scaler.transform([morph_raw])[0]

    combined = np.expand_dims(np.concatenate([deep_features, morph_scaled]), axis=0)
    prediction = fusion_model.predict(combined, verbose=0)[0]  # shape: (2,)

    class_id = int(np.argmax(prediction))
    good_prob = float(prediction[1])   # probability of being Good quality
    poor_prob = float(prediction[0])   # probability of being Poor quality

    return {
        "class_id": class_id,
        "label": "Good Quality Embryo" if class_id == 1 else "Poor Quality Embryo",
        "confidence": round(float(np.max(prediction)), 4),
        "viability_score_percent": round(good_prob * 100, 2),  # always "good" probability as score
        "good_probability": round(good_prob, 4),
        "poor_probability": round(poor_prob, 4),
        "symmetry_score": round(float(morph_raw[0]), 4),
        "fragmentation_ratio": round(float(morph_raw[1]), 6),
    }


# ── Schemas ───────────────────────────────────────────────────────────────────
class SingleRequest(BaseModel):
    image_url: str

class RankRequest(BaseModel):
    embryos: list[dict]  # each: {"id": "E1", "image_url": "https://..."}

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
        "endpoints": {
            "POST /predict": "Analyze a single embryo image",
            "POST /rank":    "Rank multiple embryos by viability score",
        },
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_single(request: SingleRequest):
    """Analyze one embryo image from a URL."""
    img = download_image(request.image_url)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not download image.")
    return analyze_single_image(img)


@app.post("/rank", response_model=RankResponse)
def rank_embryos(request: RankRequest):
    """
    Rank multiple embryos from a list of image URLs.

    Request body example:
    {
      "embryos": [
        {"id": "E1", "image_url": "https://..."},
        {"id": "E2", "image_url": "https://..."},
        {"id": "E3", "image_url": "https://..."}
      ]
    }

    Returns all embryos ranked #1 (best viability) to #N (worst).
    Each embryo gets a viability_score_percent, label, and transfer recommendation.
    """
    if len(request.embryos) < 1:
        raise HTTPException(status_code=400, detail="Provide at least 1 embryo.")
    if len(request.embryos) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 embryos per request.")

    results = []
    for i, embryo in enumerate(request.embryos):
        embryo_id = embryo.get("id") or f"Embryo_{i+1}"
        image_url = embryo.get("image_url")

        if not image_url:
            raise HTTPException(status_code=400, detail=f"Missing image_url for '{embryo_id}'.")

        img = download_image(image_url)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not download image for '{embryo_id}'.")

        analysis = analyze_single_image(img)
        results.append({"id": embryo_id, **analysis})

    # Sort best → worst by viability score
    results.sort(key=lambda x: x["viability_score_percent"], reverse=True)

    ranked = []
    for rank_pos, r in enumerate(results, start=1):
        score = r["viability_score_percent"]

        if score >= 80:
            rec = "Highly recommended for transfer"
        elif score >= 60:
            rec = "Suitable for transfer"
        elif score >= 40:
            rec = "Marginal quality — use only if no better option available"
        else:
            rec = "Poor quality — not recommended for transfer"

        ranked.append(EmbryoResult(
            rank=rank_pos,
            id=r["id"],
            label=r["label"],
            viability_score_percent=r["viability_score_percent"],
            confidence=r["confidence"],
            good_probability=r["good_probability"],
            poor_probability=r["poor_probability"],
            symmetry_score=r["symmetry_score"],
            fragmentation_ratio=r["fragmentation_ratio"],
            recommendation=rec,
        ))

    return RankResponse(
        total_analyzed=len(ranked),
        best_embryo_id=ranked[0].id,
        ranked_embryos=ranked,
    )
