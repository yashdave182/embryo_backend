import sys
import os
import traceback
import pickle
import json

import numpy as np
import cv2
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

print("=== STARTUP ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Files: {os.listdir('.')}", flush=True)

app = FastAPI(
    title="Embryo Quality Classifier & Ranker",
    description="Upload 1 to N embryo images — ranked results with Groq AI explanations",
    version="4.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GOOD_THRESHOLD = 0.15
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = "llama3-8b-8192"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

_feature_extractor = None
_fusion_model      = None
_scaler            = None

def get_models():
    global _feature_extractor, _fusion_model, _scaler
    if _fusion_model is None:
        try:
            from tensorflow.keras.models import load_model, Model as KModel
            print("Loading EfficientNet...", flush=True)
            full_model = load_model("efficientnet_embryo_model.h5")
            _feature_extractor = KModel(inputs=full_model.input, outputs=full_model.layers[-3].output)
            print("Loading fusion model...", flush=True)
            _fusion_model = load_model("dual_branch_embryo_model.keras")
            print("Loading scaler...", flush=True)
            with open("morph_scaler.pkl", "rb") as f:
                _scaler = pickle.load(f)
            print("All models loaded!", flush=True)
        except Exception as e:
            print(f"Model loading failed: {e}", flush=True)
            traceback.print_exc()
            raise RuntimeError(f"Model loading failed: {e}")
    return _feature_extractor, _fusion_model, _scaler


def decode_upload(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_efficientnet_features(img, extractor):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    features = extractor.predict(np.expand_dims(img_resized, axis=0), verbose=0)
    return features.flatten()


def extract_morphological_features(img):
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
        distances = [np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
                     for i in range(len(centroids)) for j in range(i+1, len(centroids))]
        symmetry_score = float(np.mean(distances))

    embryo_area = int(np.sum(thresh == 255))
    fragmented_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) < 500)
    fragmentation_ratio = fragmented_area / embryo_area if embryo_area > 0 else 0.0
    return np.array([symmetry_score, fragmentation_ratio])


def get_groq_analysis(model_result):
    if not GROQ_API_KEY:
        return {
            "ai_label": model_result["label"],
            "ai_explanation": "AI explanation unavailable — no GROQ_API_KEY configured.",
            "ai_clinical_note": "",
            "ai_overridden": False,
        }

    prompt = f"""You are an expert IVF embryologist reviewing an AI model's embryo analysis.

Model metrics:
- Viability score: {model_result['viability_score_percent']}%
- Good probability: {model_result['good_probability']}
- Poor probability: {model_result['poor_probability']}
- Model confidence: {model_result['confidence']}
- Symmetry score: {model_result['symmetry_score']} (higher = more asymmetric cells)
- Fragmentation ratio: {model_result['fragmentation_ratio']} (higher = more fragmented)
- Model label: {model_result['label']}

NOTE: Model was trained on imbalanced data (716 Poor vs 124 Good), threshold lowered to 0.15 to compensate. It tends to under-predict Good embryos.

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{"final_label": "Good Quality Embryo" or "Poor Quality Embryo", "overridden": true or false, "explanation": "2-3 sentence plain English explanation of what these metrics indicate about embryo quality", "clinical_note": "1 sentence clinical recommendation for the embryologist"}}

Override the model only if metrics strongly contradict its label."""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 300},
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        return {
            "ai_label":         parsed.get("final_label", model_result["label"]),
            "ai_explanation":   parsed.get("explanation", ""),
            "ai_clinical_note": parsed.get("clinical_note", ""),
            "ai_overridden":    parsed.get("overridden", False),
        }
    except Exception as e:
        print(f"Groq call failed: {e}", flush=True)
        return {
            "ai_label":         model_result["label"],
            "ai_explanation":   "AI explanation temporarily unavailable.",
            "ai_clinical_note": "",
            "ai_overridden":    False,
        }


def analyze_image(img):
    extractor, fusion, scaler = get_models()

    deep_features = extract_efficientnet_features(img, extractor)
    morph_raw     = extract_morphological_features(img)
    morph_scaled  = scaler.transform([morph_raw])[0]

    combined   = np.expand_dims(np.concatenate([deep_features, morph_scaled]), axis=0)
    prediction = fusion.predict(combined, verbose=0)[0]

    good_prob  = float(prediction[1])
    poor_prob  = float(prediction[0])
    confidence = float(np.max(prediction))
    score      = round(good_prob * 100, 2)
    is_good    = good_prob >= GOOD_THRESHOLD

    if score >= 80:   rec = "Highly recommended for transfer"
    elif score >= 60: rec = "Suitable for transfer"
    elif score >= GOOD_THRESHOLD * 100: rec = "Marginal quality — use only if no better option"
    else:             rec = "Poor quality — not recommended for transfer"

    model_result = {
        "label":                   "Good Quality Embryo" if is_good else "Poor Quality Embryo",
        "confidence":              round(confidence, 4),
        "viability_score_percent": score,
        "good_probability":        round(good_prob, 4),
        "poor_probability":        round(poor_prob, 4),
        "symmetry_score":          round(float(morph_raw[0]), 4),
        "fragmentation_ratio":     round(float(morph_raw[1]), 6),
        "recommendation":          rec,
    }

    ai = get_groq_analysis(model_result)

    return {
        **model_result,
        "label":            ai["ai_label"],          # final label = AI override or model
        "model_label":      model_result["label"],   # original model label always preserved
        "ai_overridden":    ai["ai_overridden"],
        "ai_explanation":   ai["ai_explanation"],
        "ai_clinical_note": ai["ai_clinical_note"],
    }


@app.get("/")
def root():
    return {"message": "Embryo Quality Classifier & Ranker", "version": "4.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "groq": "enabled" if GROQ_API_KEY else "disabled"}

@app.get("/debug")
def debug():
    files = ["efficientnet_embryo_model.h5", "dual_branch_embryo_model.keras", "morph_scaler.pkl"]
    return {
        "threshold": GOOD_THRESHOLD,
        "groq_enabled": bool(GROQ_API_KEY),
        "files": {f: {"exists": os.path.exists(f),
                      "size_mb": round(os.path.getsize(f)/1024/1024, 2) if os.path.exists(f) else 0}
                  for f in files}
    }

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Must be an image, got: {file.content_type}")
    img = decode_upload(await file.read())
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
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
            raise HTTPException(status_code=400, detail=f"'{file.filename}' is not an image.")
        img = decode_upload(await file.read())
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode '{file.filename}'.")
        analysis = analyze_image(img)
        results.append({"filename": file.filename, "embryo_id": f"E{i+1}", **analysis})
    results.sort(key=lambda x: x["viability_score_percent"], reverse=True)
    return {
        "total_analyzed":       len(results),
        "best_embryo":          results[0]["filename"],
        "best_viability_score": results[0]["viability_score_percent"],
        "ranked_embryos":       [{"rank": i+1, **r} for i, r in enumerate(results)]
    }


# ── Groq AI Insights ──────────────────────────────────────────────────────────
from pydantic import BaseModel

class InsightRequest(BaseModel):
    label: str
    viability_score_percent: float
    good_probability: float
    confidence: float
    symmetry_score: float
    fragmentation_ratio: float
    cell_count: int = 0
    recommendation: str
    rank: int = 1
    total_embryos: int = 1
    groq_api_key: str

@app.post("/insights")
async def get_insights(req: InsightRequest):
    """Call Groq LLM to generate clinical AI insights for an embryo result."""
    try:
        prompt = f"""Embryo analysis results:
- Label: {req.label}
- Viability Score: {req.viability_score_percent}%
- Good Probability: {round(req.good_probability * 100, 1)}%
- Confidence: {round(req.confidence * 100, 1)}%
- Symmetry Score: {req.symmetry_score}
- Fragmentation Ratio: {round(req.fragmentation_ratio * 100, 2)}%
- Cell Count: {req.cell_count}
- Rank: {req.rank} of {req.total_embryos}
- Recommendation: {req.recommendation}

Provide a 2-3 sentence clinical interpretation for the embryologist. Be specific about the numbers. No markdown, no bullet points, plain text only."""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {req.groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "max_tokens": 180,
                "temperature": 0.4,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert IVF embryologist AI assistant. Give concise, clinical insights about embryo quality based on metrics. Always respond in 2-3 sentences max. Be specific about the numbers. Do not give generic advice. Format: plain text only, no markdown, no bullet points."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=15
        )

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq API error: {response.text}")

        data = response.json()
        insight = data["choices"][0]["message"]["content"].strip()
        return {"insight": insight}

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Groq API timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")
