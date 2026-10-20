"""
AgriQuest Leaf Detector API (runs on Render).

POST /predict   multipart form with an image file in the field "file"
GET  /health    used to wake the server up and by Render's health check

The model is the EfficientNet-B0 from classify_leaf_pipeline.py (best.pt),
converted to ONNX so it fits in Render's free 512 MB instance.
Pre-processing matches the script's evaluation transform exactly:
    Resize((320, 320)) -> ToTensor() -> Normalize(ImageNet mean, std)
"""
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

HERE = Path(__file__).parent
IMG_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024          # 10 MB
LOW_CONFIDENCE = 0.50                        # below this the app should suggest a retake
API_KEY = os.environ.get("API_KEY")          # optional: set it in Render to block strangers

CLASSES = json.loads((HERE / "classes.json").read_text())

# Friendly names the Android app shows. "Healthy Leaf", "Early Blight" and
# "Bacterial Spot" match what ScanPresentation.kt already looks for.
DISPLAY = {
    "Tomato-Bacterial-spot": ("Bacterial Spot", "Xanthomonas spp."),
    "Tomato_Early_blight": ("Early Blight", "Alternaria solani"),
    "Tomato_Late_blight": ("Late Blight", "Phytophthora infestans"),
    "Tomato_Septoria_Leaf_Spot": ("Septoria Leaf Spot", "Septoria lycopersici"),
    "Tomato__Healthy": ("Healthy Leaf", None),
    "Tomato___Leaf_Mold": ("Leaf Mold", "Passalora fulva"),
    "Tomato___Spider_mites-Two-spotted_spider_mite": ("Spider Mites", "Tetranychus urticae"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Yellow Leaf Curl Virus", "TYLCV"),
    "Tomato___Tomato_mosaic_virus": ("Mosaic Virus", "ToMV"),
}
HEALTHY = "Tomato__Healthy"

# 0.1 CPU on the free plan: one thread avoids wasted context switching.
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
session = ort.InferenceSession(str(HERE / "model.onnx"), opts, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

app = FastAPI(title="AgriQuest Leaf Detector", version="1.0")


def preprocess(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)        # phone photos are often stored sideways
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]         # (1, 3, 320, 320)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model": "efficientnet_b0", "classes": len(CLASSES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB")
    try:
        x = preprocess(raw)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image")

    t0 = time.perf_counter()
    logits = session.run(None, {INPUT_NAME: x})[0][0]
    ms = round((time.perf_counter() - t0) * 1000)
    prob = softmax(logits)

    order = np.argsort(prob)[::-1]
    best = CLASSES[int(order[0])]
    label, pathogen = DISPLAY.get(best, (best, None))
    conf = float(prob[order[0]])
    return {
        "label": label,                       # e.g. "Early Blight"
        "classId": best,                      # raw name from data.yaml
        "pathogen": pathogen,
        "confidence": round(conf, 4),         # 0..1
        "confidenceText": f"{round(conf * 100)}%",
        "isHealthy": best == HEALTHY,
        "lowConfidence": conf < LOW_CONFIDENCE,
        "topResults": [
            {"label": DISPLAY.get(CLASSES[int(i)], (CLASSES[int(i)],))[0],
             "confidence": round(float(prob[i]), 4)}
            for i in order[:3]
        ],
        "inferenceMs": ms,
    }
