"""
AgriQuest Leaf Detector API (runs on Render).

POST /predict   multipart form with an image file in the field "file"
GET  /health    used to wake the server up and by Render's health check
GET  /reliability          k-fold + Cohen's kappa results (reliability.json)
GET  /reference?label=...  example photos of a disease (reference/<classId>/1.jpg ...)

The model is the EfficientNet-B0 from classify_leaf_pipeline.py (best.pt),
converted to ONNX so it fits in Render's free 512 MB instance.
Pre-processing: pad the photo to a square (never stretch it), resize to 320 x 320,
then normalize with the ImageNet mean and std. The model also returns class activation
maps (CAM, the Grad-CAM shortcut for global pooling + one linear layer) for the heatmap.
"""
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

HERE = Path(__file__).parent
IMG_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_COLOR = (124, 116, 104)                    # gray fill used when padding, same as in training
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

app = FastAPI(title="AgriQuest Leaf Detector", version="1.1")

# ---- model reliability and example photos (read-only, no secrets) ----
REFERENCE_DIR = HERE / "reference"
RELIABILITY_FILE = HERE / "reliability.json"
LABEL_TO_CLASS = {label: cid for cid, (label, _) in DISPLAY.items()}
if REFERENCE_DIR.is_dir():
    app.mount("/reference-files", StaticFiles(directory=str(REFERENCE_DIR)), name="reference-files")


def preprocess(raw: bytes):
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)        # phone photos are often stored sideways
    img = img.convert("RGB")
    size = img.size                           # original (width, height), needed to place the heatmap
    img = ImageOps.pad(img, (IMG_SIZE, IMG_SIZE), color=PAD_COLOR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None], size   # (1, 3, 320, 320)


def heatmap_png(cam: np.ndarray, size) -> str:
    """cam: (h, w) map of the predicted class. Returns a base64 PNG overlay with the photo's own
    shape, at most 480 px on the long side (the gray padding is cut off; the app stretches it over the photo), transparent where the model paid little attention."""
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    big = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    w, h = size
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    cw, ch = round(w * scale), round(h * scale)
    left, top = (IMG_SIZE - cw) // 2, (IMG_SIZE - ch) // 2
    # keep the overlay small (long side <= 480 px, same shape as the photo): a full-size overlay of a
    # 12 MP phone photo needs hundreds of MB and crashes Render's free 512 MB instance
    k = min(1.0, 480 / max(w, h))
    big = big.crop((left, top, left + cw, top + ch)).resize((max(1, round(w * k)), max(1, round(h * k))), Image.BICUBIC)
    v = np.asarray(big, dtype=np.float32) / 255.0
    # simple blue -> green -> yellow -> red ramp
    r = np.clip(4 * v - 2, 0, 1); g = np.clip(np.minimum(4 * v, 4 - 4 * v), 0, 1); b = np.clip(2 - 4 * v, 0, 1)
    a = np.clip(v * 1.6, 0, 1) * 0.65
    rgba = (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


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
        x, size = preprocess(raw)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image")

    t0 = time.perf_counter()
    logits, cams = session.run(None, {INPUT_NAME: x})
    logits, cams = logits[0], cams[0]
    ms = round((time.perf_counter() - t0) * 1000)
    prob = softmax(logits)

    order = np.argsort(prob)[::-1]
    best = CLASSES[int(order[0])]
    label, pathogen = DISPLAY.get(best, (best, None))
    conf = float(prob[order[0]])
    heatmap = heatmap_png(cams[int(order[0])], size)
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
        "heatmapPng": heatmap,                # base64 PNG overlay, same shape as the photo, max 480 px
    }


@app.get("/reliability")
def reliability():
    """The 5-fold cross-validation and Cohen's kappa results shown in the app."""
    if not RELIABILITY_FILE.exists():
        raise HTTPException(status_code=404, detail="No reliability results on this server")
    return json.loads(RELIABILITY_FILE.read_text())


@app.get("/reference")
def reference(label: str):
    """Example photos of one disease, e.g. /reference?label=Early Blight (labels as /predict returns them)."""
    class_id = LABEL_TO_CLASS.get(label)
    folder = REFERENCE_DIR / class_id if class_id else None
    if folder is None or not folder.is_dir():
        raise HTTPException(status_code=404, detail="No example photos for that label")
    files = sorted(folder.glob("*.jpg"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0)
    return {
        "label": label,
        "classId": class_id,
        "photos": [f"/reference-files/{class_id}/{f.name}" for f in files],
    }
