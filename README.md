# AgriQuest Leaf Detector API

A small FastAPI server that runs the tomato-leaf EfficientNet-B0 model (`cnn_best.pt`,
converted to `model.onnx`) for the AgriQuest app. Built for Render's free plan.

| File | What it is |
|---|---|
| `app.py` | The web app: `GET /health` and `POST /predict` |
| `model.onnx` | The trained model (16 MB), same outputs as `cnn_best.pt` |
| `classes.json` | The 9 class names, in the model's output order (from `data.yaml`) |
| `requirements.txt` | Python packages |

Image preparation matches the training script's evaluation transform:
resize to 320×320 → scale to 0–1 → ImageNet mean/std.

## Deploy on Render (free)

1. Commit and push this `ai-server` folder to GitHub (it lives inside the AgriQuest repo).
2. On render.com: **New → Web Service** → connect the `AgriQuest` repository.
3. Settings:
   - **Root Directory:** `ai-server`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. **Environment variables:** add `PYTHON_VERSION` = `3.13`.
   Optional: add `API_KEY` = any secret; the app must then send it in an `X-API-Key` header.
5. Under **Advanced**, set **Health Check Path** to `/health`.
6. Deploy. Test at `https://<your-service>.onrender.com/docs` → `POST /predict` → *Try it out* → upload a leaf photo.

## Free-plan notes

- The server sleeps after 15 minutes without requests; the first request after that takes about a minute.
  The app should call `/health` when the scanner screen opens and allow ~90 s for the upload.
- Memory use is about 130 MB (limit 512 MB).

## API

`POST /predict` — multipart form, field name `file` (JPEG/PNG, max 10 MB).

```json
{
  "label": "Early Blight",
  "classId": "Tomato_Early_blight",
  "pathogen": "Alternaria solani",
  "confidence": 0.8912,
  "confidenceText": "89%",
  "isHealthy": false,
  "lowConfidence": false,
  "topResults": [{"label": "Early Blight", "confidence": 0.8912}, ...],
  "inferenceMs": 950
}
```

`label` is one of: Bacterial Spot, Early Blight, Late Blight, Septoria Leaf Spot, Healthy Leaf,
Leaf Mold, Spider Mites, Yellow Leaf Curl Virus, Mosaic Virus.

`lowConfidence` is true below 50 %. The model has no "not a leaf" class, so it always picks one
of the nine; use this flag to ask for a clearer photo.

Errors: `400` not an image / empty, `401` wrong API key, `413` file over 10 MB.

## Run locally

```
pip install -r requirements.txt
uvicorn app:app --reload
```
Then open http://localhost:8000/docs
