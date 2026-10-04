"""Web UI for the ISL recogniser.

Serves the browser client and wraps the same RandomForest model that src/main.py uses.
The browser runs MediaPipe Hands and sends the 21 (x, y) landmarks of one hand; nothing
else leaves the page.

    pip install -r requirements.txt
    python web/server.py            # http://localhost:8000
"""
import os
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("ISL_MODEL", os.path.join(ROOT, "models", "isl_model.p"))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
CLASSES = [str(c) for c in model.classes_]

app = FastAPI(title="ISL Recognition", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Landmarks(BaseModel):
    # 21 hand landmarks flattened as x0, y0, x1, y1, ... exactly like src/capture.py
    landmarks: list[float] = Field(..., min_length=42, max_length=42)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES, "model": type(model).__name__}


@app.post("/predict")
def predict(body: Landmarks):
    x = np.asarray([body.landmarks], dtype=np.float64)
    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="landmarks must be finite numbers")
    probs = model.predict_proba(x)[0]
    order = np.argsort(probs)[::-1][:3]
    return {
        "label": CLASSES[order[0]],
        "confidence": round(float(probs[order[0]]), 4),
        "top": [{"label": CLASSES[i], "confidence": round(float(probs[i]), 4)} for i in order],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
