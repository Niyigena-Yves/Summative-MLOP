"""FastAPI service: prediction, bulk upload, retraining, monitoring and the web UI."""
import os
import socket
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import database as db
from src import retrain as rt
from src.insights import load_insights
from src.model import MODEL_PATH, load_metadata
from src.prediction import get_predictor
from src.preprocessing import CLASS_NAMES, InvalidImageError, save_upload

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
START_TIME = time.time()
STATS = {"requests": 0, "predictions": 0, "errors": 0, "latency_sum_ms": 0.0}
MAX_UPLOAD_MB = 10

app = FastAPI(title="Bean Leaf Disease Classifier",
              description="End-to-end ML pipeline: predict, upload data, retrain, monitor.",
              version="1.0.0")


@app.on_event("startup")
def startup():
    db.init_db()
    db.mark_interrupted(socket.gethostname())
    if os.path.exists(MODEL_PATH):
        get_predictor()._ensure_loaded()  # warm the model so the first request is fast


@app.middleware("http")
async def count_requests(request, call_next):
    STATS["requests"] += 1
    return await call_next(request)


@app.get("/health")
def health():
    """Model up-time / liveness information."""
    meta = load_metadata()
    p = get_predictor()
    return {
        "status": "ok" if p.loaded else "model_not_loaded",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "started_at": START_TIME,
        "container": socket.gethostname(),
        "model_loaded": p.loaded,
        "model_version": meta.get("version"),
        "model_trained_at": meta.get("trained_at"),
        "requests_served": STATS["requests"],
        "predictions_served": STATS["predictions"],
        "avg_prediction_latency_ms": round(STATS["latency_sum_ms"] / STATS["predictions"], 2)
        if STATS["predictions"] else None,
        "errors": STATS["errors"],
        "retraining": rt.status()["running"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of a single bean leaf image."""
    start = time.perf_counter()
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        result = get_predictor().predict(content)
    except InvalidImageError as exc:
        STATS["errors"] += 1
        raise HTTPException(400, str(exc))
    latency = (time.perf_counter() - start) * 1000
    STATS["predictions"] += 1
    STATS["latency_sum_ms"] += latency
    result["latency_ms"] = round(latency, 2)
    result["container"] = socket.gethostname()
    if os.environ.get("LOG_PREDICTIONS", "1") == "1":
        db.log_prediction(result["label"], result["confidence"], latency, result["model_version"])
    return result


@app.post("/upload")
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload labelled images; they are validated, preprocessed and saved to
    data/uploads/<label>/ and recorded in the database for retraining."""
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    saved, rejected = [], []
    for f in files:
        content = await f.read()
        try:
            path = save_upload(content, f.filename, label)
            db.add_upload(f.filename, label, path)
            saved.append(f.filename)
        except (InvalidImageError, ValueError) as exc:
            rejected.append({"file": f.filename, "reason": str(exc)})
    auto = rt.maybe_auto_trigger()
    return {"saved": len(saved), "rejected": rejected, "label": label,
            "auto_retrain_triggered": bool(auto and auto.get("started")),
            "summary": db.upload_summary()}


@app.get("/uploads")
def uploads():
    return db.upload_summary()


@app.post("/retrain")
def retrain():
    """Trigger retraining on all pending uploaded data (runs in the background)."""
    return rt.trigger_retraining(trigger="manual")


@app.get("/retrain/status")
def retrain_status():
    return {**rt.status(), "runs": db.list_runs(), "pending": db.upload_summary()["pending_total"]}


@app.get("/model")
def model_info():
    """Model metadata and evaluation metrics on the held-out test set."""
    meta = load_metadata()
    if not meta:
        raise HTTPException(404, "No trained model found")
    return meta


@app.get("/insights")
def insights():
    """Precomputed image feature statistics for the visualisations."""
    return load_insights()


@app.get("/stats")
def stats():
    return {"live": STATS, "predictions": db.prediction_stats()}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
