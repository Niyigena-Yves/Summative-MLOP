"""Retraining pipeline and trigger.

Pipeline: uploaded files (stored in SQLite + data/uploads) -> preprocessing
(validated, RGB, resized on upload; decoded/resized again in tf.data) ->
retraining that starts from the current model as the pre-trained model ->
evaluation on the test set -> promotion of the new model version.

Triggers:
  * manual  - the "Retrain" button in the UI / POST /retrain
  * auto    - fires when pending uploads reach RETRAIN_THRESHOLD
"""
import os
import threading
import traceback

from src import database as db
from src.preprocessing import CLASS_NAMES

RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", 30))
RETRAIN_EPOCHS = int(os.environ.get("RETRAIN_EPOCHS", 4))

_state = {"running": False, "run_id": None, "last_result": None, "last_error": None}
_lock = threading.Lock()


def status():
    return dict(_state, threshold=RETRAIN_THRESHOLD)


def _run(trigger, run_id, uploads):
    from src.model import retrain
    try:
        paths = [u["path"] for u in uploads]
        labels = [CLASS_NAMES.index(u["label"]) for u in uploads]
        result = retrain(paths, labels, epochs=RETRAIN_EPOCHS, trigger=trigger)
        db.mark_used([u["id"] for u in uploads], run_id)
        db.finish_run(run_id, "completed", result=result)
        _state["last_result"] = result
        _state["last_error"] = None
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        db.finish_run(run_id, "failed", error=str(exc))
        _state["last_error"] = f"{exc}\n{traceback.format_exc(limit=3)}"
    finally:
        _state["running"] = False


def trigger_retraining(trigger="manual", background=True):
    """Start a retraining run on all pending uploads. Returns a status dict."""
    with _lock:
        if _state["running"]:
            return {"started": False, "reason": "A retraining run is already in progress", **status()}
        uploads = [u for u in db.pending_uploads() if os.path.exists(u["path"])]
        if not uploads:
            return {"started": False, "reason": "No new uploaded data to retrain on. Upload images first.",
                    **status()}
        run_id = db.start_run(trigger, len(uploads))
        _state.update(running=True, run_id=run_id)
    if background:
        threading.Thread(target=_run, args=(trigger, run_id, uploads), daemon=True).start()
    else:
        _run(trigger, run_id, uploads)
    return {"started": True, "run_id": run_id, "n_samples": len(uploads), **status()}


def maybe_auto_trigger():
    """Automatic trigger: retrain once enough new labelled data has arrived."""
    if db.upload_summary()["pending_total"] >= RETRAIN_THRESHOLD and not _state["running"]:
        return trigger_retraining(trigger="auto")
    return None
