"""SQLite storage for uploaded training data, retraining runs and prediction logs."""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from src.preprocessing import DATA_DIR

DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "mlops.db"))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    used_in_training INTEGER DEFAULT 0,
    training_run_id INTEGER
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_samples INTEGER,
    host TEXT,
    result TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    latency_ms REAL,
    model_version INTEGER
);
"""


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _lock:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db():
    with connect() as c:
        c.executescript(SCHEMA)
        cols = [r["name"] for r in c.execute("PRAGMA table_info(training_runs)")]
        if "host" not in cols:
            c.execute("ALTER TABLE training_runs ADD COLUMN host TEXT")


def add_upload(filename, label, path):
    with connect() as c:
        cur = c.execute("INSERT INTO uploads (filename, label, path, uploaded_at) VALUES (?,?,?,?)",
                        (filename, label, path, now()))
        return cur.lastrowid


def pending_uploads():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM uploads WHERE used_in_training=0")]


def upload_summary():
    with connect() as c:
        rows = c.execute("SELECT label, used_in_training, COUNT(*) n FROM uploads "
                         "GROUP BY label, used_in_training").fetchall()
        recent = c.execute("SELECT filename, label, uploaded_at, used_in_training FROM uploads "
                           "ORDER BY id DESC LIMIT 15").fetchall()
    summary = {"pending": {}, "used": {}, "recent": [dict(r) for r in recent]}
    for r in rows:
        summary["used" if r["used_in_training"] else "pending"][r["label"]] = r["n"]
    summary["pending_total"] = sum(summary["pending"].values())
    return summary


def mark_used(upload_ids, run_id):
    with connect() as c:
        c.executemany("UPDATE uploads SET used_in_training=1, training_run_id=? WHERE id=?",
                      [(run_id, i) for i in upload_ids])


def start_run(trigger, n_samples, host=None):
    with connect() as c:
        cur = c.execute("INSERT INTO training_runs (trigger, status, started_at, n_samples, host) "
                        "VALUES (?,?,?,?,?)", (trigger, "running", now(), n_samples, host))
        return cur.lastrowid


def active_run(max_age_minutes=60):
    """The retraining run currently in progress (shared by all replicas), if any."""
    with connect() as c:
        r = c.execute("SELECT * FROM training_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
    if r is None:
        return None
    started = datetime.fromisoformat(r["started_at"])
    if (datetime.now(timezone.utc) - started).total_seconds() > max_age_minutes * 60:
        return None
    return dict(r)


def mark_interrupted(host):
    """On startup, runs this container left in 'running' state were interrupted (crash/restart)."""
    with connect() as c:
        c.execute("UPDATE training_runs SET status='failed', finished_at=?, error=? "
                  "WHERE status='running' AND host=?", (now(), "interrupted (container restarted)", host))


def finish_run(run_id, status, result=None, error=None):
    with connect() as c:
        c.execute("UPDATE training_runs SET status=?, finished_at=?, result=?, error=? WHERE id=?",
                  (status, now(), json.dumps(result) if result else None, error, run_id))


def list_runs(limit=10):
    with connect() as c:
        rows = c.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["result"] = json.loads(d["result"]) if d["result"] else None
        out.append(d)
    return out


def log_prediction(label, confidence, latency_ms, model_version):
    with connect() as c:
        c.execute("INSERT INTO predictions (created_at, label, confidence, latency_ms, model_version) "
                  "VALUES (?,?,?,?,?)", (now(), label, confidence, latency_ms, model_version))


def prediction_stats():
    with connect() as c:
        total = c.execute("SELECT COUNT(*), AVG(latency_ms) FROM predictions").fetchone()
        by_label = c.execute("SELECT label, COUNT(*) n FROM predictions GROUP BY label").fetchall()
    return {"total": total[0], "avg_latency_ms": round(total[1] or 0, 2),
            "by_label": {r["label"]: r["n"] for r in by_label}}
