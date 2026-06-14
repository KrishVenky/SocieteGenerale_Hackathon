"""
FastAPI server for SentinelAI dashboard.
Run: uvicorn app.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from threading import Thread

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.narrator import NarratorEngine, SG_RESOURCE_MAP, SG_DEPT_MAP

app = FastAPI(title="WatchDog")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

_df_cache: dict = {}


def _load_df() -> pd.DataFrame:
    if "df" not in _df_cache:
        path = ROOT / os.getenv("FEATURES_PATH", "data/processed/real/features.parquet")
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        _df_cache["df"] = df
    return _df_cache["df"]


def _safe_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            pass
    return []


def _float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (v != v) else v  # NaN check
    except Exception:
        return default


@app.get("/", response_class=HTMLResponse)
async def root():
    return (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/alerts")
async def get_alerts():
    threshold = int(os.getenv("RISK_THRESHOLD", "65"))
    df = _load_df()

    alerts_df = (
        df[df["risk_score"] >= threshold]
        .sort_values("risk_score", ascending=False)
        .head(50)
    )

    alerts = []
    for _, row in alerts_df.iterrows():
        resource = str(row.get("resource", ""))
        sg_name, sg_class = SG_RESOURCE_MAP.get(resource, (resource, "Unknown"))
        dept = str(row.get("department", ""))
        sg_dept = SG_DEPT_MAP.get(dept, dept)

        alerts.append({
            "event": {
                "user_id": str(row.get("user_id", "")),
                "username": str(row.get("username", "")),
                "action": str(row.get("action", "")),
                "resource": resource,
                "resource_sensitivity": str(row.get("resource_sensitivity", "")),
                "status": str(row.get("status", "")),
                "source_ip": str(row.get("source_ip", "")),
                "time_classification": str(row.get("time_classification", "")),
                "timestamp": str(row["timestamp"]),
            },
            "profile": {
                "username": str(row.get("username", "")),
                "department": dept,
                "department_display": sg_dept,
                "job_title": str(row.get("job_title", "")),
                "privilege_level": str(row.get("privilege_level", "")),
                "days_inactive": int(_float(row.get("days_inactive", 0))),
                "hire_date": str(row.get("hire_date", "")),
                "systems_access": str(row.get("systems_access", "")),
            },
            "scored": {
                "risk_score": int(row.get("risk_score", 0)),
                "severity": str(row.get("severity", "LOW")),
                "triggered_signals": _safe_list(row.get("triggered_signals")),
                "mitre_techniques": _safe_list(row.get("mitre_techniques")),
                "component_scores": {
                    "behavioral": _float(row.get("behavioral_score")),
                    "lstm_sequence": _float(row.get("lstm_component")),
                    "graph_divergence": _float(row.get("graph_component")),
                },
            },
            "resource_display": sg_name,
            "resource_class": sg_class,
            "peer_deviation_pct": _float(row.get("peer_deviation_pct")),
        })

    total = len(df)
    n_critical = int((df["risk_score"] >= 85).sum())
    n_high = int(((df["risk_score"] >= 70) & (df["risk_score"] < 85)).sum())

    return {
        "alerts": alerts,
        "total_events": total,
        "total_alerts": len(alerts_df),
        "n_critical": n_critical,
        "n_high": n_high,
        "avg_risk": round(_float(df["risk_score"].mean()), 1),
        "threshold": threshold,
    }


@app.get("/api/demo")
async def get_demo():
    demo_path = ROOT / os.getenv(
        "DEMO_EVENTS_PATH", "data/processed/real/demo_replay/scripted_events.jsonl"
    )
    events = []
    if demo_path.exists():
        with open(demo_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return sorted(
        events, key=lambda e: e.get("scored", {}).get("risk_score", 0), reverse=True
    )


@app.get("/api/baseline/{user_id}")
async def get_baseline(user_id: str):
    df = _load_df()
    user_df = df[df["user_id"] == user_id].copy()
    if user_df.empty:
        return {"dates": [], "counts": []}
    daily = user_df.groupby(user_df["timestamp"].dt.date).size().reset_index()
    daily.columns = ["date", "count"]
    return {
        "dates": [str(d) for d in daily["date"]],
        "counts": daily["count"].tolist(),
    }


@app.post("/api/narrative")
async def narrative_stream(alert: dict):
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            narrator = NarratorEngine()
            for chunk in narrator.stream(alert):
                loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
        loop.call_soon_threadsafe(q.put_nowait, ("done", None))

    Thread(target=_run, daemon=True).start()

    async def _gen():
        while True:
            try:
                msg_type, content = await asyncio.wait_for(q.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield 'data: {"error":"timeout"}\n\n'
                break
            if msg_type == "chunk":
                yield f"data: {json.dumps({'chunk': content})}\n\n"
            elif msg_type == "error":
                yield f"data: {json.dumps({'error': content})}\n\n"
                break
            elif msg_type == "done":
                break
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/action")
async def take_action(payload: dict):
    action = payload.get("action", "")
    username = payload.get("username", "unknown")
    messages = {
        "freeze": f"Account {username} frozen — AD session revoked, IT Security notified.",
        "escalate": "Incident escalated to CISO — ticket #INC-2026-0471 opened in ServiceNow.",
        "audit": "Audit log snapshot preserved to immutable S3 — chain of custody maintained.",
    }
    return {"status": "success", "message": messages.get(action, "Action completed.")}
