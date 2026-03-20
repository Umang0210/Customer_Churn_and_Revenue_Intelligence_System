"""
Webapp — Customer Churn Intelligence Dashboard
===============================================
Serves the unified single-page dashboard and proxies API calls
to the inference API (port 5000).

Run:
    python -m uvicorn src.webapp.main:app --port 8000 --reload
"""

import os
import json
import httpx
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

INFERENCE_API = os.getenv("API_BASE_URL", "http://localhost:5000")

app = FastAPI(title="Churn Intelligence Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# API PROXY — /api/* → inference API on port 5000
# ══════════════════════════════════════════════════════════════════════════════
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_api(path: str, request: Request):
    target_url = f"{INFERENCE_API}/api/{path}"
    if request.query_params:
        target_url += "?" + str(request.query_params)

    body    = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                method=request.method, url=target_url,
                headers=headers, content=body,
            )
        return Response(
            content=resp.content, status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.ConnectError:
        return Response(
            content=json.dumps({"detail": "API server not running on port 5000."}).encode(),
            status_code=503, media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=json.dumps({"detail": str(e)}).encode(),
            status_code=500, media_type="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH PROXY
# ══════════════════════════════════════════════════════════════════════════════
@app.api_route("/health", methods=["GET"])
async def proxy_health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{INFERENCE_API}/health")
        return Response(content=r.content, status_code=r.status_code,
                        media_type="application/json")
    except Exception:
        return Response(
            content=json.dumps({"status": "api_offline"}).encode(),
            media_type="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ROOT — serve index.html
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found in static/</h1>", status_code=404)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC FILES — serve dashboard_data.js, dashboard.css etc at ROOT path
# e.g. GET /dashboard_data.js  →  static/dashboard_data.js
# MUST be registered AFTER all other routes so it doesn't swallow /api/*
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/{filename:path}")
async def serve_file(filename: str):
    """
    Serves any file from the static/ folder at the root URL path.
    This means /dashboard_data.js serves static/dashboard_data.js directly,
    which is what the browser expects when index.html does:
        <script src="dashboard_data.js"></script>
    """
    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = STATIC_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    # Content-type map
    suffix = file_path.suffix.lower()
    content_types = {
        ".html": "text/html; charset=utf-8",
        ".js":   "application/javascript; charset=utf-8",
        ".css":  "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".png":  "image/png",
        ".ico":  "image/x-icon",
    }
    media_type = content_types.get(suffix, "application/octet-stream")

    content = file_path.read_bytes()
    return Response(content=content, media_type=media_type)
