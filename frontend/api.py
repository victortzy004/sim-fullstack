# file: api.py
import os
import requests
from typing import Any, Dict, Optional

# BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
BASE = "http://concept.alkimiya.io/api"


class APIError(RuntimeError):
    pass

def _url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return BASE.rstrip("/") + path

def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0):
    r = requests.get(_url(path), params=params, timeout=timeout)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise APIError(f"GET {path} -> {r.status_code}: {detail}")
    return r.json()

def api_post(path: str, json: Optional[Dict[str, Any]] = None, timeout: float = 10.0):
    r = requests.post(_url(path), json=json, timeout=timeout)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise APIError(f"POST {path} -> {r.status_code}: {detail}")
    return r.json()
