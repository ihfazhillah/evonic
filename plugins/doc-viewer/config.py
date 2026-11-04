"""Plugin variable resolution for doc-viewer."""
from __future__ import annotations
import os

TEMPLATES = ("paper", "course-dark", "explainer")
DEFAULTS = {"ENABLED": True, "PUBLIC_BASE_URL": "", "TOKEN_TTL_SECONDS": 604800,
            "DEFAULT_TEMPLATE": "paper", "PDF_ENGINE": "weasyprint"}


def var(name):
    try:
        from backend.tools.lib import artifact_delivery as d
        v = d._plugin_config("doc-viewer", name)
    except Exception:
        v = None
    return DEFAULTS.get(name) if v in (None, "") else v


def enabled() -> bool:
    v = var("ENABLED")
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def ttl() -> int:
    try:
        return max(60, int(float(var("TOKEN_TTL_SECONDS"))))
    except (TypeError, ValueError):
        return DEFAULTS["TOKEN_TTL_SECONDS"]


def base_url() -> str:
    v = var("PUBLIC_BASE_URL") or os.environ.get("EVONIC_PUBLIC_URL") or "http://127.0.0.1:8080"
    return str(v).rstrip("/")


def template(name=None) -> str:
    n = (name or var("DEFAULT_TEMPLATE") or "paper").strip()
    return n if n in TEMPLATES else "paper"
