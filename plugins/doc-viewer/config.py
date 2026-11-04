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
    v = var("PUBLIC_BASE_URL") or _evonic_base_url()
    return str(v).rstrip("/")


def _evonic_base_url() -> str:
    """Follow Evonic's root config: EVONIC_PUBLIC_URL, then http://127.0.0.1:<PORT>."""
    try:
        import importlib
        root = importlib.import_module("config")  # Evonic root config.py
        pub = getattr(root, "EVONIC_PUBLIC_URL", "") or ""
        port = getattr(root, "PORT", None)
    except Exception:
        pub, port = "", None
    pub = pub or os.environ.get("EVONIC_PUBLIC_URL", "").strip()
    return pub or f"http://127.0.0.1:{port or os.environ.get('PORT') or 8080}"


def template(name=None) -> str:
    n = (name or var("DEFAULT_TEMPLATE") or "paper").strip()
    return n if n in TEMPLATES else "paper"
