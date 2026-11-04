"""doc-viewer plugin: token-protected markdown viewer and export routes.

  GET /plugin/doc-viewer/                       JSON status probe
  GET /plugin/doc-viewer/v/<token>              themed viewer
  GET /plugin/doc-viewer/v/<token>/export.html  (task 8)
  GET /plugin/doc-viewer/v/<token>/export.pdf   (task 9)

Unknown, expired, and revoked tokens all return an identical 404.
"""
from __future__ import annotations
import os, time
from flask import Blueprint, jsonify, render_template

from backend.tools.lib import artifact_delivery as d
from . import config as cfg
from .db import db as view_db

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 5 * 1024 * 1024


def _not_found(**_):
    return jsonify({"error": "not_found"}), 404


def _secure(resp):
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Robots-Tag"] = "noindex"
    return resp


def resolve(token):
    """Return (row, markdown_text) for a live token, or None."""
    if not cfg.enabled() or not d.is_valid_token_format(token):
        return None
    h = d.hash_token(token)
    row = view_db.get(h)
    if not row or not d.constant_time_equal(row.get("token_hash", ""), h):
        return None
    if row.get("revoked_at") is not None or float(row.get("expires_at") or 0) <= time.time():
        return None
    path = row.get("artifact_path") or ""
    if os.path.islink(path) or not os.path.isfile(path) or os.path.getsize(path) > MAX_BYTES:
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return row, f.read()


def _view(token):
    hit = resolve(token)
    if hit is None:
        return _not_found()
    row, md = hit
    view_db.touch(row["token_hash"])
    resp = render_template("doc_viewer/view.html", title=row["filename"], markdown=md,
                           template=cfg.template(row.get("template")), token=token)
    from flask import make_response
    return _secure(make_response(resp))


_JS = ("vendor/markdown-it.min.js", "vendor/purify.min.js", "vendor/pele.global.js", "viewer.js")


def _static(rel):
    with open(os.path.join(PLUGIN_DIR, "static", rel), encoding="utf-8") as f:
        # Prevent a literal closing tag inside vendored JS from ending the inline <script>.
        return f.read().replace("</script", "<\\/script")


def _build_export(row, md):
    tpl = cfg.template(row.get("template"))
    return render_template("doc_viewer/export.html", title=row["filename"], markdown=md, template=tpl,
                           css=_static(f"themes/{tpl}.css").replace("</style", "<\\/style"),
                           scripts=[_static(p) for p in _JS])


def _download_name(row, ext):
    return os.path.splitext(d.sanitize_download_name(row["filename"], "document.md"))[0] + ext


def _export_html(token):
    hit = resolve(token)
    if hit is None:
        return _not_found()
    row, md = hit
    from flask import make_response
    resp = make_response(_build_export(row, md))
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="{_download_name(row, ".html")}"'
    return _secure(resp)


def _export_pdf(token):
    hit = resolve(token)
    if hit is None:
        return _not_found()
    row, md = hit
    from flask import make_response
    from . import pdf as pdfmod
    try:
        data, used = pdfmod.render_pdf(_build_export(row, md), str(cfg.var("PDF_ENGINE") or "weasyprint"))
    except Exception:
        return jsonify({"error": "pdf_failed"}), 500
    resp = make_response(data)
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["X-Doc-Viewer-Engine"] = used
    resp.headers["Content-Disposition"] = f'attachment; filename="{_download_name(row, ".pdf")}"'
    return _secure(resp)


def _status():
    return jsonify({"plugin": "doc-viewer", "version": "0.1.0", "enabled": cfg.enabled()})


def create_blueprint() -> Blueprint:
    bp = Blueprint("doc_viewer", __name__,
                   template_folder=os.path.join(PLUGIN_DIR, "templates"),
                   static_folder=os.path.join(PLUGIN_DIR, "static"),
                   static_url_path="/plugin/doc-viewer/static")
    bp.add_url_rule("/plugin/doc-viewer/", "status", _status, methods=["GET"])
    bp.add_url_rule("/plugin/doc-viewer/v/<token>", "view", _view, methods=["GET"])
    bp.add_url_rule("/plugin/doc-viewer/v/<token>/export.html", "export_html", _export_html, methods=["GET"])
    bp.add_url_rule("/plugin/doc-viewer/v/<token>/export.pdf", "export_pdf", _export_pdf, methods=["GET"])
    bp.add_url_rule("/plugin/doc-viewer/v/<path:_rest>", "nf", _not_found, methods=["GET"])
    return bp
