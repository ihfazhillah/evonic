"""share_doc: mint a token-protected viewer URL for one of the agent's markdown artifacts."""
from __future__ import annotations
import importlib.util, os, sys, time
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXTS = (".md", ".markdown")


def _load(mod, fname):
    m = sys.modules.get(mod)
    if m is None:
        spec = importlib.util.spec_from_file_location(mod, os.path.join(_ROOT, fname))
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
    return m


def execute(agent: dict, args: dict) -> dict:
    from backend.tools.lib import artifact_delivery as d
    from backend.tools._workspace import effective_agent_id
    cfg = _load("doc_viewer_plugin_config", "config.py")
    if not cfg.enabled():
        return {"error": "doc-viewer is disabled."}
    filename = str(args.get("filename") or "")
    if not filename.lower().endswith(_EXTS):
        return {"error": "Only markdown artifacts (.md, .markdown) can be viewed."}
    path, err = d.resolve_owned_artifact(agent, filename)
    if err:
        return {"error": err}
    tpl = cfg.template(args.get("template"))
    token, now = d.mint_token(), time.time()
    expires = now + cfg.ttl()
    _load("doc_viewer_plugin_db", "db.py").db.create(
        d.hash_token(token), effective_agent_id(agent), os.path.basename(path), path, tpl, now, expires)
    url = f"{cfg.base_url()}/plugin/doc-viewer/v/{token}"
    return {"result": "ok", "url": url, "export_html": url + "/export.html", "export_pdf": url + "/export.pdf",
            "template": tpl, "filename": os.path.basename(path),
            "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat()}
