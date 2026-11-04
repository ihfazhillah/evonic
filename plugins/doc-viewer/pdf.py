"""PDF export: pre-render JS in headless Chrome, flatten CSS vars, then WeasyPrint (Chrome fallback)."""
from __future__ import annotations
import os, re, shutil, subprocess, tempfile, logging

_log = logging.getLogger(__name__)
CHROME_CANDIDATES = ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable")
TIMEOUT = 60
_VAR_RE = re.compile(r"var\((--[\w-]+)(?:,\s*([^()]*(?:\([^()]*\))?[^()]*))?\)")


def chrome_bin():
    for c in CHROME_CANDIDATES:
        p = shutil.which(c)
        if p:
            return p
    return None


def _chrome(args, cwd):
    exe = chrome_bin()
    if not exe:
        raise RuntimeError("chrome not available")
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
           f"--user-data-dir={os.path.join(cwd, 'profile')}", "--virtual-time-budget=5000", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=TIMEOUT, check=True)


def flatten_css_vars(html: str) -> str:
    """Replace var(--x[, fallback]) with resolved values (WeasyPrint lacks SVG var support)."""
    for _ in range(10):
        decls = dict(re.findall(r"(--[\w-]+)\s*:\s*([^;\"}]+)", html))

        def sub(m):
            v = decls.get(m.group(1))
            if v is not None and "var(" not in v:
                return v.strip()
            if v is not None:
                return m.group(0)
            return (m.group(2) or "currentColor").strip()

        n = _VAR_RE.sub(sub, html)
        if n == html:
            break
        html = n
    return _VAR_RE.sub(lambda m: (m.group(2) or "currentColor").strip(), html)


_SVG_OPEN_RE = re.compile(r"<svg\b[^>]*>", re.I)


def fix_svg_for_weasy(html: str, fg: str = "#1f1f1f") -> str:
    """WeasyPrint drops SVGs sized via max-width/height:auto or colored with currentColor."""
    def fix(m):
        t = m.group(0).replace("max-width:100%;height:auto;", "")
        return t.replace("color:currentColor", f"color:{fg}")
    return _SVG_OPEN_RE.sub(fix, html)


def _strip_scripts(html: str) -> str:
    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)


def render_pdf(export_html: str, engine: str = "weasyprint") -> tuple[bytes, str]:
    """Return (pdf_bytes, engine_used)."""
    with tempfile.TemporaryDirectory(prefix="docviewer-pdf-") as tmp:
        src = os.path.join(tmp, "doc.html")
        with open(src, "w", encoding="utf-8") as f:
            f.write(export_html)
        if engine != "chromium":
            try:
                dom = _chrome(["--dump-dom", f"file://{src}"], tmp).stdout.decode("utf-8", "replace")
                import weasyprint
                flat = fix_svg_for_weasy(flatten_css_vars(_strip_scripts(dom)))
                return weasyprint.HTML(string=flat, base_url=tmp).write_pdf(), "weasyprint"
            except Exception as exc:
                _log.warning("doc-viewer weasyprint path failed, falling back to chrome: %s", exc)
        out = os.path.join(tmp, "doc.pdf")
        _chrome(["--no-pdf-header-footer", f"--print-to-pdf={out}", f"file://{src}"], tmp)
        with open(out, "rb") as f:
            return f.read(), "chromium"
