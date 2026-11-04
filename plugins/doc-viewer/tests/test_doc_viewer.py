import importlib, os, shutil, subprocess, sys, time, types
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PLUGIN = os.path.join(ROOT, "plugins", "doc-viewer")
FIXTURE = os.path.join(PLUGIN, "tests", "fixtures", "sample.md")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture()
def env(tmp_path):
    pkg = "doc_viewer_pkg_test"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg); m.__path__ = [PLUGIN]; sys.modules[pkg] = m
    routes = importlib.import_module(pkg + ".routes")
    dbmod = importlib.import_module(pkg + ".db")
    routes.view_db = dbmod.DocViewerDB(str(tmp_path / "dv.db"))
    from flask import Flask
    app = Flask("t"); app.secret_key = "t"; app.register_blueprint(routes.create_blueprint())
    from backend.tools.lib import artifact_delivery as d

    def mint(expires=600, path=FIXTURE):
        tok = d.mint_token(); now = time.time()
        routes.view_db.create(d.hash_token(tok), "boss", "sample.md", path, "paper", now, now + expires)
        return tok
    return types.SimpleNamespace(c=app.test_client(), routes=routes, mint=mint, d=d)


def test_view_ok_and_headers(env):
    r = env.c.get(f"/plugin/doc-viewer/v/{env.mint()}")
    assert r.status_code == 200
    assert r.headers["Cache-Control"] == "no-store"
    assert b"Contoh Dokumen" in r.data


@pytest.mark.parametrize("suffix", ["", "/export.html", "/export.pdf"])
def test_unknown_token_404(env, suffix):
    assert env.c.get(f"/plugin/doc-viewer/v/{env.d.mint_token()}{suffix}").status_code == 404


def test_traversal_404(env):
    assert env.c.get("/plugin/doc-viewer/v/../../etc/passwd").status_code == 404


def test_expired_and_revoked_404(env):
    assert env.c.get(f"/plugin/doc-viewer/v/{env.mint(expires=-1)}").status_code == 404
    tok = env.mint()
    env.routes.view_db.revoke(env.d.hash_token(tok), time.time())
    assert env.c.get(f"/plugin/doc-viewer/v/{tok}").status_code == 404


def test_export_html_self_contained(env):
    r = env.c.get(f"/plugin/doc-viewer/v/{env.mint()}/export.html")
    assert r.status_code == 200
    assert 'filename="sample.html"' in r.headers["Content-Disposition"]
    assert b'src="/plugin' not in r.data


def test_flatten_css_vars_nested():
    pdf = importlib.import_module("doc_viewer_pkg_test.pdf") if "doc_viewer_pkg_test" in sys.modules else None
    if pdf is None:
        pytest.skip("package not loaded")
    html = '<svg style="--_s:var(--pele-surface,#f3f3f3)"><rect fill="var(--_s)"/></svg>'
    out = pdf.flatten_css_vars(html)
    assert "var(" not in out and 'fill="#f3f3f3"' in out


@pytest.mark.skipif(not (shutil.which("google-chrome") or shutil.which("chromium")), reason="chrome missing")
@pytest.mark.parametrize("engine", ["weasyprint", "chromium"])
def test_export_pdf(env, engine):
    env.routes.cfg.DEFAULTS["PDF_ENGINE"] = engine
    r = env.c.get(f"/plugin/doc-viewer/v/{env.mint()}/export.pdf")
    assert r.status_code == 200 and r.data[:5] == b"%PDF-"
    assert r.headers["X-Doc-Viewer-Engine"] == engine
    if shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-", "-"], input=r.data, capture_output=True).stdout.decode()
        assert "Mulai" in txt and "Catatan" in txt


def _login(c):
    with c.session_transaction() as s:
        s["authenticated"] = True


def test_files_requires_auth(env):
    assert env.c.get("/plugin/doc-viewer/files").status_code == 404
    assert env.c.get("/plugin/doc-viewer/files/1").status_code == 404


def test_files_list_and_detail(env):
    env.mint()
    _login(env.c)
    r = env.c.get("/plugin/doc-viewer/files")
    assert r.status_code == 200 and b"sample.md" in r.data and b"/plugin/doc-viewer/files/1" in r.data
    d = env.c.get("/plugin/doc-viewer/files/1")
    assert d.status_code == 200 and b"Contoh Dokumen" in d.data
    assert env.c.get("/plugin/doc-viewer/files/999").status_code == 404
