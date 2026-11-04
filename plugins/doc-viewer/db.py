"""DocViewerDB: SQLite store for token-protected doc viewer links (digest only)."""
from __future__ import annotations
import os, sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_shared = os.path.join(BASE_DIR, "shared", "data")
_root = _shared if os.path.isdir(_shared) else os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(_root, "db", "plugins", "doc_viewer.db")


class DocViewerDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        with self._connect() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS doc_viewer_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash TEXT NOT NULL UNIQUE,
                owner_agent_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                template TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                revoked_at REAL,
                view_count INTEGER NOT NULL DEFAULT 0)""")

    @contextmanager
    def _connect(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create(self, token_hash, owner, filename, path, template, now, expires_at) -> None:
        with self._connect() as c:
            c.execute("INSERT INTO doc_viewer_tokens (token_hash, owner_agent_id, filename, artifact_path, template, created_at, expires_at) VALUES (?,?,?,?,?,?,?)",
                      (token_hash, owner, filename, path, template, now, expires_at))

    def get(self, token_hash) -> Optional[Dict[str, Any]]:
        with self._connect() as c:
            r = c.execute("SELECT * FROM doc_viewer_tokens WHERE token_hash=?", (token_hash,)).fetchone()
            return dict(r) if r else None

    def list_all(self, limit=500):
        with self._connect() as c:
            rows = c.execute("SELECT id, owner_agent_id, filename, template, created_at, expires_at, revoked_at, view_count FROM doc_viewer_tokens ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, row_id) -> Optional[Dict[str, Any]]:
        with self._connect() as c:
            r = c.execute("SELECT * FROM doc_viewer_tokens WHERE id=?", (row_id,)).fetchone()
            return dict(r) if r else None

    def touch(self, token_hash) -> None:
        with self._connect() as c:
            c.execute("UPDATE doc_viewer_tokens SET view_count=view_count+1 WHERE token_hash=?", (token_hash,))

    def revoke(self, token_hash, now) -> bool:
        with self._connect() as c:
            return c.execute("UPDATE doc_viewer_tokens SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL", (now, token_hash)).rowcount > 0

    def delete_expired(self, now) -> int:
        with self._connect() as c:
            return c.execute("DELETE FROM doc_viewer_tokens WHERE expires_at<=? OR revoked_at IS NOT NULL", (now,)).rowcount


db = DocViewerDB()
