"""Local browser UI for configuring zotero-mcp (``zotero-mcp config-ui``).

A small stdlib HTTP server (no extra dependencies) bound to 127.0.0.1. It reads
and writes ``~/.config/zotero-mcp/config.json`` — the same file the CLI and the
server use — so changes take effect the next time the MCP server starts.

The UI runs in its own process, separate from the (stdio) MCP server that
Claude launches, so it cannot switch the library of an already-running server.
Restart the MCP client to pick up a new default.

Because it can write credentials, the server is hardened against other web
pages reaching it through the browser: it only answers requests whose ``Host``
is loopback, and every ``/api`` call needs a per-launch token that is embedded
in the served page. POSTs must be JSON, which a cross-origin page cannot send
without a CORS preflight that this server never grants.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from zotero_mcp import client as _client

logger = logging.getLogger(__name__)

DEFAULT_PORT = 23120
LOCAL_ZOTERO_URL = "http://localhost:23119/api/"

# Embedding provider names, kept in sync with ``embeddings.registry.PROVIDERS``
# (enforced by a test). Not imported from there: the registry pulls in chromadb,
# an optional extra, and the config UI must work on a base install.
EMBEDDING_PROVIDERS: tuple[str, ...] = (
    "default", "openai", "openai-compatible", "gemini", "voyage", "ollama", "huggingface",
)

# Credentials the UI may store (under ``client_env`` in config.json, which the
# CLI applies to the environment at startup without overriding real env vars).
SECRET_KEYS: tuple[str, ...] = (
    "ZOTERO_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_COMPAT_API_KEY",
    "GEMINI_API_KEY",
    "VOYAGE_API_KEY",
    "HUGGINGFACE_APIKEY",
)

_MAX_BODY = 64 * 1024


class ConfigError(ValueError):
    """A request the UI rejects; the message is shown to the user."""


# ---------------------------------------------------------------------------
# config.json access
# ---------------------------------------------------------------------------


def _read_config() -> dict:
    return _client._readable_config_for_update()


def _write_config(cfg: dict) -> None:
    path = _client.ZOTERO_MCP_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    # The file can hold API keys — keep it owner-only (no-op without POSIX perms).
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _mask(value: str) -> str:
    return "••••" + value[-4:] if len(value) > 8 else "••••"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _zotero_running() -> bool:
    if not _client.is_local_mode():
        return False
    try:
        return requests.get(LOCAL_ZOTERO_URL, timeout=1.5).status_code < 500
    except requests.RequestException:
        return False


def get_state() -> dict[str, Any]:
    cfg = _read_config()
    sem = cfg.get("semantic_search") or {}
    emb_cfg = sem.get("embedding_config") or {}
    env = cfg.get("client_env") or {}

    env_id = os.getenv("ZOTERO_LIBRARY_ID")
    default = cfg.get("default_library") or {}
    if env_id:
        library = {
            "library_id": env_id,
            "library_type": os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
            "source": "env",
        }
    elif default.get("library_id"):
        library = {**default, "source": "config"}
    else:
        library = {"library_id": "0", "library_type": "user", "source": "default"}

    secrets_state = {}
    for key in SECRET_KEYS:
        stored = env.get(key)
        if stored:
            secrets_state[key] = {"set": True, "preview": _mask(str(stored)), "source": "config"}
        elif os.getenv(key):
            secrets_state[key] = {"set": True, "preview": _mask(os.environ[key]), "source": "env"}
        else:
            secrets_state[key] = {"set": False, "preview": "", "source": None}

    return {
        "config_path": str(_client.ZOTERO_MCP_CONFIG_PATH),
        "local_mode": _client.is_local_mode(),
        "zotero_running": _zotero_running(),
        "library": library,
        "embedding": {
            "provider": sem.get("embedding_model", "default"),
            "model": emb_cfg.get("model_name", ""),
            "base_url": emb_cfg.get("base_url", ""),
        },
        "providers": list(EMBEDDING_PROVIDERS),
        "secrets": secrets_state,
    }


# ---------------------------------------------------------------------------
# Libraries & collections
# ---------------------------------------------------------------------------


def list_libraries() -> list[dict[str, Any]]:
    """Personal + group libraries as ``{library_id, library_type, name, item_count?}``."""
    libs: list[dict[str, Any]] = [
        {"library_id": "0", "library_type": "user", "name": "My Library"}
    ]
    if _client.is_local_mode():
        from zotero_mcp.config import load_config
        from zotero_mcp.local_db import LocalZoteroReader

        reader = LocalZoteroReader(db_path=load_config().resolve_zotero_db_path())
        try:
            for lib in reader.get_libraries():
                if lib["type"] == "user":
                    libs[0]["item_count"] = lib["itemCount"]
                elif lib["type"] == "group":
                    libs.append({
                        "library_id": str(lib["groupID"]),
                        "library_type": "group",
                        "name": lib["groupName"],
                        "item_count": lib["itemCount"],
                    })
        finally:
            reader.close()
        return libs

    zot = _client.get_zotero_client()
    libs[0]["library_id"] = str(zot.library_id)
    for group in zot.groups():
        libs.append({
            "library_id": str(group.get("id")),
            "library_type": "group",
            "name": group.get("data", {}).get("name", "Unknown"),
        })
    return libs


def _validate_library(library_id: Any, library_type: Any) -> tuple[str, str]:
    library_type = str(library_type or "")
    library_id = str(library_id or "")
    if library_type not in ("user", "group"):
        raise ConfigError("library_type must be 'user' or 'group'")
    if library_type == "group" and not library_id.isdigit():
        raise ConfigError("group library_id must be numeric")
    return library_id, library_type


def list_collections(library_id: str, library_type: str) -> list[dict[str, Any]]:
    """Collections of a library as a flat ``{key, name, parent}`` list."""
    library_id, library_type = _validate_library(library_id, library_type)
    if _client.is_local_mode():
        from zotero_mcp.config import load_config
        from zotero_mcp.local_db import LocalZoteroReader

        group_id = int(library_id) if library_type == "group" else 0
        reader = LocalZoteroReader(db_path=load_config().resolve_zotero_db_path())
        try:
            raw = reader.list_collections(group_id=group_id)
        finally:
            reader.close()
    else:
        from pyzotero import zotero

        api_key = os.getenv("ZOTERO_API_KEY") or (_read_config().get("client_env") or {}).get(
            "ZOTERO_API_KEY"
        )
        if not api_key:
            raise ConfigError("ZOTERO_API_KEY is needed to list collections in web mode")
        zot = zotero.Zotero(library_id, library_type, api_key)
        raw = zot.everything(zot.collections())
    return [
        {
            "key": c["key"],
            "name": c["data"]["name"],
            "parent": c["data"].get("parentCollection") or None,
        }
        for c in raw
    ]


def save_library(library_id: Any, library_type: Any) -> dict[str, str]:
    library_id, library_type = _validate_library(library_id, library_type)
    if library_type == "user":
        library_id = "0" if _client.is_local_mode() else library_id or "0"
    _client._persist_library_to_config(library_id, library_type)
    return {"library_id": library_id, "library_type": library_type}


def clear_library() -> None:
    _client._clear_library_from_config()


# ---------------------------------------------------------------------------
# Credentials & embedding
# ---------------------------------------------------------------------------


def save_credentials(values: dict[str, Any], clear: list[str] | None = None) -> list[str]:
    """Store non-empty *values*; remove keys in *clear*. Returns changed key names."""
    clear = clear or []
    unknown = [k for k in [*values, *clear] if k not in SECRET_KEYS]
    if unknown:
        raise ConfigError(f"Unsupported credential(s): {', '.join(sorted(unknown))}")
    cfg = _read_config()
    env = dict(cfg.get("client_env") or {})
    changed = []
    for key, value in values.items():
        value = str(value or "").strip()
        if value:  # an empty field means "leave unchanged"
            env[key] = value
            changed.append(key)
    for key in clear:
        if env.pop(key, None) is not None:
            changed.append(key)
    cfg["client_env"] = env
    _write_config(cfg)
    return changed


def save_embedding(provider: Any, model: Any = "", base_url: Any = "") -> dict[str, str]:
    provider = str(provider or "")
    if provider not in EMBEDDING_PROVIDERS:
        raise ConfigError(f"Unknown embedding provider: {provider!r}")
    model = str(model or "").strip()
    base_url = str(base_url or "").strip()
    if provider == "openai-compatible" and not (model and base_url):
        raise ConfigError("openai-compatible needs both a model name and a base URL")
    if base_url and not base_url.startswith(("http://", "https://")):
        raise ConfigError("base URL must start with http:// or https://")

    cfg = _read_config()
    sem = dict(cfg.get("semantic_search") or {})
    emb_cfg = dict(sem.get("embedding_config") or {})
    for key, value in (("model_name", model), ("base_url", base_url)):
        if value:
            emb_cfg[key] = value
        else:
            emb_cfg.pop(key, None)
    sem["embedding_model"] = provider
    sem["embedding_config"] = emb_cfg
    cfg["semantic_search"] = sem
    _write_config(cfg)
    return {"provider": provider, "model": model, "base_url": base_url}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

_PAGE = Path(__file__).with_name("data") / "config_ui.html"


def _make_handler(token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "zotero-mcp-config"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            logger.debug(fmt, *args)

        # -- helpers --------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _guard(self, api: bool) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json({"error": "forbidden host"}, HTTPStatus.FORBIDDEN)
                return False
            if api and not secrets.compare_digest(
                self.headers.get("X-Config-Token", ""), token
            ):
                self._json({"error": "bad token"}, HTTPStatus.FORBIDDEN)
                return False
            return True

        # -- routes ---------------------------------------------------------
        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/":
                if not self._guard(api=False):
                    return
                html = _PAGE.read_text(encoding="utf-8").replace("__TOKEN__", token)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if not url.path.startswith("/api/"):
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            if not self._guard(api=True):
                return
            try:
                if url.path == "/api/state":
                    return self._json(get_state())
                if url.path == "/api/libraries":
                    return self._json(list_libraries())
                if url.path == "/api/collections":
                    q = parse_qs(url.query)
                    return self._json(list_collections(
                        q.get("library_id", [""])[0], q.get("library_type", [""])[0]
                    ))
            except ConfigError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # surface to the UI instead of a bare 500
                logger.exception("config-ui request failed")
                return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            if not self._guard(api=True):
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._json({"error": "JSON required"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= _MAX_BODY:
                    raise ConfigError("request body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ConfigError("expected a JSON object")
                path = urlparse(self.path).path
                if path == "/api/library":
                    if body.get("clear"):
                        clear_library()
                        return self._json({"ok": True})
                    return self._json(
                        {"ok": True, **save_library(body.get("library_id"), body.get("library_type"))}
                    )
                if path == "/api/credentials":
                    changed = save_credentials(body.get("values") or {}, body.get("clear") or [])
                    return self._json({"ok": True, "changed": changed})
                if path == "/api/embedding":
                    return self._json({"ok": True, **save_embedding(
                        body.get("provider"), body.get("model"), body.get("base_url")
                    )})
            except (ConfigError, json.JSONDecodeError, ValueError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except OSError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def create_server(port: int = DEFAULT_PORT, token: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    """Bind a server on 127.0.0.1 (port 0 = pick a free one). Returns (server, token)."""
    token = token or secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), lambda *a, **k: None)
    server.RequestHandlerClass = _make_handler(token, server.server_address[1])
    return server, token


def run(port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    try:
        server, token = create_server(port)
    except OSError as exc:
        print(f"Could not start the config UI on port {port}: {exc}")
        return 1
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"Zotero MCP config UI: {url}")
    print("Press Ctrl+C to stop. Changes apply the next time the MCP server starts.")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
