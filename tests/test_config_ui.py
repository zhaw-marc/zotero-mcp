"""Tests for the browser config UI (Issue #8).

Covers the config read/write helpers and the HTTP hardening (token, Host
header, JSON-only POST). No Zotero instance is needed.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

import zotero_mcp.client as _client
from zotero_mcp import config_ui


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", path)
    for var in ("ZOTERO_LIBRARY_ID", "ZOTERO_LIBRARY_TYPE", "ZOTERO_LOCAL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    _client._active_library_override.clear()
    yield path
    _client._active_library_override.clear()


def _read(path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_save_library_persists_group(config_path):
    out = config_ui.save_library("5294983", "group")
    assert out == {"library_id": "5294983", "library_type": "group"}
    assert _read(config_path)["default_library"] == out


@pytest.mark.parametrize("lid,ltype", [("abc", "group"), ("1", "feed"), ("", "")])
def test_save_library_rejects_bad_input(config_path, lid, ltype):
    with pytest.raises(config_ui.ConfigError):
        config_ui.save_library(lid, ltype)
    assert not config_path.exists()


def test_clear_library(config_path):
    config_ui.save_library("1", "group")
    config_ui.clear_library()
    assert "default_library" not in _read(config_path)


def test_save_credentials_stores_and_preserves_other_keys(config_path):
    config_path.write_text(json.dumps({"default_library": {"library_id": "1", "library_type": "group"}}))
    changed = config_ui.save_credentials({"OPENAI_API_KEY": " sk-test ", "VOYAGE_API_KEY": ""})
    assert changed == ["OPENAI_API_KEY"]
    cfg = _read(config_path)
    assert cfg["client_env"] == {"OPENAI_API_KEY": "sk-test"}
    assert cfg["default_library"]["library_id"] == "1"


def test_save_credentials_clear_and_unknown_key(config_path):
    config_ui.save_credentials({"OPENAI_API_KEY": "sk-test"})
    assert config_ui.save_credentials({}, clear=["OPENAI_API_KEY"]) == ["OPENAI_API_KEY"]
    assert _read(config_path)["client_env"] == {}
    with pytest.raises(config_ui.ConfigError):
        config_ui.save_credentials({"PATH": "/evil"})


def test_credentials_file_is_owner_only(config_path):
    config_ui.save_credentials({"OPENAI_API_KEY": "sk-test"})
    assert config_path.stat().st_mode & 0o077 == 0


def test_state_never_exposes_secret_values(config_path):
    config_ui.save_credentials({"OPENAI_API_KEY": "sk-supersecret-1234"})
    state = config_ui.get_state()
    assert "supersecret" not in json.dumps(state)
    assert state["secrets"]["OPENAI_API_KEY"]["set"] is True
    assert state["secrets"]["OPENAI_API_KEY"]["preview"].endswith("1234")
    assert state["secrets"]["VOYAGE_API_KEY"]["set"] is False


def test_state_library_precedence(config_path, monkeypatch):
    assert config_ui.get_state()["library"]["source"] == "default"
    config_ui.save_library("42", "group")
    assert config_ui.get_state()["library"]["source"] == "config"
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "7")
    assert config_ui.get_state()["library"] == {
        "library_id": "7", "library_type": "user", "source": "env",
    }


def test_save_embedding_openai_compatible(config_path):
    config_ui.save_embedding("openai-compatible", "bge-m3", "http://localhost:8000/v1")
    sem = _read(config_path)["semantic_search"]
    assert sem["embedding_model"] == "openai-compatible"
    assert sem["embedding_config"] == {"model_name": "bge-m3", "base_url": "http://localhost:8000/v1"}


def test_save_embedding_validation(config_path):
    with pytest.raises(config_ui.ConfigError):
        config_ui.save_embedding("nonsense")
    with pytest.raises(config_ui.ConfigError):
        config_ui.save_embedding("openai-compatible", "m", "")
    with pytest.raises(config_ui.ConfigError):
        config_ui.save_embedding("ollama", "m", "file:///etc/passwd")


def test_save_embedding_clears_stale_model(config_path):
    config_ui.save_embedding("voyage", "voyage-3")
    config_ui.save_embedding("default")
    assert _read(config_path)["semantic_search"]["embedding_config"] == {}


def test_invalid_config_is_not_overwritten(config_path):
    config_path.write_text("{not json")
    with pytest.raises(OSError):
        config_ui.save_credentials({"OPENAI_API_KEY": "sk-test"})
    assert config_path.read_text() == "{not json"


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@pytest.fixture()
def server(config_path):
    srv, token = config_ui.create_server(port=0, token="tok")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{srv.server_address[1]}", token
    srv.shutdown()
    srv.server_close()


def _request(host, path, *, token=None, body=None, headers=None, host_header=None):
    hdrs = {"Host": host_header or host, **(headers or {})}
    if token:
        hdrs["X-Config-Token"] = token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(f"http://{host}{path}", data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_index_embeds_token_and_escapes_nothing_unsafe(server):
    host, token = server
    status, body = _request(host, "/")
    assert status == 200
    assert token.encode() in body
    assert b"__TOKEN__" not in body


def test_api_requires_token(server):
    host, token = server
    assert _request(host, "/api/state")[0] == 403
    assert _request(host, "/api/state", token="wrong")[0] == 403
    assert _request(host, "/api/state", token=token)[0] == 200


def test_rejects_foreign_host_header(server):
    host, token = server
    assert _request(host, "/", host_header="evil.example:80")[0] == 403
    assert _request(host, "/api/state", token=token, host_header="evil.example:80")[0] == 403


def test_post_requires_json_content_type(server):
    host, token = server
    status, _ = _request(
        host, "/api/library", token=token, body={"clear": True},
        headers={"Content-Type": "text/plain"},
    )
    assert status == 415


def test_post_library_roundtrip_over_http(server, config_path):
    host, token = server
    status, body = _request(
        host, "/api/library", token=token,
        body={"library_id": "99", "library_type": "group"},
    )
    assert status == 200 and json.loads(body)["ok"] is True
    assert _read(config_path)["default_library"]["library_id"] == "99"
    status, body = _request(host, "/api/library", token=token, body={"library_id": "x", "library_type": "group"})
    assert status == 400
    assert "numeric" in json.loads(body)["error"]


def test_unknown_route_404(server):
    host, token = server
    assert _request(host, "/api/nope", token=token)[0] == 404
