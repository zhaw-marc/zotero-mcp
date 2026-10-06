"""Tests for persistent library/collection context (Issue #15).

Verifies:
- set_active_library writes default_library to config
- clear_active_library removes it from config
- load_library_from_config populates the override on startup
- ZOTERO_LIBRARY_ID env var takes precedence over config
- CLI set-library and clear-library commands round-trip correctly
"""

import json

import pytest

from zotero_mcp.client import (
    _active_library_override,
    clear_active_library,
    load_library_from_config,
    set_active_library,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_override():
    """Reset in-memory override before and after each test."""
    _active_library_override.clear()
    yield
    _active_library_override.clear()


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    """Redirect config reads/writes to a temp directory."""
    import zotero_mcp.client as _client

    path = tmp_path / "config.json"
    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", path)
    return path


# ---------------------------------------------------------------------------
# set_active_library persists
# ---------------------------------------------------------------------------


def test_set_active_library_writes_to_config(config_path):
    set_active_library("5294983", "group")

    data = json.loads(config_path.read_text())
    assert data["default_library"] == {"library_id": "5294983", "library_type": "group"}


def test_set_active_library_preserves_other_config_keys(config_path):
    config_path.write_text(json.dumps({"semantic_search": {"provider": "ollama"}}))

    set_active_library("5294983", "group")

    data = json.loads(config_path.read_text())
    assert data["semantic_search"]["provider"] == "ollama"
    assert data["default_library"]["library_id"] == "5294983"


def test_set_active_library_no_persist_does_not_write(config_path):
    set_active_library("5294983", "group", persist=False)

    assert not config_path.exists()
    assert _active_library_override["library_id"] == "5294983"


# ---------------------------------------------------------------------------
# clear_active_library persists
# ---------------------------------------------------------------------------


def test_clear_active_library_removes_from_config(config_path):
    config_path.write_text(
        json.dumps({"default_library": {"library_id": "5294983", "library_type": "group"}})
    )

    set_active_library("5294983", "group", persist=False)  # populate in-memory
    clear_active_library()

    data = json.loads(config_path.read_text())
    assert "default_library" not in data


def test_clear_active_library_keeps_other_config_keys(config_path):
    config_path.write_text(
        json.dumps({
            "default_library": {"library_id": "5294983", "library_type": "group"},
            "semantic_search": {"provider": "ollama"},
        })
    )

    set_active_library("5294983", "group", persist=False)
    clear_active_library()

    data = json.loads(config_path.read_text())
    assert "default_library" not in data
    assert data["semantic_search"]["provider"] == "ollama"


def test_clear_active_library_noop_when_no_config(config_path):
    clear_active_library()  # must not raise


# ---------------------------------------------------------------------------
# load_library_from_config
# ---------------------------------------------------------------------------


def test_load_library_from_config_populates_override(config_path):
    config_path.write_text(
        json.dumps({"default_library": {"library_id": "5294983", "library_type": "group"}})
    )

    load_library_from_config()

    assert _active_library_override["library_id"] == "5294983"
    assert _active_library_override["library_type"] == "group"


def test_load_library_from_config_noop_when_no_config(config_path):
    load_library_from_config()

    assert _active_library_override == {}


def test_load_library_from_config_noop_when_key_missing(config_path):
    config_path.write_text(json.dumps({"semantic_search": {}}))

    load_library_from_config()

    assert _active_library_override == {}


def test_env_var_takes_precedence_over_config(config_path, monkeypatch):
    config_path.write_text(
        json.dumps({"default_library": {"library_id": "5294983", "library_type": "group"}})
    )
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "9999999")

    load_library_from_config()

    # Config must not have overwritten the env var — override stays empty.
    assert _active_library_override == {}


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


def test_cli_set_library_writes_config(tmp_path, monkeypatch):
    import zotero_mcp.client as _client

    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", config_path)

    from zotero_mcp.cli import main

    monkeypatch.setattr(
        "sys.argv",
        ["zotero-mcp", "set-library", "--library-id", "5294983", "--library-type", "group"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0

    data = json.loads(config_path.read_text())
    assert data["default_library"] == {"library_id": "5294983", "library_type": "group"}


def test_cli_set_library_personal_flag(tmp_path, monkeypatch):
    import zotero_mcp.client as _client

    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", config_path)

    from zotero_mcp.cli import main

    monkeypatch.setattr("sys.argv", ["zotero-mcp", "set-library", "--personal"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0

    data = json.loads(config_path.read_text())
    assert data["default_library"] == {"library_id": "0", "library_type": "user"}


def test_cli_clear_library_removes_key(tmp_path, monkeypatch):
    import zotero_mcp.client as _client

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"default_library": {"library_id": "5294983", "library_type": "group"}})
    )
    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", config_path)

    from zotero_mcp.cli import main

    monkeypatch.setattr("sys.argv", ["zotero-mcp", "clear-library"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0

    data = json.loads(config_path.read_text())
    assert "default_library" not in data


def test_cli_set_library_missing_args_exits_nonzero(tmp_path, monkeypatch):
    import zotero_mcp.client as _client

    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", tmp_path / "config.json")

    from zotero_mcp.cli import main

    monkeypatch.setattr("sys.argv", ["zotero-mcp", "set-library"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0
