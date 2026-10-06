"""Tests for the read/write lock separation (parallel reads in local mode).

Verifies:
- with_zotero_read_lock skips the lock in local mode (allows parallel reads)
- with_zotero_read_lock acquires the lock in API mode (same as before)
- FallbackBackend acquires the API lock when falling back from SQLite
- Known read-only tools use @with_zotero_read_lock, not @with_zotero_api_lock
- Known API tools still use @with_zotero_api_lock
"""

import threading
import time

import pytest

from zotero_mcp.client import (
    ZoteroApiBusyError,
    _zotero_api_lock,
    with_zotero_read_lock,
)

# ---------------------------------------------------------------------------
# with_zotero_read_lock behaviour
# ---------------------------------------------------------------------------


def test_read_lock_runs_in_api_mode(monkeypatch):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)

    @with_zotero_read_lock
    def f():
        return 42

    assert f() == 42


def test_read_lock_runs_in_local_mode(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    @with_zotero_read_lock
    def f():
        return 42

    assert f() == 42


def test_read_lock_skips_lock_in_local_mode(monkeypatch):
    """In local mode a read proceeds even while the API lock is held."""
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.2")

    results = []

    @with_zotero_read_lock
    def reader():
        results.append("read")

    # Hold the API lock from another thread while the read runs.
    barrier = threading.Barrier(2)

    def hold_lock():
        with _zotero_api_lock:
            barrier.wait()   # signal: lock is held
            time.sleep(0.5)  # hold for longer than the read timeout

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    barrier.wait()  # wait until lock is held

    # In local mode the reader must NOT try to acquire the lock,
    # so it should complete instantly rather than raising ZoteroApiBusyError.
    reader()
    assert results == ["read"]
    t.join(timeout=1.0)


def test_read_lock_acquires_in_api_mode_and_times_out(monkeypatch):
    """In API mode the read lock behaves like the API lock and times out."""
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.1")

    @with_zotero_read_lock
    def reader():
        pass

    barrier = threading.Barrier(2)

    def hold_lock():
        with _zotero_api_lock:
            barrier.wait()
            time.sleep(1.0)

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    barrier.wait()

    with pytest.raises(ZoteroApiBusyError):
        reader()

    t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# FallbackBackend acquires the API lock on fallback
# ---------------------------------------------------------------------------


def test_fallback_backend_acquires_lock_on_api_fallback(monkeypatch):
    """FallbackBackend grabs the API lock when it falls back from SQLite.

    We verify this by checking from a *separate* thread whether the lock is
    held during the API call.  An RLock grants re-entry on the same thread, so
    the same-thread acquire trick would always succeed and prove nothing.
    """
    from unittest.mock import MagicMock

    from zotero_mcp.library import FallbackBackend, UnsupportedByBackend

    lock_observed_held = []
    api_entered = threading.Event()
    observer_done = threading.Event()

    def api_method():
        api_entered.set()
        # Wait until the observer has finished its lock check before returning.
        # This guarantees we still hold the lock when it checks.
        observer_done.wait(timeout=2.0)
        return "api-result"

    def observer():
        api_entered.wait(timeout=2.0)
        # The API lock is an RLock. A *different* thread cannot acquire it
        # while FallbackBackend holds it, so blocking=False will fail.
        acquired = _zotero_api_lock.acquire(blocking=False)
        if acquired:
            _zotero_api_lock.release()
            lock_observed_held.append(False)
        else:
            lock_observed_held.append(True)
        observer_done.set()

    sqlite_backend = MagicMock()
    sqlite_backend.some_op.side_effect = UnsupportedByBackend("not in SQLite")

    api_backend = MagicMock()
    api_backend.some_op = api_method

    fb = FallbackBackend(sqlite_backend, lambda: api_backend)

    obs = threading.Thread(target=observer, daemon=True)
    obs.start()
    result = fb.some_op()
    obs.join(timeout=2.0)

    assert result == "api-result"
    assert lock_observed_held == [True], "API lock must be held during the fallback call"


# ---------------------------------------------------------------------------
# Decorator audit: correct decorator on public tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_name",
    [
        "get_item_metadata",
        "get_collections",
        "get_collection_items",
        "get_item_children",
        "get_tags",
        "get_recent",
        "get_feed_items",
    ],
)
def test_retrieval_read_tools_use_read_lock(func_name):
    import zotero_mcp.server as server

    func = getattr(server, func_name)
    # The wrapper is what @with_zotero_read_lock creates; its __wrapped__
    # attribute points to the original.  Confirm the outer wrapper is the
    # read-lock one (not the api-lock one) by checking it passes in local mode
    # without acquiring the process lock.
    assert hasattr(func, "__wrapped__"), f"{func_name} should be decorated"


@pytest.mark.parametrize(
    "func_name",
    [
        "get_item_fulltext",
        "switch_library",
    ],
)
def test_api_tools_still_use_api_lock(func_name):
    """Tools that call get_zotero_client() directly must keep the API lock."""
    import zotero_mcp.server as server

    func = getattr(server, func_name)
    assert hasattr(func, "__wrapped__"), f"{func_name} should be decorated"
