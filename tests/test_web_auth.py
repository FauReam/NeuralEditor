"""Tests for web authentication helpers — API key extraction and route-level auth."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.web.server import (
    _extract_api_key,
    _authenticate,
    _check_admin,
    _create_admin_session,
    ADMIN_SESSION_TOKENS,
    _admin_sessions_lock,
    api_key_manager,
)


# ── Helpers: minimal mock BaseHTTPRequestHandler ──

class MockHeaders(dict):
    """Case-insensitive dict-like with .get method for header access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build a lower-case lookup
        self._lower = {k.lower(): k for k in self}

    def get(self, key, default=""):
        real_key = self._lower.get(key.lower())
        if real_key is None:
            return default
        return super().get(real_key, default)


def _make_handler(headers: dict | None = None, path: str = "/") -> MagicMock:
    """Build a MagicMock that looks enough like BaseHTTPRequestHandler for auth helpers."""
    h = MagicMock()
    h.headers = MockHeaders(headers or {})
    h.path = path
    return h


# ═══════════════════════════════════════════════════════════
#  _extract_api_key
# ═══════════════════════════════════════════════════════════

class TestExtractApiKey:
    def test_header_x_api_key(self):
        h = _make_handler(headers={"X-API-Key": "ne_abc123"})
        assert _extract_api_key(h) == "ne_abc123"

    def test_header_x_api_key_case_insensitive(self):
        h = _make_handler(headers={"x-api-key": "ne_lowercase"})
        assert _extract_api_key(h) == "ne_lowercase"

    def test_query_param(self):
        h = _make_handler(path="/api/romance/state?api_key=ne_query123")
        assert _extract_api_key(h) == "ne_query123"

    def test_query_param_multiple_values(self):
        h = _make_handler(path="/api/romance/state?api_key=ne_first&api_key=ne_second")
        assert _extract_api_key(h) == "ne_first"

    def test_bearer_token(self):
        h = _make_handler(headers={"Authorization": "Bearer ne_bearer456"})
        assert _extract_api_key(h) == "ne_bearer456"

    def test_header_priority_over_query(self):
        # Header should win when both present
        h = _make_handler(
            headers={"X-API-Key": "ne_header"},
            path="/api?api_key=ne_query",
        )
        assert _extract_api_key(h) == "ne_header"

    def test_no_key_returns_none(self):
        h = _make_handler()
        assert _extract_api_key(h) is None

    def test_empty_header_value(self):
        h = _make_handler(headers={"X-API-Key": ""})
        assert _extract_api_key(h) is None

    def test_header_value_with_whitespace(self):
        h = _make_handler(headers={"X-API-Key": "  ne_spaced  "})
        assert _extract_api_key(h) == "ne_spaced"

    def test_authorization_not_bearer(self):
        h = _make_handler(headers={"Authorization": "Basic something"})
        assert _extract_api_key(h) is None

    def test_query_without_api_key_param(self):
        h = _make_handler(path="/api?other=value")
        assert _extract_api_key(h) is None


# ═══════════════════════════════════════════════════════════
#  _authenticate
# ═══════════════════════════════════════════════════════════

class TestAuthenticate:
    def setup_method(self):
        """Create a temp store so real api_keys.json isn't touched."""
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig_store = api_key_manager.STORE_PATH
        self._orig_pw = api_key_manager.ADMIN_PASSWORD_FILE
        api_key_manager.STORE_PATH = self.tmpdir / "api_keys.json"
        api_key_manager.ADMIN_PASSWORD_FILE = self.tmpdir / "admin_password.hash"
        api_key_manager._ensure_store()

    def teardown_method(self):
        api_key_manager.STORE_PATH = self._orig_store
        api_key_manager.ADMIN_PASSWORD_FILE = self._orig_pw

    def test_authenticate_valid_key(self):
        result = api_key_manager.generate_key(description="auth-test")
        h = _make_handler(headers={"X-API-Key": result["api_key"]})
        info = _authenticate(h)
        assert info is not None
        assert info["description"] == "auth-test"

    def test_authenticate_invalid_key(self):
        h = _make_handler(headers={"X-API-Key": "ne_bad0000000000000000000000000000000000000000000000"})
        assert _authenticate(h) is None

    def test_authenticate_no_key(self):
        h = _make_handler()
        assert _authenticate(h) is None

    def test_authenticate_revoked_key(self):
        result = api_key_manager.generate_key()
        api_key_manager.revoke_key(result["api_key"])
        h = _make_handler(headers={"X-API-Key": result["api_key"]})
        assert _authenticate(h) is None


# ═══════════════════════════════════════════════════════════
#  Admin session helpers
# ═══════════════════════════════════════════════════════════

class TestAdminAuth:
    def test_create_admin_session_returns_token(self):
        token = _create_admin_session()
        assert len(token) == 64  # token_hex(32) → 64 hex chars
        with _admin_sessions_lock:
            assert token in ADMIN_SESSION_TOKENS

    def test_check_admin_with_valid_token(self):
        token = _create_admin_session()
        h = _make_handler(headers={"Cookie": f"admin_token={token}"})
        assert _check_admin(h) is True

    def test_check_admin_with_invalid_token(self):
        h = _make_handler(headers={"Cookie": "admin_token=bad_token"})
        assert _check_admin(h) is False

    def test_check_admin_no_cookie(self):
        h = _make_handler()
        assert _check_admin(h) is False

    def test_check_admin_expired_token(self):
        token = _create_admin_session()
        # Artificially expire it
        with _admin_sessions_lock:
            ADMIN_SESSION_TOKENS[token] = time.time() - 10
        h = _make_handler(headers={"Cookie": f"admin_token={token}"})
        assert _check_admin(h) is False
        # Should be cleaned up
        with _admin_sessions_lock:
            assert token not in ADMIN_SESSION_TOKENS

    def test_check_admin_cookie_with_extra_attrs(self):
        token = _create_admin_session()
        # Real cookies can have extra whitespace variations
        cookie = f"other=val; admin_token={token}; foo=bar"
        h = _make_handler(headers={"Cookie": cookie})
        assert _check_admin(h) is True


# ═══════════════════════════════════════════════════════════
#  HTTP-level auth response (route testing via mock handler)
# ═══════════════════════════════════════════════════════════

class TestHttpAuthResponses:
    """Test that routes return 401 when auth is missing or invalid."""

    def setup_method(self):
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig_store = api_key_manager.STORE_PATH
        self._orig_pw = api_key_manager.ADMIN_PASSWORD_FILE
        api_key_manager.STORE_PATH = self.tmpdir / "api_keys.json"
        api_key_manager.ADMIN_PASSWORD_FILE = self.tmpdir / "admin_password.hash"
        api_key_manager._ensure_store()

    def teardown_method(self):
        api_key_manager.STORE_PATH = self._orig_store
        api_key_manager.ADMIN_PASSWORD_FILE = self._orig_pw

    def _call_get(self, handler_class, path: str, headers: dict | None = None):
        """Instantiate a handler and call do_GET, capturing the JSON response."""
        from io import BytesIO

        h = handler_class.__new__(handler_class)
        h.headers = MockHeaders(headers or {})
        h.path = path
        h.rfile = BytesIO()
        h.wfile = BytesIO()
        h.send_response = MagicMock()
        h.send_header = MagicMock()
        h.end_headers = MagicMock()

        # We need to capture what _json writes
        captured = {}

        def _fake_json(data, code=200, extra_headers=None):
            captured["data"] = data
            captured["code"] = code
            # Write bytes to wfile as real _json does
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            h.wfile.write(body)

        h._json = _fake_json
        h._html = MagicMock()
        h._serve_file = MagicMock()
        h._read = MagicMock(return_value={})
        h.log_message = MagicMock()

        h.do_GET()
        return captured

    def test_auth_status_no_key_returns_401(self):
        from src.web.server import Handler
        # We need to intercept the route; path-based dispatch in do_GET
        # We'll call the handler with a mocked _json and check the result
        captured = self._call_get(Handler, "/api/auth/status")
        assert captured.get("code") == 401
        assert captured["data"]["authenticated"] is False

    def test_auth_status_valid_key(self):
        from src.web.server import Handler
        result = api_key_manager.generate_key(description="web-test")
        captured = self._call_get(
            Handler,
            "/api/auth/status",
            headers={"X-API-Key": result["api_key"]},
        )
        assert captured.get("code") == 200
        assert captured["data"]["authenticated"] is True
        assert captured["data"]["description"] == "web-test"

    def test_admin_keys_without_login_returns_401(self):
        from src.web.server import Handler
        captured = self._call_get(Handler, "/api/admin/keys")
        assert captured.get("code") == 401

    def test_admin_check_no_auth(self):
        from src.web.server import Handler
        captured = self._call_get(Handler, "/api/admin/check")
        assert captured["data"]["authenticated"] is False
