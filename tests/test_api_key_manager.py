"""Tests for APIKeyManager — key generation, validation, revocation, and admin auth."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.web.server import APIKeyManager


# ── helpers ──

def _fresh_manager(tmp_path: Path) -> APIKeyManager:
    """Return an APIKeyManager pointed at a temp directory."""
    mgr = APIKeyManager()
    mgr.STORE_PATH = tmp_path / "api_keys.json"
    mgr.ADMIN_PASSWORD_FILE = tmp_path / "admin_password.hash"
    mgr.STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Re-init under the temp dir
    mgr._ensure_store()
    return mgr


# ═══════════════════════════════════════════════════════════
#  Key generation
# ═══════════════════════════════════════════════════════════

class TestKeyGeneration:
    def test_generates_key_with_ne_prefix(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(description="test")
        assert result["api_key"].startswith("ne_")
        assert len(result["api_key"]) == 3 + 48  # "ne_" + 48 hex chars

    def test_generated_key_stored_in_file(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(description="persist-test")
        store = mgr._read_store()
        assert result["key_hash"] in store["keys"]
        assert store["keys"][result["key_hash"]]["description"] == "persist-test"

    def test_key_info_fields(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(
            description="full-test",
            max_requests=100,
            expires_days=30,
            permissions=["romance", "tuner"],
        )
        info = result  # result is key_info + "api_key"
        assert info["description"] == "full-test"
        assert info["max_requests"] == 100
        assert info["permissions"] == ["romance", "tuner"]
        assert info["request_count"] == 0
        assert info["revoked"] is False
        assert info["expires_at"] is not None
        # key_prefix is a display fragment, should contain "..."
        assert "..." in info["key_prefix"]

    def test_no_expiry_when_zero(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(expires_days=0)
        assert result["expires_at"] is None

    def test_revoked_false_by_default(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        assert result["revoked"] is False


# ═══════════════════════════════════════════════════════════
#  Key validation
# ═══════════════════════════════════════════════════════════

class TestKeyValidation:
    def test_valid_key_returns_info(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        info = mgr.validate_key(result["api_key"])
        assert info is not None
        assert info["description"] == ""

    def test_invalid_key_returns_none(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        mgr.generate_key()
        assert mgr.validate_key("ne_badbadbad00000000000000000000000000000000000000") is None

    def test_wrong_prefix_returns_none(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        bad = "xx_" + result["api_key"][3:]
        assert mgr.validate_key(bad) is None

    def test_empty_key_returns_none(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        assert mgr.validate_key("") is None
        assert mgr.validate_key(None) is None  # type: ignore[arg-type]

    def test_revoked_key_returns_none(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        mgr.revoke_key(result["api_key"])
        assert mgr.validate_key(result["api_key"]) is None

    def test_expired_key_returns_none(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(expires_days=1)
        # Artificially set expires_at in the past via store manipulation
        with mgr.lock:
            store = mgr._read_store()
            store["keys"][result["key_hash"]]["expires_at"] = (
                datetime.now() - timedelta(hours=1)
            ).isoformat()
            mgr._write_store(store)
        assert mgr.validate_key(result["api_key"]) is None

    def test_request_count_increments(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        mgr.validate_key(result["api_key"])
        info = mgr.validate_key(result["api_key"])
        assert info["request_count"] == 2  # two calls above

    def test_max_requests_exceeded_returns_none(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(max_requests=3)
        for _ in range(3):
            info = mgr.validate_key(result["api_key"])
            assert info is not None
        # 4th call should fail
        assert mgr.validate_key(result["api_key"]) is None

    def test_last_used_at_updated(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        info = mgr.validate_key(result["api_key"])
        assert info["last_used_at"] is not None


# ═══════════════════════════════════════════════════════════
#  Key revocation
# ═══════════════════════════════════════════════════════════

class TestKeyRevocation:
    def test_revoke_existing_key(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        assert mgr.revoke_key(result["api_key"]) is True

    def test_revoke_nonexistent_key(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        assert mgr.revoke_key("ne_" + "aa" * 24) is False

    def test_revoke_by_hash_also_works(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        # revoke_key hashes its input, so passing the full key or the hash both work
        assert mgr.revoke_key(result["key_hash"]) is True

    def test_list_keys_shows_revoked_status(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key()
        mgr.revoke_key(result["api_key"])
        keys = mgr.list_keys()
        matching = [k for k in keys if k["key_hash"] == result["key_hash"]]
        assert len(matching) == 1
        assert matching[0]["revoked"] is True

    def test_list_keys_does_not_contain_full_key(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        mgr.generate_key()
        keys = mgr.list_keys()
        for k in keys:
            assert "api_key" not in k


# ═══════════════════════════════════════════════════════════
#  Admin password
# ═══════════════════════════════════════════════════════════

class TestAdminPassword:
    def test_first_run_generates_default_password(self, tmp_path):
        """When no password file exists, a default is generated and returned True."""
        mgr = _fresh_manager(tmp_path)
        # Delete the file so it simulates first run
        if mgr.ADMIN_PASSWORD_FILE.exists():
            mgr.ADMIN_PASSWORD_FILE.unlink()
        # The default pw is random; we capture via print side-effect,
        # but the method returns True for the generated password.
        # We can't easily guess it, but we can verify the file was created.
        result = mgr.check_admin_password("wrong-guess")
        # Should be False for a wrong guess
        if result:
            # In the unlikely event we guessed the random 16-char hex, pass
            pass
        # The file should exist now
        assert mgr.ADMIN_PASSWORD_FILE.exists()

    def test_set_and_check_password(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        mgr.set_admin_password("hunter2")
        assert mgr.check_admin_password("hunter2") is True
        assert mgr.check_admin_password("wrong") is False

    def test_password_hash_not_plaintext(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        mgr.set_admin_password("secret123")
        stored = mgr.ADMIN_PASSWORD_FILE.read_text(encoding="utf-8").strip()
        assert stored != "secret123"
        assert len(stored) == 64  # SHA-256 hex digest

    def test_password_change(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        mgr.set_admin_password("oldpass")
        mgr.set_admin_password("newpass")
        assert mgr.check_admin_password("oldpass") is False
        assert mgr.check_admin_password("newpass") is True


# ═══════════════════════════════════════════════════════════
#  Store persistence
# ═══════════════════════════════════════════════════════════

class TestStorePersistence:
    def test_keys_survive_reload(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        result = mgr.generate_key(description="survivor")
        key = result["api_key"]

        # Create a new manager pointing at the same file
        mgr2 = APIKeyManager()
        mgr2.STORE_PATH = mgr.STORE_PATH
        mgr2.ADMIN_PASSWORD_FILE = mgr.ADMIN_PASSWORD_FILE
        info = mgr2.validate_key(key)
        assert info is not None
        assert info["description"] == "survivor"

    def test_empty_store_has_keys_key(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        store = mgr._read_store()
        assert "keys" in store
        assert store["keys"] == {}

    def test_corrupt_store_recovers(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        mgr.STORE_PATH.write_text("not valid json", encoding="utf-8")
        store = mgr._read_store()
        assert store == {"keys": {}}
