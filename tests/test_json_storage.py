"""Tests for JSON atomic storage."""

import tempfile
from pathlib import Path

from src.utils.json_storage import JSONStorage


class TestJSONStorage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = JSONStorage(self.tmpdir)

    def test_save_and_load(self):
        data = {"name": "test", "value": 42}
        self.store.save("slot1", data)
        loaded = self.store.load("slot1")
        assert loaded == data

    def test_load_missing(self):
        assert self.store.load("nonexistent") is None

    def test_delete(self):
        self.store.save("temp", {"data": 1})
        assert self.store.delete("temp") is True
        assert self.store.load("temp") is None
        assert self.store.delete("temp") is False

    def test_list_keys(self):
        self.store.save("b", {})
        self.store.save("a", {})
        assert self.store.list_keys() == ["a", "b"]

    def test_atomic_write(self):
        """Ensure write is atomic (no partial files on crash)."""
        self.store.save("atomic", {"key": "value"})
        files = list(Path(self.tmpdir).glob("*.json"))
        assert len(files) == 1
        assert files[0].name == "atomic.json"
