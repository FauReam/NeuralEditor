"""JSON-based persistent storage with atomic writes."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


class JSONStorage:
    """Atomic JSON read/write with backup support."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base_dir / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, key: str, data: dict[str, Any]) -> None:
        path = self._path(key)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=self.base_dir,
            suffix=".tmp",
        )
        try:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.close()
            shutil.move(tmp.name, path)
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_keys(self) -> list[str]:
        return sorted([
            p.stem for p in self.base_dir.glob("*.json")
            if p.is_file()
        ])
