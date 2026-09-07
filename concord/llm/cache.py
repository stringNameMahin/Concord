import hashlib
import json
from pathlib import Path
from typing import Any

from concord import config


class MissingFromCache(RuntimeError):
    pass


def request_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


class DiskCache:
    """Content-addressed cache of LLM responses.

    Committed to the repo so a reviewer can run the whole pipeline offline, and
    so reruns during development are free.
    """

    def __init__(self, root: Path | None = None, namespace: str = "llm"):
        self.root = Path(root or config.CACHE_DIR) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def put(self, key: str, payload: dict[str, Any], response: Any) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "request": payload, "response": response}
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )

    def stats(self) -> dict[str, int]:
        entries = list(self.root.rglob("*.json"))
        return {
            "entries": len(entries),
            "bytes": sum(p.stat().st_size for p in entries),
        }
