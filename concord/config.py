import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_DIR = Path(os.environ.get("CONCORD_CACHE") or DATA / "cache")
WORK_DIR = Path(os.environ.get("CONCORD_WORK") or DATA / "work")
DB_PATH = Path(os.environ.get("CONCORD_DB") or DATA / "db" / "concord.sqlite")

def _keys() -> list[str]:
    """Read one or more API keys. A pool only helps for keys on separate
    projects, since the free-tier quota is per project, not per key."""
    raw = os.environ.get("GEMINI_API_KEYS") or ""
    pool = [k.strip() for k in raw.split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if single and single not in pool:
        pool.append(single)
    return pool


LLM_API_KEYS = _keys()
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else None
LLM_MODEL = os.environ.get("CONCORD_LLM_MODEL", "gemini-3.6-flash")
LLM_ENDPOINT = os.environ.get(
    "CONCORD_LLM_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models",
)

EMBED_MODEL = os.environ.get("CONCORD_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Passages per LLM request. The binding free-tier limit is requests per minute,
# not tokens, so batching is what makes a rate-limited key usable.
BATCH_SIZE = int(os.environ.get("CONCORD_BATCH_SIZE", "8"))

# Offline mode serves everything from committed cache artifacts and refuses to
# make a network call, so a reviewer can evaluate without an API key.
OFFLINE = os.environ.get("CONCORD_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def ensure_dirs() -> None:
    for path in (CACHE_DIR, WORK_DIR, DB_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
