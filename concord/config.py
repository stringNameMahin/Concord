import os
from pathlib import Path

from dotenv import load_dotenv

from concord.llm import providers

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_DIR = Path(os.environ.get("CONCORD_CACHE") or DATA / "cache")
WORK_DIR = Path(os.environ.get("CONCORD_WORK") or DATA / "work")
DB_PATH = Path(os.environ.get("CONCORD_DB") or DATA / "db" / "concord.sqlite")

LLM_MODEL = os.environ.get("CONCORD_LLM_MODEL", "gemini-3.6-flash")

# Inferred from the model name - OpenRouter ids are always `vendor/model` -
# and overridable when that guess is wrong.
LLM_PROVIDER = os.environ.get("CONCORD_LLM_PROVIDER") or providers.for_model(LLM_MODEL)

# An endpoint of None means "whatever this provider's default is", so switching
# providers needs one environment variable rather than two.
LLM_ENDPOINT = os.environ.get("CONCORD_LLM_ENDPOINT") or None

KEY_NAMES = {
    "gemini": ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
}


def _keys(provider: str = "") -> list[str]:
    """Read the keys belonging to one provider.

    A pool of Gemini keys only helps when they sit on separate projects, since
    the free-tier quota is per project rather than per key. The pool exists for
    that case and for a key revoked mid-run, not as a way around a quota.
    """
    plural, *singular = KEY_NAMES.get(provider or LLM_PROVIDER, KEY_NAMES["gemini"])
    pool = [k.strip() for k in (os.environ.get(plural) or "").split(",") if k.strip()]
    for name in singular:
        value = os.environ.get(name)
        if value and value not in pool:
            pool.append(value)
    return pool


LLM_API_KEYS = _keys(LLM_PROVIDER)
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else None

EMBED_MODEL = os.environ.get("CONCORD_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Passages per LLM request. The binding free-tier limit is requests per minute,
# not tokens, so batching is what makes a rate-limited key usable.
BATCH_SIZE = int(os.environ.get("CONCORD_BATCH_SIZE", "8"))

# Batches in flight at once. A per-token provider rewards concurrency; a
# per-minute-rate-limited one does not, so this is tuned per provider rather
# than fixed.
WORKERS = int(os.environ.get("CONCORD_WORKERS", "6"))

# Offline mode serves everything from committed cache artifacts and refuses to
# make a network call, so a reviewer can evaluate without an API key.
OFFLINE = os.environ.get("CONCORD_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def ensure_dirs() -> None:
    for path in (CACHE_DIR, WORK_DIR, DB_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
