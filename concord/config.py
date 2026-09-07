import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_DIR = Path(os.environ.get("CONCORD_CACHE") or DATA / "cache")
WORK_DIR = Path(os.environ.get("CONCORD_WORK") or DATA / "work")
DB_PATH = Path(os.environ.get("CONCORD_DB") or DATA / "db" / "concord.sqlite")

LLM_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
LLM_MODEL = os.environ.get("CONCORD_LLM_MODEL", "gemini-3.6-flash")
LLM_ENDPOINT = os.environ.get(
    "CONCORD_LLM_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models",
)

EMBED_MODEL = os.environ.get("CONCORD_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Offline mode serves everything from committed cache artifacts and refuses to
# make a network call, so a reviewer can evaluate without an API key.
OFFLINE = os.environ.get("CONCORD_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def ensure_dirs() -> None:
    for path in (CACHE_DIR, WORK_DIR, DB_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
