import sqlite3
from contextlib import contextmanager
from pathlib import Path

from concord import config

SCHEMA = Path(__file__).with_name("schema.sql")


# Columns added after the first ledger was committed. `CREATE TABLE IF NOT
# EXISTS` cannot widen a table that already exists, so a file written by an
# earlier version would keep its old shape and every read of a new column would
# fail. Adding them here keeps `schema.sql` the single description of the
# database while letting an existing ledger catch up in place.
LATER_COLUMNS = (
    ("facts", "embedding_model", "TEXT"),
    ("relations", "judged", "INTEGER NOT NULL DEFAULT 0"),
)


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any column `schema.sql` has grown since this file was written."""
    added = []
    for table, column, spec in LATER_COLUMNS:
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")
            added.append(f"{table}.{column}")
    return added


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    migrate(conn)
    return conn


@contextmanager
def session(db_path: Path | None = None):
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
