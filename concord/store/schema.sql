PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    sha256        TEXT NOT NULL UNIQUE,
    n_pages       INTEGER,
    doc_type      TEXT,
    publisher     TEXT,
    published     TEXT,
    parser        TEXT,
    parser_version TEXT,
    text_path     TEXT,
    doc_context   TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL,
    ingested_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id             TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    claim_text          TEXT NOT NULL,
    subject_surface     TEXT,
    subject_key         TEXT,
    subject_type        TEXT,
    predicate           TEXT NOT NULL,
    predicate_canonical TEXT,
    comparison_key      TEXT,
    value_json          TEXT NOT NULL,
    value_kind          TEXT,
    value_normalized    REAL,
    qualifiers_json     TEXT NOT NULL DEFAULT '{}',
    evidence_json       TEXT NOT NULL,
    align_status        TEXT NOT NULL,
    page                INTEGER,
    confidence          REAL,
    flags_json          TEXT NOT NULL DEFAULT '[]',
    embedding           BLOB,
    extractor           TEXT,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_doc        ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_cmpkey     ON facts(comparison_key);
CREATE INDEX IF NOT EXISTS idx_facts_align      ON facts(align_status);
CREATE INDEX IF NOT EXISTS idx_facts_normalized ON facts(value_normalized);

CREATE TABLE IF NOT EXISTS relations (
    relation_id     TEXT PRIMARY KEY,
    fact_a          TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    fact_b          TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    verdict         TEXT NOT NULL,
    rule_fired      TEXT NOT NULL,
    qualifier_key   TEXT,
    explanation     TEXT,
    decided_by      TEXT NOT NULL,
    self_consistent INTEGER,
    cross_document  INTEGER NOT NULL DEFAULT 0,
    blocked_by      TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    UNIQUE (fact_a, fact_b)
);

CREATE INDEX IF NOT EXISTS idx_relations_verdict ON relations(verdict);
CREATE INDEX IF NOT EXISTS idx_relations_a       ON relations(fact_a);
CREATE INDEX IF NOT EXISTS idx_relations_b       ON relations(fact_b);

CREATE TABLE IF NOT EXISTS predicate_registry (
    canonical      TEXT PRIMARY KEY,
    aliases_json   TEXT NOT NULL DEFAULT '[]',
    embedding      BLOB,
    first_seen_doc TEXT,
    count          INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     TEXT,
    event_type TEXT NOT NULL,
    subject    TEXT NOT NULL,
    target     TEXT,
    similarity REAL,
    decided_by TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_doc ON schema_events(doc_id);
