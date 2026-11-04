"""Persistent SQLite store for Open Knowledge Format (OKF) memories."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "okf_memory.db"

# Single-value attributes where a new statement should replace previous values
SINGLE_VALUE_ATTRIBUTES = {
    "name",
    "location",
    "major",
    "job",
    "work",
    "profession",
    "field_of_study",
}

# Common English stopwords to ignore during keyword retrieval scoring
STOPWORDS = {
    "a", "about", "am", "an", "and", "are", "as", "at", "be", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "him", "his", "how", "i", "if", "in", "is", "it", "its", "me",
    "my", "no", "not", "of", "on", "or", "our", "she", "so", "some", "tell",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def is_single_value_attribute(attribute: str | None) -> bool:
    """Return True if the attribute represents a single value (e.g. location, favorite_X)."""
    if not attribute:
        return False
    attr = attribute.lower().strip()
    return attr in SINGLE_VALUE_ATTRIBUTES or attr.startswith("favorite_")


class OKFStore:
    """Manages persistent SQLite storage for OKF structured memories."""

    def __init__(self, db_path: str | Path | None = None, timeout: float = 30.0):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.timeout = timeout
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that opens, configures, commits, and reliably closes a SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        except sqlite3.DatabaseError:
            pass

        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        """Create the okf_memories table and indexes if they do not exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS okf_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, text)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_okf_memories_user
                ON okf_memories(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_okf_memories_category
                ON okf_memories(user_id, category)
            """)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "category": row["category"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
        }

    def upsert(
        self,
        user_id: str,
        text: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a memory record for a user.

        For single-value attributes (like location or name), replaces conflicting
        earlier facts so memory remains consistent and non-conflicting.
        """
        user_id = user_id.strip()
        text = text.strip()
        category = category.strip().lower() or "general"

        if not user_id or not text:
            return

        meta = dict(metadata or {})
        attribute = meta.get("attribute")

        with self._connect() as conn:
            # Attribute-aware replacement: Remove older conflicting values for single-value attributes
            if is_single_value_attribute(attribute):
                conn.execute(
                    """
                    DELETE FROM okf_memories
                    WHERE user_id = ?
                      AND category = ?
                      AND json_extract(metadata, '$.attribute') = ?
                      AND text != ?
                    """,
                    (user_id, category, attribute, text),
                )

            # Insert or update memory record
            conn.execute(
                """
                INSERT INTO okf_memories (user_id, category, text, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, text) DO UPDATE SET
                    category = excluded.category,
                    metadata = excluded.metadata
                """,
                (
                    user_id,
                    category,
                    text,
                    json.dumps(meta),
                ),
            )

    def list_memories(
        self,
        user_id: str,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List stored memories for a user, optionally filtered by category."""
        user_id = user_id.strip()
        if not user_id or limit <= 0:
            return []

        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    """
                    SELECT id, user_id, category, text, metadata, created_at
                    FROM okf_memories
                    WHERE user_id = ? AND category = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, category.strip().lower(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, user_id, category, text, metadata, created_at
                    FROM okf_memories
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def search(
        self,
        user_id: str,
        query: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve memories relevant to the query keywords for a user.

        Filters stopwords, matches against text, category, attribute and value,
        and returns results ranked by relevance score in descending order.
        """
        user_id = user_id.strip()
        query = query.strip()
        if not user_id or not query or limit <= 0:
            return []

        memories = self.list_memories(user_id=user_id, category=category, limit=500)
        raw_tokens = set(re.findall(r"\w+", query.lower()))
        if not raw_tokens:
            return []

        # Remove stopwords; fall back to all tokens if query consists only of stopwords
        query_words = {w for w in raw_tokens if w not in STOPWORDS} or raw_tokens

        scored: list[tuple[float, dict[str, Any]]] = []
        for mem in memories:
            text_words = set(re.findall(r"\w+", mem["text"].lower()))
            score = 0.0

            # 1. Text content overlap
            overlap = len(query_words & text_words)
            score += float(overlap)

            meta = mem.get("metadata", {})

            # 2. Match entity value (+3 points)
            val = str(meta.get("value", "")).lower()
            val_tokens = set(re.findall(r"\w+", val))
            if query_words & val_tokens:
                score += 3.0

            # 3. Match attribute name (+2 points, e.g. 'location', 'name')
            attr = str(meta.get("attribute", "")).lower()
            attr_tokens = set(re.findall(r"\w+", attr))
            if query_words & attr_tokens:
                score += 2.0

            # 4. Match category (+1.5 points, e.g. 'preference', 'education')
            cat = str(mem.get("category", "")).lower()
            if cat in query_words:
                score += 1.5

            if score > 0:
                item = dict(mem)
                item["score"] = round(score, 2)
                scored.append((score, item))

        # Sort in descending relevance order
        scored.sort(key=lambda item: item[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    def count(self, user_id: str | None = None) -> int:
        """Count total stored memories or memories for a specific user."""
        with self._connect() as conn:
            if user_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM okf_memories").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM okf_memories WHERE user_id = ?",
                    (user_id.strip(),),
                ).fetchone()

        return int(row["n"])
