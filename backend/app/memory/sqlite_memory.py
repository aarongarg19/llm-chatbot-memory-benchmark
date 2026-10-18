
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.memory.base import MemoryItem


DB_PATH = Path(__file__).resolve().parents[2] / "chat_history.db"


class SQLiteMemoryProvider:
    """Store and retrieve persistent user memories using SQLite."""

    def __init__(self):
        self._initialize_db()

    def _get_connection(self):
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_db(self):
        with self._get_connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, text)
                )
            """)

    async def remember(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        text = text.strip()
        if not user_id.strip() or not text:
            return

        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO user_memories (user_id, text, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, text) DO NOTHING
                """,
                (
                    user_id,
                    text,
                    json.dumps(metadata or {}),
                ),
            )

    async def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryItem]:
        if not user_id.strip() or limit <= 0:
            return []

        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT text, metadata
                FROM user_memories
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 500
                """,
                (user_id,),
            ).fetchall()

        # Simple keyword matching for the baseline implementation.
        query_words = set(query.lower().split())
        scored = []

        for row in rows:
            memory_words = set(row["text"].lower().split())
            score = len(query_words & memory_words)

            if score > 0:
                scored.append(
                    (
                        score,
                        MemoryItem(
                            text=row["text"],
                            score=float(score),
                            metadata=json.loads(row["metadata"]),
                        ),
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in scored[:limit]]
