
"""Persistent SQLite-backed vector store for RAG memories."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "rag_memory.db"


class SQLiteVectorStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, text)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_memories_user
                ON rag_memories(user_id)
            """)

    def upsert(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO rag_memories
                    (user_id, text, embedding, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, text) DO UPDATE SET
                    embedding = excluded.embedding,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_id,
                text,
                json.dumps(embedding),
                json.dumps(metadata or {}),
            ))

    @staticmethod
    def _cosine_normalized(
        left: list[float],
        right: list[float],
    ) -> float:
        if len(left) != len(right) or not left:
            return 0.0

        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return sum(
            a * b for a, b in zip(left, right)
        ) / (left_norm * right_norm)

    @staticmethod
    def _row_to_memory(row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        if not user_id.strip() or limit <= 0 or not query_embedding:
            return []

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, text, embedding, metadata,
                       created_at, updated_at
                FROM rag_memories
                WHERE user_id = ?
            """, (user_id,)).fetchall()

        ranked = []

        for row in rows:
            stored_embedding = json.loads(row["embedding"])
            score = self._cosine_normalized(
                query_embedding,
                stored_embedding,
            )

            if score >= min_score:
                item = self._row_to_memory(row)
                item["score"] = float(score)
                ranked.append(item)

        ranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:limit]

    def list_memories(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the user's most recently updated memories."""
        if not user_id.strip() or limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, text, metadata, created_at, updated_at
                FROM rag_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def count(self, user_id: str | None = None) -> int:
        with self._connect() as conn:
            if user_id is None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM rag_memories"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM rag_memories "
                    "WHERE user_id = ?",
                    (user_id,),
                ).fetchone()

        return int(row["n"])
