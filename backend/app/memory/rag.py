
"""RAG-backed implementation of the shared MemoryProvider contract."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol

from app.memory.base import MemoryItem
from app.memory.embeddings import SentenceTransformerEmbedder
from app.memory.vector_store import SQLiteVectorStore


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class RAGMemoryProvider:
    """Persist and retrieve user-scoped memories using embeddings."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: SQLiteVectorStore | None = None,
        min_score: float | None = None,
    ):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_store = vector_store or SQLiteVectorStore(
            os.getenv("RAG_DB_PATH") or None
        )
        self.min_score = (
            min_score
            if min_score is not None
            else float(os.getenv("RAG_MIN_SIMILARITY", "0.10"))
        )

    async def remember(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        user_id = user_id.strip()
        text = text.strip()

        if not user_id or not text:
            return

        vectors = await asyncio.to_thread(
            self.embedder.embed,
            [text],
        )

        if not vectors or not vectors[0]:
            return

        await asyncio.to_thread(
            self.vector_store.upsert,
            user_id,
            text,
            vectors[0],
            metadata or {},
        )

    @staticmethod
    def _to_memory_item(
        item: dict[str, Any],
        score: float | None = None,
    ) -> MemoryItem:
        metadata = {
            **item.get("metadata", {}),
            "memory_id": item["id"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }

        return MemoryItem(
            text=item["text"],
            score=score,
            metadata=metadata,
        )

    async def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryItem]:
        user_id = user_id.strip()
        query = query.strip()

        if not user_id or not query or limit <= 0:
            return []

        vectors = await asyncio.to_thread(
            self.embedder.embed,
            [query],
        )

        if not vectors or not vectors[0]:
            return []

        matches = await asyncio.to_thread(
            self.vector_store.search,
            user_id,
            vectors[0],
            limit,
            self.min_score,
        )

        return [
            self._to_memory_item(
                item,
                score=item["score"],
            )
            for item in matches
        ]

    async def list_memories(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[MemoryItem]:
        """List saved memories without semantic similarity filtering."""
        user_id = user_id.strip()

        if not user_id or limit <= 0:
            return []

        items = await asyncio.to_thread(
            self.vector_store.list_memories,
            user_id,
            limit,
        )

        return [
            self._to_memory_item(item)
            for item in items
        ]
