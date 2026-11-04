"""OKF (Open Knowledge Format) memory provider."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from app.memory.base import MemoryItem
from app.memory.okf_parser import OKFParser
from app.memory.okf_store import OKFStore


class OKFMemoryProvider:
    """Open Knowledge Format (OKF) structured memory provider.

    Implements the MemoryProvider protocol to integrate with the
    benchmark chatbot architecture, backed by regex extraction and
    SQLite persistence.
    """

    def __init__(
        self,
        store: OKFStore | None = None,
        parser: OKFParser | None = None,
        db_path: str | Path | None = None,
    ):
        self.store = store or OKFStore(
            db_path or os.getenv("OKF_DB_PATH") or None
        )
        self.parser = parser or OKFParser()

    @staticmethod
    def _to_memory_item(item: dict[str, Any]) -> MemoryItem:
        metadata = {
            **item.get("metadata", {}),
            "memory_id": item["id"],
            "category": item["category"],
            "created_at": item["created_at"],
        }
        return MemoryItem(
            text=item["text"],
            score=float(item["score"]) if "score" in item else None,
            metadata=metadata,
        )

    async def remember(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Parse structured facts from text and persist in the OKF store.

        If no structured fact can be extracted (e.g. conversational chit-chat,
        greetings, or non-facts), the message is safely ignored to prevent
        long-term memory pollution.
        """
        user_id = user_id.strip()
        text = text.strip()
        if not user_id or not text:
            return

        parsed = self.parser.parse_fact(text)
        # Avoid conversational noise: do not save messages where no structured fact can be extracted
        if not parsed:
            return

        base_meta = dict(metadata or {})
        category = parsed.category
        stored_text = parsed.text
        final_metadata = {
            **base_meta,
            **parsed.metadata,
        }

        await asyncio.to_thread(
            self.store.upsert,
            user_id=user_id,
            text=stored_text,
            category=category,
            metadata=final_metadata,
        )

    async def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve relevant memories matching query keywords."""
        user_id = user_id.strip()
        query = query.strip()
        if not user_id or not query or limit <= 0:
            return []

        matches = await asyncio.to_thread(
            self.store.search,
            user_id=user_id,
            query=query,
            limit=limit,
        )

        return [self._to_memory_item(item) for item in matches]

    async def list_memories(
        self,
        user_id: str,
        category: str | None = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """List stored memories for a user, optionally filtered by category."""
        user_id = user_id.strip()
        if not user_id or limit <= 0:
            return []

        items = await asyncio.to_thread(
            self.store.list_memories,
            user_id=user_id,
            category=category,
            limit=limit,
        )

        return [self._to_memory_item(item) for item in items]
