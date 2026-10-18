
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MemoryItem:
    """A memory returned by any memory implementation."""

    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryProvider(Protocol):
    """Common contract for RAG and OKF memory providers."""

    async def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve memories relevant to a user's query."""
        ...

    async def remember(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory for later retrieval."""
        ...


class EmptyMemoryProvider:
    """Temporary provider used until RAG and OKF are integrated."""

    async def retrieve(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryItem]:
        return []

    async def remember(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None
