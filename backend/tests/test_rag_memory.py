"""Unit tests for RAG using a deterministic fake embedder (no model download)."""
import math

import pytest

from app.memory.rag import RAGMemoryProvider
from app.memory.vector_store import SQLiteVectorStore


class FakeEmbedder:
    """Tiny deterministic vectors to test retrieval plumbing without ML packages."""
    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "python" in lower or "programming language" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "music" in lower or "song" in lower:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


@pytest.fixture
def provider(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag_test.db")
    return RAGMemoryProvider(
        embedder=FakeEmbedder(),
        vector_store=store,
        min_score=0.25,
    )


@pytest.mark.asyncio
async def test_semantic_memory_retrieval(provider):
    await provider.remember(
        "user-a", "My favorite programming language is Python.",
        {"source": "test"},
    )
    results = await provider.retrieve(
        "user-a", "Which programming language do I like?", limit=3
    )
    assert len(results) == 1
    assert "Python" in results[0].text
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata["source"] == "test"


@pytest.mark.asyncio
async def test_user_memories_are_isolated(provider):
    await provider.remember("user-a", "Python is my preferred language.")
    assert await provider.retrieve("user-b", "Python programming language") == []


@pytest.mark.asyncio
async def test_duplicate_memory_updates_metadata(provider):
    await provider.remember("user-a", "Python is my preferred language.", {"v": 1})
    await provider.remember("user-a", "Python is my preferred language.", {"v": 2})
    results = await provider.retrieve("user-a", "Python language")
    assert len(results) == 1
    assert results[0].metadata["v"] == 2


@pytest.mark.asyncio
async def test_irrelevant_memory_is_filtered(provider):
    await provider.remember("user-a", "I like music and songs.")
    results = await provider.retrieve("user-a", "Python programming language")
    assert results == []


@pytest.mark.asyncio
async def test_empty_values_and_bad_limit_are_safe(provider):
    await provider.remember(" ", "some fact")
    await provider.remember("user-a", " ")
    assert await provider.retrieve("user-a", "", limit=3) == []
    assert await provider.retrieve("user-a", "Python", limit=0) == []


def test_vector_store_persists_between_instances(tmp_path):
    path = tmp_path / "persistent.db"
    store1 = SQLiteVectorStore(path)
    store1.upsert("user-a", "Python is my language.", [1.0, 0.0], {"source": "test"})
    store2 = SQLiteVectorStore(path)
    matches = store2.search("user-a", [1.0, 0.0], limit=1, min_score=0.25)
    assert len(matches) == 1
    assert matches[0]["text"] == "Python is my language."
