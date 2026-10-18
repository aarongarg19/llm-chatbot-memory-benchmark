
import pytest

from app.memory.base import EmptyMemoryProvider
from app.memory import sqlite_memory
from app.memory.factory import create_memory_provider
from app.memory.sqlite_memory import SQLiteMemoryProvider


@pytest.fixture
def memory_provider(tmp_path, monkeypatch):
    """Use a temporary database so real project data stays untouched."""
    test_db = tmp_path / "test_memory.db"

    monkeypatch.setattr(sqlite_memory, "DB_PATH", test_db)

    return SQLiteMemoryProvider()


@pytest.mark.asyncio
async def test_store_and_retrieve_memory(memory_provider):
    await memory_provider.remember(
        user_id="user-1",
        text="My favorite programming language is Java.",
    )

    results = await memory_provider.retrieve(
        user_id="user-1",
        query="favorite programming language Java",
    )

    assert len(results) == 1
    assert results[0].text == "My favorite programming language is Java."


@pytest.mark.asyncio
async def test_users_memories_are_isolated(memory_provider):
    await memory_provider.remember(
        user_id="user-1",
        text="My favorite language is Java.",
    )

    results = await memory_provider.retrieve(
        user_id="user-2",
        query="favorite language Java",
    )

    assert results == []


@pytest.mark.asyncio
async def test_duplicate_memory_is_not_stored(memory_provider):
    fact = "My favorite language is Java."

    await memory_provider.remember("user-1", fact)
    await memory_provider.remember("user-1", fact)

    results = await memory_provider.retrieve(
        user_id="user-1",
        query="favorite language Java",
    )

    assert len(results) == 1


@pytest.mark.asyncio
async def test_empty_memory_is_ignored(memory_provider):
    await memory_provider.remember("user-1", "   ")

    results = await memory_provider.retrieve(
        user_id="user-1",
        query="anything",
    )

    assert results == []


@pytest.mark.asyncio
async def test_invalid_retrieval_limit_returns_empty_list(memory_provider):
    results = await memory_provider.retrieve(
        user_id="user-1",
        query="Java",
        limit=0,
    )

    assert results == []


@pytest.mark.asyncio
async def test_empty_provider_returns_no_memories():
    provider = EmptyMemoryProvider()

    results = await provider.retrieve(
        user_id="user-1",
        query="Java",
    )

    assert results == []


def test_factory_selects_none_provider(monkeypatch):
    monkeypatch.setenv("MEMORY_MODE", "none")

    provider = create_memory_provider()

    assert isinstance(provider, EmptyMemoryProvider)


def test_factory_selects_sqlite_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_MODE", "sqlite")
    monkeypatch.setattr(sqlite_memory, "DB_PATH", tmp_path / "factory.db")

    provider = create_memory_provider()

    assert isinstance(provider, SQLiteMemoryProvider)


def test_factory_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("MEMORY_MODE", "invalid")

    with pytest.raises(ValueError, match="Invalid MEMORY_MODE"):
        create_memory_provider()
