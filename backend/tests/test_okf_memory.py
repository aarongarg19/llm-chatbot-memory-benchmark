"""Unit tests for the OKF structured memory system covering all 5 fixes."""
from __future__ import annotations

import pytest

from app.memory.factory import create_memory_provider
from app.memory.okf import OKFMemoryProvider
from app.memory.okf_parser import OKFParser, parse_fact
from app.memory.okf_store import OKFStore


@pytest.fixture
def okf_store(tmp_path):
    test_db = tmp_path / "test_okf.db"
    return OKFStore(test_db)


@pytest.fixture
def provider(okf_store):
    return OKFMemoryProvider(store=okf_store)


# -----------------------------------------------------------------------------
# Fix 1: Attribute-Aware Updating
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attribute_aware_updating_location(provider, okf_store):
    """Updating a single-value attribute (location) replaces the old fact."""
    await provider.remember("user-1", "I live in Hyderabad.")
    results = await provider.retrieve("user-1", "Where do I live?")
    assert len(results) == 1
    assert "Hyderabad" in results[0].text

    # User updates their location
    await provider.remember("user-1", "I live in Bangalore.")
    updated = await provider.retrieve("user-1", "Where do I live?")

    # Only Bangalore must be returned, not Hyderabad
    assert len(updated) == 1
    assert "Bangalore" in updated[0].text
    assert okf_store.count("user-1") == 1


@pytest.mark.asyncio
async def test_attribute_aware_updating_favorite(provider, okf_store):
    """Updating a favorite attribute replaces the prior preference."""
    await provider.remember("user-1", "My favorite language is Python.")
    await provider.remember("user-1", "My favorite language is Rust.")

    results = await provider.retrieve("user-1", "What is my favorite language?")
    assert len(results) == 1
    assert "Rust" in results[0].text
    assert okf_store.count("user-1") == 1


@pytest.mark.asyncio
async def test_distinct_attributes_coexist(provider, okf_store):
    """Facts with different attributes or categories are preserved together."""
    await provider.remember("user-1", "My name is Rahul.")
    await provider.remember("user-1", "I live in Bangalore.")
    await provider.remember("user-1", "My favorite language is Python.")
    await provider.remember("user-1", "I study AI.")

    assert okf_store.count("user-1") == 4
    name_res = await provider.retrieve("user-1", "name")
    assert len(name_res) >= 1
    assert "Rahul" in name_res[0].text


# -----------------------------------------------------------------------------
# Fix 2: Avoid Conversational Noise
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conversational_noise_is_not_saved(provider, okf_store):
    """Greetings, chit-chat, and unparseable messages are safely ignored."""
    noise_messages = [
        "Hello!",
        "Good morning chatbot.",
        "How are you today?",
        "Can you help me write an essay?",
        "I am wondering if it will rain.",
        "What is the capital of France?",
        "I am happy today.",
    ]

    for msg in noise_messages:
        await provider.remember("user-1", msg)

    assert okf_store.count("user-1") == 0
    assert await provider.retrieve("user-1", "hello") == []
    assert await provider.retrieve("user-1", "rain") == []


# -----------------------------------------------------------------------------
# Fix 3: SQLite Connection Management
# -----------------------------------------------------------------------------

def test_sqlite_connection_context_management(tmp_path):
    """Connection manager sets WAL mode, commits, and reliably closes connections."""
    db_file = tmp_path / "conn_test.db"
    store = OKFStore(db_file, timeout=10.0)

    # Verify WAL mode is active
    with store._connect() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

    # Store memory and verify it persists after connection closes
    store.upsert("user-1", "I live in Hyderabad.", "location", {"attribute": "location", "value": "Hyderabad"})
    assert store.count("user-1") == 1

    # Verify a second store instance reads the persisted data cleanly
    store2 = OKFStore(db_file)
    assert store2.count("user-1") == 1


# -----------------------------------------------------------------------------
# Fix 4: Improved Retrieval & Stopword Filtering
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_improved_retrieval_filters_stopwords(provider):
    """Stopwords ('what', 'is', 'my') do not skew retrieval toward unrelated facts."""
    await provider.remember("user-1", "My name is Rahul.")
    await provider.remember("user-1", "My favorite language is Python.")
    await provider.remember("user-1", "I study AI.")

    # Query 'What is my name?' should match the name fact, NOT the Python fact
    results = await provider.retrieve("user-1", "What is my name?")
    assert len(results) >= 1
    assert "Rahul" in results[0].text
    # Ensure the top-ranked item has the highest relevance score
    assert results[0].score is not None and results[0].score > 0


@pytest.mark.asyncio
async def test_retrieval_ranking_by_value_and_attribute(provider):
    """Queries matching specific entity values or attributes receive higher rank."""
    await provider.remember("user-1", "I live in Hyderabad.")
    await provider.remember("user-1", "I study AI.")

    # Direct value search
    results = await provider.retrieve("user-1", "Hyderabad")
    assert len(results) == 1
    assert "Hyderabad" in results[0].text

    # Attribute search
    loc_results = await provider.retrieve("user-1", "location")
    assert len(loc_results) == 1
    assert "Hyderabad" in loc_results[0].text


# -----------------------------------------------------------------------------
# Fix 5: Improved Name Extraction
# -----------------------------------------------------------------------------

def test_name_extraction_supports_i_am():
    """Parser supports 'I am <Name>', 'My name is <Name>', and 'Call me <Name>'."""
    cases = [
        ("My name is Rahul.", "Rahul"),
        ("Call me Rahul.", "Rahul"),
        ("I am Rahul.", "Rahul"),
        ("I am Rahul Sharma.", "Rahul Sharma"),
        ("I'm Alice.", "Alice"),
    ]

    for sentence, expected_name in cases:
        fact = parse_fact(sentence)
        assert fact is not None, f"Failed to parse: {sentence}"
        assert fact.category == "identity"
        assert fact.attribute == "name"
        assert fact.value == expected_name


def test_name_extraction_rejects_non_names():
    """Sentences starting with 'I am <verb/phrase>' are not falsely treated as names."""
    invalid_cases = [
        "I am learning Python.",
        "I am studying AI.",
        "I am working on a project.",
        "I am wondering if it will rain.",
        "I am a software engineer.",
        "I am tired.",
        "I am happy.",
        "I am based in Hyderabad.",
    ]

    for sentence in invalid_cases:
        fact = parse_fact(sentence)
        if fact is not None:
            # If it matches something, it must NOT be a person's name!
            assert fact.attribute != "name", f"False name extraction on: {sentence}"


# -----------------------------------------------------------------------------
# Isolation & Factory Integration
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_memory_isolation(provider):
    """User A's memories are never accessible to User B."""
    await provider.remember("user-a", "My favorite language is Python.")

    res_a = await provider.retrieve("user-a", "Python")
    res_b = await provider.retrieve("user-b", "Python")

    assert len(res_a) == 1
    assert res_b == []


def test_factory_creates_okf_provider(monkeypatch, tmp_path):
    """Factory returns OKFMemoryProvider when MEMORY_MODE=okf."""
    monkeypatch.setenv("MEMORY_MODE", "okf")
    monkeypatch.setenv("OKF_DB_PATH", str(tmp_path / "factory_okf.db"))

    provider = create_memory_provider()
    assert isinstance(provider, OKFMemoryProvider)
