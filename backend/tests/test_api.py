
import pytest
from fastapi.testclient import TestClient

from app import database, main
from app.memory import sqlite_memory


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Keep tests isolated from your real database.
    monkeypatch.setattr(
        database,
        "DB_PATH",
        tmp_path / "test_history.db",
    )
    monkeypatch.setattr(
        sqlite_memory,
        "DB_PATH",
        tmp_path / "test_history.db",
    )
    monkeypatch.setenv("MEMORY_MODE", "sqlite")

    with TestClient(main.app) as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["memory_mode"] == "sqlite"


def test_chat_saves_conversation(client, monkeypatch):
    async def fake_generate_response(message, history=None, memories=None):
        return "Hello! This is a test response."

    monkeypatch.setattr(
        main,
        "generate_response",
        fake_generate_response,
    )

    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "user_id": "test-user",
        },
    )

    assert response.status_code == 200

    data = response.json()
    conversation_id = data["conversation_id"]

    assert data["reply"] == "Hello! This is a test response."
    assert data["memory_mode"] == "sqlite"

    history_response = client.get(
        f"/conversations/{conversation_id}/messages"
    )

    assert history_response.status_code == 200

    messages = history_response.json()["messages"]
    assert [item["role"] for item in messages] == [
        "user",
        "assistant",
    ]


def test_unknown_conversation_returns_404(client):
    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "conversation_id": "does-not-exist",
        },
    )

    assert response.status_code == 404


def test_missing_message_is_rejected(client):
    response = client.post("/chat", json={})

    assert response.status_code == 422
