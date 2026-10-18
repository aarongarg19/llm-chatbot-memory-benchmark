
import uuid
import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.database import (
    init_db,
    create_conversation,
    conversation_exists,
    save_message,
    get_recent_messages,
    get_conversation_messages,
)
from app.llm import generate_response
from app.memory.base import EmptyMemoryProvider, MemoryProvider


app = FastAPI(
    title="LLM Chatbot Memory Benchmark",
    description="Custom chatbot with persistent conversations and pluggable RAG/OKF memory",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary provider until the RAG and OKF implementations are integrated.
memory_provider: MemoryProvider = EmptyMemoryProvider()


@app.on_event("startup")
def startup():
    init_db()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


class MessageResponse(BaseModel):
    conversation_id: str
    messages: list[dict]


@app.get("/")
def home():
    return {
        "project": "LLM Chatbot Memory Benchmark",
        "module": "LLM and Backend",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "memory_mode": os.getenv("MEMORY_MODE", "none").strip().lower()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty or whitespace only.",
        )

    conversation_id = request.conversation_id

    # Create a new conversation when no ID is supplied.
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        create_conversation(conversation_id)

    elif not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    # Load recent conversation history from SQLite.
    history = get_recent_messages(
        conversation_id,
        limit=12,
    )

    # Retrieve long-term memories through the shared interface.
    memories = []

    if request.user_id:
        try:
            retrieved = await memory_provider.retrieve(
                user_id=request.user_id,
                query=message,
                limit=5,
            )
            memories = [item.text for item in retrieved]

        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Memory retrieval failed.",
            ) from exc

    # Save the user message before generating a response.
    save_message(
        conversation_id,
        "user",
        message,
    )

    try:
        reply = await generate_response(
            message,
            history,
            memories,
        )

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Check that it is running.",
        ) from exc

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The LLM returned an invalid response.",
        ) from exc

    # Save the assistant response.
    save_message(
        conversation_id,
        "assistant",
        reply,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
    )


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
)
def read_conversation(conversation_id: str):
    if not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    messages = get_conversation_messages(conversation_id)

    return MessageResponse(
        conversation_id=conversation_id,
        messages=messages,
    )
