
import os
import re
import time
import uuid

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
from app.memory.base import MemoryProvider
from app.memory.factory import create_memory_provider


app = FastAPI(
    title="LLM Chatbot Memory Benchmark",
    description="Chatbot with persistent conversations and pluggable memory",
    version="0.4.1",
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

memory_provider: MemoryProvider | None = None


@app.on_event("startup")
def startup():
    global memory_provider
    init_db()
    memory_provider = create_memory_provider()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    memory_mode: str
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class MessageResponse(BaseModel):
    conversation_id: str
    messages: list[dict]


def extract_explicit_memory(message: str) -> str | None:
    """Save simple facts explicitly stated by the user."""
    text = message.strip()

    patterns = [
        r"my\s+.+\s+(?:is|are)\s+.+",
        r"i\s+am\s+.+",
        r"i'm\s+.+",
        r"i\s+live\s+.+",
        r"i\s+prefer\s+.+",
        r"i\s+like\s+.+",
        r"i\s+love\s+.+",
        r"i\s+study\s+.+",
        r"i\s+work\s+.+",
        r"remember\s+that\s+.+",
    ]

    if len(text) > 500:
        return None

    if any(re.fullmatch(pattern, text, re.IGNORECASE) for pattern in patterns):
        return text

    return None


@app.get("/")
def home():
    return {
        "project": "LLM Chatbot Memory Benchmark",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "memory_mode": os.getenv("MEMORY_MODE", "none").strip().lower(),
        "llm_model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    total_start = time.perf_counter()
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty or whitespace only.",
        )

    if memory_provider is None:
        raise HTTPException(
            status_code=503,
            detail="Memory provider has not been initialized.",
        )

    conversation_id = request.conversation_id

    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        create_conversation(conversation_id)
    elif not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    history = get_recent_messages(conversation_id, limit=12)

    memories = []
    retrieval_start = time.perf_counter()

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

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # Save explicit facts as long-term memories.
    if request.user_id:
        fact = extract_explicit_memory(message)

        if fact:
            try:
                await memory_provider.remember(
                    user_id=request.user_id,
                    text=fact,
                    metadata={"source": "explicit_user_statement"},
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Memory storage failed.",
                ) from exc

    save_message(conversation_id, "user", message)

    llm_start = time.perf_counter()

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

    llm_ms = (time.perf_counter() - llm_start) * 1000

    save_message(conversation_id, "assistant", reply)

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
        memory_mode=os.getenv("MEMORY_MODE", "none").strip().lower(),
        retrieval_ms=round(retrieval_ms, 2),
        llm_ms=round(llm_ms, 2),
        total_ms=round((time.perf_counter() - total_start) * 1000, 2),
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

    return MessageResponse(
        conversation_id=conversation_id,
        messages=get_conversation_messages(conversation_id),
    )
