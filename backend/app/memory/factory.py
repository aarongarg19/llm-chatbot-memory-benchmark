
import os

from app.memory.base import EmptyMemoryProvider, MemoryProvider


def create_memory_provider() -> MemoryProvider:
    mode = os.getenv("MEMORY_MODE", "none").strip().lower()

    if mode == "none":
        return EmptyMemoryProvider()

    if mode == "sqlite":
        from app.memory.sqlite_memory import SQLiteMemoryProvider

        return SQLiteMemoryProvider()

    if mode == "rag":
        try:
            from app.memory.rag import RAGMemoryProvider
        except ImportError as exc:
            raise RuntimeError(
                "RAG mode is selected, but app.memory.rag "
                "has not been integrated yet."
            ) from exc

        return RAGMemoryProvider()

    if mode == "okf":
        try:
            from app.memory.okf import OKFMemoryProvider
        except ImportError as exc:
            raise RuntimeError(
                "OKF mode is selected, but app.memory.okf "
                "has not been integrated yet."
            ) from exc

        return OKFMemoryProvider()

    raise ValueError(
        f"Invalid MEMORY_MODE: {mode}. "
        "Choose none, sqlite, rag, or okf."
    )
