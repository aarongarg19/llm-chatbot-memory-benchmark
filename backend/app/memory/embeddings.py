
"""Local text embeddings for RAG memory retrieval."""
from __future__ import annotations

import os
import threading
from typing import Sequence

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Generate normalized embeddings using a shared local model."""

    _models: dict[str, SentenceTransformer] = {}
    _model_lock = threading.Lock()

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def _get_model(self) -> SentenceTransformer:
        model = self._models.get(self.model_name)
        if model is not None:
            return model

        with self._model_lock:
            model = self._models.get(self.model_name)
            if model is None:
                model = SentenceTransformer(self.model_name)
                self._models[self.model_name] = model

        return model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized embedding vectors."""
        if not texts:
            return []

        vectors = self._get_model().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
