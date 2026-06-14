"""OpenAI-compatible LLM client for benchmarks.

Supports Azure OpenAI (default) and OpenAI-compatible servers (vLLM, etc.)
via OPENAI_BASE_URL. When the remote server has no /embeddings endpoint,
falls back to local sentence-transformers.
"""

from __future__ import annotations

import os
from typing import List, Protocol

import numpy as np


class EmbeddingClient(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


class AzureEmbeddingClient:
    def __init__(self, client, model: str, dimension: int = 1536):
        self._client = client
        self._model = model
        self._dimension = dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)


class SentenceTransformerEmbeddingClient:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dim_fn = getattr(self._model, "get_embedding_dimension", None)
        self._dimension = dim_fn() if dim_fn else self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)


def _server_has_embeddings(base_url: str, api_key: str, model: str) -> bool:
    import httpx

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                url,
                headers=headers,
                json={"model": model, "input": "ping"},
            )
        return resp.status_code == 200
    except Exception:
        return False


def create_embedding_client(config) -> EmbeddingClient:
    """Create an embedding client from benchmark Config."""
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")
    model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        os.getenv("OPENAI_MODEL", config.azure_embedding_deployment),
    )

    if base_url:
        if _server_has_embeddings(base_url, api_key, model):
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
            return AzureEmbeddingClient(client, model, config.embedding_dim)

        st_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        print(f"  Note: {base_url} has no /embeddings — using local {st_model}")
        return SentenceTransformerEmbeddingClient(st_model)

    if not config.azure_api_key or not config.azure_endpoint:
        raise ValueError(
            "Set OPENAI_BASE_URL or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT"
        )

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        azure_endpoint=config.azure_endpoint,
    )
    return AzureEmbeddingClient(client, config.azure_embedding_deployment, config.embedding_dim)


def create_chat_client(config):
    """Create chat client for OpenAI-compatible or Azure endpoints."""
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")

    if base_url:
        from openai import OpenAI

        return OpenAI(base_url=base_url, api_key=api_key)

    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        azure_endpoint=config.azure_endpoint,
    )


def chat_model_name(config) -> str:
    return os.getenv(
        "OPENAI_MODEL",
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", config.azure_chat_deployment),
    )