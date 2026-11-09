"""
Shared Ollama Embedding Service Implementation for Unified RAG
- Connects to local Ollama daemon (/api/embed endpoint)
- Generates 768-dimensional embeddings using embeddinggemma
- Enforces strict dimension validation, batching, timeouts, and model availability checks
"""

import json
from typing import Any, Dict, List, Optional
import requests

from rag import logger
from model_lifecycle import (
    get_model_manager,
    ModelLeaseTimeoutError,
    ModelLifecycleError,
)
from rag.config import (
    OLLAMA_BASE_URL,
    PRIMARY_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_TIMEOUT_SECONDS,
)


class OllamaConnectionError(Exception):
    """Raised when Ollama daemon is unreachable or timed out."""
    pass


class EmbeddingModelNotFoundError(Exception):
    """Raised when the requested embedding model is not pulled in Ollama."""
    pass


class EmbeddingDimensionError(ValueError):
    """Raised when embedding dimensions are inconsistent or invalid."""
    pass


class EmbeddingGenerationError(Exception):
    """Raised when Ollama /api/embed encounters an unrecoverable error."""
    pass


class EmbeddingService:
    """
    Shared, production-grade vector embedding generator using local Ollama.
    Never silently switches models; strictly validates model presence and vector consistency.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = PRIMARY_EMBEDDING_MODEL,
        batch_size: int = 16,
        timeout: int = EMBEDDING_TIMEOUT_SECONDS,
        expected_dimension: Optional[int] = EMBEDDING_DIMENSION,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name.strip()
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.expected_dimension = expected_dimension
        self._verified: bool = False

    def check_ollama_alive(self) -> bool:
        """Verify that Ollama server is reachable and responding."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        """Fetch list of all installed Ollama model identifiers."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(
                f"Failed to query Ollama at {self.base_url}: {e}"
            ) from e
        return []

    def is_model_installed(self) -> bool:
        """Check if target embedding model exists in Ollama."""
        models = self.list_installed_models()
        target = self.model_name.lower()
        target_base = target.split(":")[0]

        for m in models:
            m_lower = m.lower()
            if m_lower == target or m_lower.startswith(f"{target}:"):
                return True
            if m_lower == target_base or m_lower.startswith(f"{target_base}:"):
                return True
        return False

    def verify_model_available(self, force: bool = False) -> None:
        """
        Validate that Ollama is reachable and target model is installed.
        Raises descriptive errors with exact instructions if checks fail.
        """
        if self._verified and not force:
            return

        if not self.check_ollama_alive():
            raise OllamaConnectionError(
                f"Cannot reach Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running ('ollama serve')."
            )

        if not self.is_model_installed():
            pull_cmd = f"ollama pull {self.model_name}"
            raise EmbeddingModelNotFoundError(
                f"Embedding model '{self.model_name}' is not installed in Ollama.\n"
                f"To pull this model, run the following command in your terminal:\n\n"
                f"    {pull_cmd}\n"
            )

        self._verified = True
        logger.info(f"Verified embedding model '{self.model_name}' on {self.base_url}")

    def _validate_vector(self, vector: Any, index_info: str = "") -> List[float]:
        """Validate that a single vector is a non-empty list of floats matching dimensions."""
        if not isinstance(vector, (list, tuple)) or len(vector) == 0:
            raise EmbeddingDimensionError(f"Empty or non-list vector returned {index_info}")

        length = len(vector)
        if self.expected_dimension is not None and length != self.expected_dimension:
            raise EmbeddingDimensionError(
                f"Vector dimension mismatch {index_info}: expected {self.expected_dimension}, "
                f"got {length}"
            )

        # Validate elements are numeric
        if not all(isinstance(val, (int, float)) for val in vector[:10]):
            raise EmbeddingDimensionError(f"Vector elements must be floats {index_info}")

        return [float(x) for x in vector]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single query string.
        Returns: List[float] representing 768-dimensional normalized embedding.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty query text.")

        self.verify_model_available()

        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model_name,
            "input": text.strip()
        }

        model_manager = get_model_manager()
        try:
            with model_manager.acquire_embedding_lease(timeout=float(self.timeout)):
                try:
                    resp = requests.post(url, json=payload, timeout=self.timeout)
                    if resp.status_code != 200:
                        raise EmbeddingGenerationError(
                            f"Ollama /api/embed returned HTTP {resp.status_code}: {resp.text}"
                        )

                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if not embeddings:
                        raise EmbeddingGenerationError("Ollama /api/embed returned no embeddings.")

                    vector = self._validate_vector(embeddings[0], index_info="[query]")
                    return vector

                except requests.exceptions.Timeout as te:
                    raise OllamaConnectionError(
                        f"Ollama embedding request timed out after {self.timeout}s: {te}"
                    ) from te
                except requests.exceptions.RequestException as re:
                    raise OllamaConnectionError(f"Ollama communication error: {re}") from re
        except ModelLeaseTimeoutError as te:
            raise OllamaConnectionError(
                f"Timed out waiting to acquire embedding lease for {self.model_name}: {te}"
            ) from te

    def embed_text(self, text: str) -> List[float]:
        """Convenience alias for embed_query."""
        return self.embed_query(text)

    def embed_batch(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Convenience alias for embed_documents with optional batch_size override."""
        return self.embed_documents(texts, batch_size=batch_size)

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of document or code chunks with batching.
        Returns: List[List[float]] matching input sequence.
        """
        if not texts:
            return []

        self.verify_model_available()
        all_embeddings: List[List[float]] = []

        url = f"{self.base_url}/api/embed"
        effective_batch_size = batch_size if (batch_size and batch_size > 0) else self.batch_size

        model_manager = get_model_manager()
        try:
            with model_manager.acquire_embedding_lease(timeout=float(self.timeout)):
                # Process in batches
                for start_idx in range(0, len(texts), effective_batch_size):
                    batch = texts[start_idx : start_idx + effective_batch_size]
                    cleaned_batch = [t.strip() if t and t.strip() else " " for t in batch]

                    payload = {
                        "model": self.model_name,
                        "input": cleaned_batch
                    }

                    try:
                        resp = requests.post(url, json=payload, timeout=self.timeout)
                        if resp.status_code != 200:
                            raise EmbeddingGenerationError(
                                f"Ollama batch embedding failed (HTTP {resp.status_code}): {resp.text}"
                            )

                        data = resp.json()
                        batch_embeddings = data.get("embeddings", [])
                        if len(batch_embeddings) != len(cleaned_batch):
                            raise EmbeddingGenerationError(
                                f"Batch size mismatch: sent {len(cleaned_batch)} texts, "
                                f"received {len(batch_embeddings)} vectors"
                            )

                        for i, vec in enumerate(batch_embeddings):
                            validated = self._validate_vector(
                                vec, index_info=f"[item {start_idx + i}]"
                            )
                            all_embeddings.append(validated)

                    except requests.exceptions.Timeout as te:
                        raise OllamaConnectionError(
                            f"Batch embedding timed out on items {start_idx}-{start_idx + len(batch)}: {te}"
                        ) from te
                    except requests.exceptions.RequestException as re:
                        raise OllamaConnectionError(f"Ollama batch communication error: {re}") from re
        except ModelLeaseTimeoutError as te:
            raise OllamaConnectionError(
                f"Timed out waiting to acquire embedding lease for {self.model_name}: {te}"
            ) from te

        logger.debug(f"Embedded {len(texts)} chunks using {self.model_name}")
        return all_embeddings
