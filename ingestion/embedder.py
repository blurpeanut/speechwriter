"""
OpenAI text-embedding-3-small wrapper.
Only permitted embedding provider: api.openai.com.
"""
import logging
import os
import time

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

BATCH_SIZE    = 100
EMBEDDING_DIM = 1536   # text-embedding-3-small output dimension


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """
    Embed a list of strings with OpenAI text-embedding-3-small.

    Returns a float32 numpy array of shape (len(texts), EMBEDDING_DIM).
    Vectors are L2-normalised so that inner-product == cosine similarity,
    matching FAISS IndexFlatIP.

    Processes in batches of BATCH_SIZE with exponential-backoff retry (3 attempts).
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model  = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = _client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(input=batch, model=model)
                vectors.extend(item.embedding for item in resp.data)
                logger.debug(
                    "Embedded batch %d–%d (%d total so far)",
                    start, start + len(batch), len(vectors),
                )
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Embedding attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait
                )
                time.sleep(wait)

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms
