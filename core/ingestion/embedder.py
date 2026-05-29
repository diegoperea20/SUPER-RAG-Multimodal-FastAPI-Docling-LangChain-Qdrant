"""
Embeddings multimodales con gemini-embedding-2-preview.
Usa el nuevo paquete google.genai (Client API).
"""
import logging
import time
from config import settings
from google.genai import Client

logger = logging.getLogger(__name__)

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(api_key=settings.google_api_key)
    return _client


def setup_genai():
    """Inicializa el cliente de Gemini."""
    _get_client()
    logger.info("Cliente Gemini inicializado")


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Genera embedding para texto.
    task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | SEMANTIC_SIMILARITY
    """
    if not text or not text.strip():
        raise ValueError("No se puede generar embedding de texto vacio")
    client = _get_client()
    result = client.models.embed_content(
        model=f"models/{settings.embedding_model}",
        contents=text.strip(),
        config={"task_type": task_type},
    )
    return result.embeddings[0].values


def embed_image_with_caption(caption: str, base64_image: str) -> list[float]:
    """
    Genera embedding para imagen usando su caption descriptivo.
    Si el caption esta vacio, usa un fallback generico.
    """
    text = caption.strip() if caption and caption.strip() else "Imagen extraida del documento"
    return embed_text(text, task_type="RETRIEVAL_DOCUMENT")


def embed_texts_batch(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50,
    sleep_between: float = 1.0,
) -> list[list[float]]:
    """Embedding en lotes para respetar rate limits de Gemini."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"  Embedding batch {i//batch_size + 1} ({len(batch)} textos)...")
        embeddings = [embed_text(t, task_type) for t in batch]
        all_embeddings.extend(embeddings)
        if i + batch_size < len(texts):
            time.sleep(sleep_between)
    return all_embeddings
