"""
Dual Retrieval: recupera k_text chunks de texto Y k_images imagenes
usando embeddings de consulta de Gemini.
"""
import logging
from config import settings
from core.ingestion.embedder import embed_text
from core.vectorstore.qdrant_client import search_collection

logger = logging.getLogger(__name__)


def retrieve(query: str) -> dict:
    """
    Recupera chunks de texto e imagenes relevantes para la consulta.

    Retorna:
        {
            "text_chunks": [{"score": float, "text": str, "source": str, ...}],
            "images": [{"score": float, "caption": str, "base64": str, ...}],
        }
    """
    # Embedding de la consulta
    query_vec = embed_text(query, task_type="RETRIEVAL_QUERY")

    # Busqueda en coleccion de texto
    text_results = search_collection(
        collection_name=settings.qdrant_collection_text,
        query_vector=query_vec,
        limit=settings.k_text,
    )

    # Busqueda en coleccion de imagenes
    image_results = search_collection(
        collection_name=settings.qdrant_collection_images,
        query_vector=query_vec,
        limit=settings.k_images,
    )

    text_chunks = [
        {
            "score": r["score"],
            "text": r["payload"].get("text", ""),
            "source": r["payload"].get("source", ""),
            "chunk_index": r["payload"].get("chunk_index", 0),
        }
        for r in text_results
    ]

    images = [
        {
            "score": r["score"],
            "caption": r["payload"].get("caption", ""),
            "base64": r["payload"].get("base64", ""),
            "source": r["payload"].get("source", ""),
            "page": r["payload"].get("page", 0),
        }
        for r in image_results
    ]

    logger.debug(f"Recuperados: {len(text_chunks)} chunks, {len(images)} imagenes")
    return {"text_chunks": text_chunks, "images": images}
