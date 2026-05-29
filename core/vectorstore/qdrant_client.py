"""
Cliente Qdrant: conexion, creacion de colecciones y operaciones CRUD.
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from config import settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        logger.info(f"Qdrant conectado en {settings.qdrant_host}:{settings.qdrant_port}")
    return _client


def _collection_config_ok(client: QdrantClient, collection_name: str) -> bool:
    """Verifica si la coleccion existe y tiene la configuracion de vectores correcta."""
    try:
        info = client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        # Si es un dict (named vectors), no es lo que esperamos
        if isinstance(vectors_config, dict):
            return False
        size = vectors_config.size
        distance_name = str(vectors_config.distance)
        return size == settings.embedding_dimension and "COSINE" in distance_name.upper()
    except Exception:
        return False


def ensure_collections():
    """
    Crea las colecciones si no existen.
    Si una coleccion existe con configuracion incorrecta, la elimina y recrea.
    """
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    for collection_name in [
        settings.qdrant_collection_text,
        settings.qdrant_collection_images,
    ]:
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"  Coleccion creada: {collection_name}")
        elif not _collection_config_ok(client, collection_name):
            logger.warning(f"  Coleccion {collection_name} con config incorrecta, recreando...")
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"  Coleccion recreada: {collection_name}")
        else:
            logger.info(f"  Coleccion ya existe: {collection_name}")


def get_indexed_sources(collection_name: str) -> set[str]:
    """
    Retorna el conjunto de fuentes (nombres de archivo PDF)
    que ya estan indexadas en Qdrant.
    Evita re-indexar documentos ya procesados.
    """
    client = get_qdrant_client()
    sources = set()
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            source = point.payload.get("source")
            if source:
                sources.add(source)
        if next_offset is None:
            break
        offset = next_offset

    return sources


def upsert_points(collection_name: str, points: list[PointStruct]):
    """Inserta o actualiza puntos en Qdrant."""
    client = get_qdrant_client()
    client.upsert(collection_name=collection_name, points=points)


def search_collection(
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Busqueda por similitud vectorial usando query_points."""
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [
        {
            "score": r.score,
            "payload": r.payload,
        }
        for r in response.points
    ]
