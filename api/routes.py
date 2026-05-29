"""
REST endpoints complementarios.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
from pydantic import BaseModel

from config import settings
from core.vectorstore.indexer import run_indexing_pipeline
from core.vectorstore.qdrant_client import get_qdrant_client, get_indexed_sources

router = APIRouter(prefix="/api", tags=["RAG"])


class IndexRequest(BaseModel):
    docs_folder: str | None = None


@router.post("/index")
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Dispara la indexacion de nuevos documentos en background."""
    folder = Path(request.docs_folder) if request.docs_folder else settings.docs_folder
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Carpeta no encontrada: {folder}")
    background_tasks.add_task(run_indexing_pipeline, folder)
    return {"message": "Indexacion iniciada en background", "folder": str(folder)}


@router.get("/status")
async def get_status():
    """Estado del sistema: colecciones y documentos indexados."""
    client = get_qdrant_client()
    collections_info = {}
    for c in client.get_collections().collections:
        try:
            info = client.get_collection(c.name)
            collections_info[c.name] = info.points_count
        except Exception:
            collections_info[c.name] = -1
    indexed_text = get_indexed_sources(settings.qdrant_collection_text)
    indexed_images = get_indexed_sources(settings.qdrant_collection_images)
    return {
        "status": "ok",
        "collections": collections_info,
        "indexed_documents": sorted(indexed_text | indexed_images),
        "total_documents": len(indexed_text | indexed_images),
    }


@router.delete("/collection/{name}")
async def clear_collection(name: str):
    """Elimina todos los puntos de una coleccion (no la borra)."""
    client = get_qdrant_client()
    client.delete_collection(name)
    return {"message": f"Coleccion {name} eliminada"}
