"""
Orquesta el pipeline completo de indexacion:
1. Detecta PDFs nuevos (no indexados)
2. Parsea con Docling
3. Caption con OpenRouter (vision + contexto)
4. Embedding con Gemini
5. Upsert a Qdrant
"""
import uuid
import logging
from pathlib import Path
from tqdm import tqdm

from qdrant_client.models import PointStruct

from config import settings
from core.ingestion.docling_parser import build_converter, parse_pdf, iter_pdfs
from core.ingestion.image_captioner import caption_image_with_context
from core.ingestion.chunker import chunk_text
from core.ingestion.embedder import setup_genai, embed_text, embed_image_with_caption
from core.vectorstore.qdrant_client import (
    ensure_collections,
    get_indexed_sources,
    upsert_points,
)

logger = logging.getLogger(__name__)


def run_indexing_pipeline(docs_folder: Path | None = None):
    """
    Pipeline de indexacion incremental.
    Solo procesa PDFs que NO estan ya en Qdrant.
    """
    docs_folder = docs_folder or settings.docs_folder
    setup_genai()

    # 1. Asegurar que las colecciones existen
    ensure_collections()

    # 2. Detectar fuentes ya indexadas (deben estar en AMBAS colecciones)
    indexed_text = get_indexed_sources(settings.qdrant_collection_text)
    indexed_images = get_indexed_sources(settings.qdrant_collection_images)
    already_indexed = indexed_text & indexed_images

    # 3. Obtener PDFs pendientes
    all_pdfs = list(iter_pdfs(docs_folder))
    new_pdfs = [p for p in all_pdfs if p.name not in already_indexed]

    if not new_pdfs:
        logger.info("Todos los documentos ya estan indexados. Nada que hacer.")
        return

    logger.info(f"{len(all_pdfs)} PDFs totales | {len(new_pdfs)} nuevos a indexar")

    # 4. Construir converter Docling
    converter = build_converter()

    # 5. Procesar cada PDF nuevo
    for pdf_path in tqdm(new_pdfs, desc="Indexando PDFs"):
        try:
            _index_single_pdf(pdf_path, converter)
        except Exception as e:
            logger.error(f"Error indexando {pdf_path.name}: {e}", exc_info=True)

    logger.info("Indexacion completada.")


def _index_single_pdf(pdf_path: Path, converter):
    logger.info(f"\n{'='*50}")
    logger.info(f"Procesando: {pdf_path.name}")

    # Parse
    parsed = parse_pdf(pdf_path, converter)
    metadata_base = {"source": pdf_path.name, "source_path": str(pdf_path)}

    # --- TEXTO ---
    chunks = chunk_text(parsed.markdown_text, metadata_base)
    text_points = []
    for chunk in tqdm(chunks, desc="  Embedding texto", leave=False):
        vec = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
        text_points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    **chunk["metadata"],
                    "text": chunk["text"],
                    "type": "text",
                },
            )
        )

    if text_points:
        upsert_points(settings.qdrant_collection_text, text_points)
        logger.info(f"  {len(text_points)} chunks de texto indexados")

    # --- IMAGENES ---
    image_points = []
    for img in tqdm(parsed.images, desc="  Captioning + embedding imagenes", leave=False):
        try:
            caption = caption_image_with_context(
                base64_image=img["base64"],
                surrounding_text=img.get("surrounding_text", ""),
            )
            vec = embed_image_with_caption(caption, img["base64"])
            image_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        **metadata_base,
                        "caption": caption,
                        "base64": img["base64"],
                        "page": img["page"],
                        "image_index": img["index"],
                        "type": "image",
                    },
                )
            )
        except Exception as e:
            logger.warning(f"  Error en imagen {img['index']}: {e}")

    if image_points:
        upsert_points(settings.qdrant_collection_images, image_points)
        logger.info(f"  {len(image_points)} imagenes indexadas")
