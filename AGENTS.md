# 🧠 RAG Multimodal — Guía Completa de Implementación

> **Stack**: Python 3.11 · UV · FastAPI · WebSocket · Docling · NVIDIA AI Endpoints · Gemini Embedding 2 Preview · Qdrant · LangChain · Gemini 2.5 Flash  
> **OS**: Windows 11 · PowerShell · Docker  
> **GPU**: No requerida (NVIDIA API es cloud-based)

---

## 📐 Arquitectura General

```
┌─────────────────────────────────────────────────────────┐
│              PDFs de entrada (una o varias carpetas)     │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼───────┐
                    │  DOCLING     │  → Markdown + imágenes en Base64
                    │  (con OCR)   │  + contexto textual cercano
                    └──────┬───────┘
                           │
              ┌────────────┴─────────────┐
              │                          │
       ┌──────▼───────┐          ┌───────▼────────┐
       │  Texto limpio│          │   Imágenes     │
       │  (Markdown)  │          │  (Base64 + contexto)│
       └──────┬───────┘          └───────┬────────┘
              │                          │
       ┌──────▼───────┐          ┌───────▼────────┐
       │   Chunking   │          │ NVIDIA Vision  │
       │  (solapado)  │          │  Captioning    │
       └──────┬───────┘          │  (context-aware)│
              │                  └───────┬────────┘
       ┌──────▼────────────────────────▼─┐
       │   gemini-embedding-2-preview     │
       │   (texto) + (texto caption)     │
       └──────────────┬───────────────────┘
                      │
               ┌──────▼───────┐
               │    Qdrant    │  type="text" | type="image"
               │  (Docker)    │
               └──────┬───────┘
                      │
               ┌──────▼───────┐
               │Dual Retrieval│  k_text + k_images
               └──────┬───────┘
                      │
               ┌──────▼────────┐
               │ Gemini 2.5    │  → Respuesta multimodal
               │   Flash       │
               └───────────────┘
                      ↑
               ┌──────┴────────┐
               │  FastAPI WS   │  WebSocket /ws/chat
               └───────────────┘
```

---

## 📁 Estructura del Proyecto

```
multimodal-rag/
├── .env                          # Variables de entorno
├── pyproject.toml                # Dependencias UV
├── main.py                       # Entry point FastAPI
├── config.py                     # Configuración global
├── core/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── docling_parser.py     # Parsing PDF con Docling + contexto textual
│   │   ├── image_captioner.py    # NVIDIA Vision Captioning (context-aware)
│   │   ├── chunker.py            # Chunking de texto
│   │   └── embedder.py           # Gemini Embedding (google.genai Client API)
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py      # Conexión y operaciones Qdrant (query_points)
│   │   └── indexer.py            # Indexar documentos nuevos
│   └── retrieval/
│       ├── __init__.py
│       ├── retriever.py          # Dual retrieval text+image
│       └── generator.py          # Gemini 2.5 Flash respuesta (google.genai)
├── api/
│   ├── __init__.py
│   ├── websocket.py              # WebSocket handler
│   └── routes.py                 # REST endpoints
├── docs/                         # Carpeta PDFs de entrada
│   └── *.pdf
└── qdrant_storage/               # Montado por Docker
```

---

## 🐳 Paso 1 — Levantar Qdrant con Docker

```powershell
# PowerShell — desde la raíz del proyecto
docker run -d `
  --name qdrant_container `
  -p 6333:6333 `
  -p 6334:6334 `
  -v ${PWD}/qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

Verifica que esté corriendo:

```powershell
docker ps
# Accede al dashboard: http://localhost:6333/dashboard
```

---

## ⚡ Paso 2 — Configurar entorno con UV

```powershell
# Instalar UV si no lo tienes
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Crear proyecto
uv init multimodal-rag
cd multimodal-rag

# Fijar Python 3.11
uv python pin 3.11
```

### `pyproject.toml`

```toml
[project]
name = "superragmultimodal"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # API y WebSocket
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "websockets>=12.0",
    "python-multipart>=0.0.9",

    # LangChain ecosystem
    "langchain>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "langchain-qdrant>=0.2.0",
    "langchain-community>=0.3.0",

    # Docling para parsing PDF
    "docling>=2.0.0",
    "docling-core>=2.0.0",

    # Qdrant
    "qdrant-client>=1.11.0",

    # Google Gemini
    "google-generativeai>=0.8.0",

    # Image handling
    "Pillow>=10.0.0",

    # Utilidades
    "python-dotenv>=1.0.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "numpy>=1.26.0",
    "tqdm>=4.66.0",
    "httpx>=0.27.0",
    "aiofiles>=24.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```powershell
# Instalar dependencias
uv sync
```

---

## 🔐 Paso 3 — Variables de Entorno

### `.env`

```env
# Google Gemini
GOOGLE_API_KEY=tu_api_key_aqui

# NVIDIA AI Endpoints
NVIDIA_API_KEY=tu_nvidia_api_key_aqui
NVIDIA_MODEL=nvidia/nemotron-nano-12b-v2-vl

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_TEXT=rag_text_chunks
QDRANT_COLLECTION_IMAGES=rag_image_chunks

# Embedding
EMBEDDING_MODEL=gemini-embedding-2-preview
EMBEDDING_DIMENSION=3072

# Generación
GENERATION_MODEL=gemini-2.5-flash

# Carpeta de documentos
DOCS_FOLDER=./filesRAG

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# Retrieval
K_TEXT=5
K_IMAGES=3
```

---

## ⚙️ Paso 4 — Archivos de Código

### `config.py`

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Google
    google_api_key: str

    # NVIDIA
    nvidia_api_key: str = ""
    nvidia_model: str = "nvidia/nemotron-nano-12b-v2-vl"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_text: str = "rag_text_chunks"
    qdrant_collection_images: str = "rag_image_chunks"

    # Modelos
    embedding_model: str = "gemini-embedding-2-preview"
    embedding_dimension: int = 3072
    generation_model: str = "gemini-2.5-flash"

    # Paths
    docs_folder: Path = Path("./filesRAG")

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    k_text: int = 5
    k_images: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
```

---

### `core/ingestion/docling_parser.py`

```python
"""
Usa Docling para convertir PDFs a Markdown + imágenes en Base64.
Soporta OCR para PDFs escaneados.
Extrae contexto textual cercano a cada imagen para captions context-aware.
"""
import logging
import base64
import io
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    source_path: Path
    markdown_text: str
    images: list[dict] = field(default_factory=list)
    # images: [{"base64": str, "page": int, "index": int, "surrounding_text": str}]


def build_converter() -> DocumentConverter:
    """Construye el converter de Docling con OCR activado."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True  # Extrae imágenes del PDF

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_pdf(pdf_path: Path, converter: DocumentConverter) -> ParsedDocument:
    """
    Parsea un PDF y devuelve texto Markdown + lista de imágenes en Base64
    con contexto textual cercano.
    """
    logger.info(f"📄 Parseando: {pdf_path.name}")
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Exportar a Markdown
    markdown_text = doc.export_to_markdown()

    # Extraer imágenes con contexto textual cercano
    images = []
    for idx, picture in enumerate(doc.pictures):
        try:
            pil_image = picture.get_image(doc)
            if pil_image is None:
                continue

            # Convertir PIL a Base64
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Extraer contexto textual cercano del Markdown
            surrounding_text = ""
            image_id = picture.id if hasattr(picture, "id") else str(idx)
            img_tag = f"![{image_id}]"
            tag_idx = markdown_text.find(img_tag)
            if tag_idx != -1:
                start = max(0, tag_idx - 300)
                end = min(len(markdown_text), tag_idx + len(img_tag) + 300)
                surrounding_text = markdown_text[start:end].replace(img_tag, "").strip()

            images.append({
                "base64": b64,
                "page": getattr(picture, "page_no", 0),
                "index": idx,
                "surrounding_text": surrounding_text,
                "source": pdf_path.name,
            })
        except Exception as e:
            logger.warning(f"  ⚠️  No se pudo extraer imagen {idx}: {e}")

    logger.info(f"  ✅ {len(markdown_text)} chars, {len(images)} imágenes")
    return ParsedDocument(
        source_path=pdf_path,
        markdown_text=markdown_text,
        images=images,
    )


def iter_pdfs(docs_folder: Path) -> Generator[Path, None, None]:
    """Itera sobre todos los PDFs de la carpeta (recursivo)."""
    for pdf in docs_folder.rglob("*.pdf"):
        yield pdf
```

---

### `core/ingestion/image_captioner.py`

```python
"""
Genera descripciones semánticas ricas de imágenes usando NVIDIA AI Endpoints.
Usa el modelo de visión configurado (nvidia/nemotron-nano-12b-v2-vl)
con el contexto textual cercano del Markdown para enriquecer el caption.
Basado en el enfoque de Mistral OCR para captions context-aware.
"""
import logging
import base64
import io
import requests

from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

MAX_IMAGE_DIM = 768  # Max dimension for resizing before sending to API
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _resize_base64_image(b64: str, mime_type: str = "image/png") -> tuple[str, str]:
    """Redimensiona imagen si es muy grande para la API de NVIDIA."""
    try:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if w <= MAX_IMAGE_DIM and h <= MAX_IMAGE_DIM:
            return b64, mime_type
        # Resize maintaining aspect ratio
        ratio = min(MAX_IMAGE_DIM / w, MAX_IMAGE_DIM / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "PNG" if "png" in mime_type else "JPEG"
        if fmt == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
            mime_type = "image/jpeg"
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), mime_type
    except Exception as e:
        logger.warning(f"Error redimensionando imagen: {e}")
        return b64, mime_type


def caption_image_with_context(
    base64_image: str,
    surrounding_text: str = "",
    mime_type: str = "image/png",
) -> str:
    """
    Genera un caption semántico rico usando NVIDIA vision model via API directa.
    Combina la imagen con el texto circundante del Markdown como contexto.
    """
    if not settings.nvidia_api_key:
        logger.warning("NVIDIA_API_KEY no configurada, caption por defecto")
        return _fallback_caption(surrounding_text)

    # Resize image to reduce payload size
    resized_b64, actual_mime = _resize_base64_image(base64_image, mime_type)
    image_url = f"data:{actual_mime};base64,{resized_b64}"

    # Build prompt based on Mistral OCR approach
    prompt_text = f"Describe esta imagen basado en este contexto, que sea una descripcion concreta y sencilla basada en el contexto, no indiques que es una imagen o que estas describiendo, solo dame la descripcion: {surrounding_text}"

    payload = {
        "model": settings.nvidia_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.6,
    }

    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            NVIDIA_API_URL,
            json=payload,
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        caption = data["choices"][0]["message"]["content"].strip()

        if not caption:
            logger.warning("Caption vacio de NVIDIA, usando fallback")
            return _fallback_caption(surrounding_text)

        logger.info(f"Caption generado: {caption[:80]}...")
        return caption
    except Exception as e:
        logger.warning(f"Error generando caption con NVIDIA: {e}")
        return _fallback_caption(surrounding_text)


def _fallback_caption(surrounding_text: str) -> str:
    """Caption de fallback cuando no hay API o falla."""
    if surrounding_text.strip():
        return f"Imagen del documento. Contexto cercano: {surrounding_text[:200]}"
    return "Imagen extraida del documento"
```

---

### `core/ingestion/chunker.py`

```python
"""
Chunking de texto Markdown con solapado usando LangChain.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )


def chunk_text(text: str, metadata: dict) -> list[dict]:
    """
    Divide el texto en chunks con metadatos.
    Retorna lista de {"text": str, "metadata": dict}
    """
    splitter = get_text_splitter()
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "metadata": {**metadata, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]
```

---

### `core/ingestion/embedder.py`

```python
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
```

---

### `core/vectorstore/qdrant_client.py`

```python
"""
Cliente Qdrant: conexión, creación de colecciones y operaciones CRUD.
Usa query_points en lugar del método search (deprecado en versiones recientes).
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
        logger.info(f"✅ Qdrant conectado en {settings.qdrant_host}:{settings.qdrant_port}")
    return _client


def ensure_collections():
    """
    Crea las colecciones si no existen.
    Verifica y recrea si la configuración de vectores es incorrecta.
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
            logger.info(f"  📦 Colección creada: {collection_name}")
        else:
            logger.info(f"  ✅ Colección ya existe: {collection_name}")


def get_indexed_sources(collection_name: str) -> set[str]:
    """
    Retorna el conjunto de fuentes (nombres de archivo PDF)
    que ya están indexadas en Qdrant.
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
    """Búsqueda por similitud vectorial usando query_points."""
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
```

---

### `core/vectorstore/indexer.py`

```python
"""
Orquesta el pipeline completo de indexación:
1. Detecta PDFs nuevos (no indexados)
2. Parsea con Docling
3. Caption con NVIDIA Vision (context-aware)
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
    Pipeline de indexación incremental.
    Solo procesa PDFs que NO están ya en Qdrant.
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
        logger.info("Todos los documentos ya están indexados. Nada que hacer.")
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

    logger.info("Indexación completada.")


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

    # --- IMÁGENES ---
    image_points = []
    for img in tqdm(parsed.images, desc="  Captioning + embedding imágenes", leave=False):
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
        logger.info(f"  {len(image_points)} imágenes indexadas")
```

---

### `core/retrieval/retriever.py`

```python
"""
Dual Retrieval: recupera k_text chunks de texto Y k_images imágenes
usando embeddings de consulta de Gemini.
"""
import logging
from config import settings
from core.ingestion.embedder import embed_text
from core.vectorstore.qdrant_client import search_collection

logger = logging.getLogger(__name__)


def retrieve(query: str) -> dict:
    """
    Recupera chunks de texto e imágenes relevantes para la consulta.

    Retorna:
        {
            "text_chunks": [{"score": float, "text": str, "source": str, ...}],
            "images": [{"score": float, "caption": str, "base64": str, ...}],
        }
    """
    # Embedding de la consulta
    query_vec = embed_text(query, task_type="RETRIEVAL_QUERY")

    # Búsqueda en colección de texto
    text_results = search_collection(
        collection_name=settings.qdrant_collection_text,
        query_vector=query_vec,
        limit=settings.k_text,
    )

    # Búsqueda en colección de imágenes
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

    logger.debug(f"📚 Recuperados: {len(text_chunks)} chunks, {len(images)} imágenes")
    return {"text_chunks": text_chunks, "images": images}
```

---

### `core/retrieval/generator.py`

```python
"""
Genera respuesta multimodal con Gemini 2.5 Flash.
Combina texto e imágenes recuperadas en el prompt.
Usa el nuevo paquete google.genai (Client API).
"""
import logging
from google.genai import Client
from google.genai import types
from PIL import Image
import base64
import io

from config import settings
from core.ingestion.embedder import _get_client

logger = logging.getLogger(__name__)


def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def build_prompt_parts(query: str, retrieval: dict) -> list:
    """
    Construye la lista de partes del prompt para Gemini (multimodal).
    Usa types.Part para texto e imágenes.
    """
    text_chunks = retrieval.get("text_chunks", [])
    images = retrieval.get("images", [])

    parts = []

    # Contexto de texto
    if text_chunks:
        context_text = "\n\n---\n\n".join(
            f"[Fuente: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
            for c in text_chunks
        )
        parts.append(types.Part.from_text(text=f"## Contexto de texto recuperado:\n\n{context_text}\n\n"))

    # Imágenes recuperadas
    if images:
        parts.append(types.Part.from_text(text="## Imágenes relevantes recuperadas:\n"))
        for img in images:
            try:
                pil = _b64_to_pil(img["base64"])
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
                parts.append(img_part)
                parts.append(types.Part.from_text(text=f"*Caption: {img['caption']} (fuente: {img['source']}, pag. {img['page']})*\n"))
            except Exception as e:
                logger.warning(f"No se pudo incluir imagen: {e}")
                parts.append(types.Part.from_text(text=f"[Imagen no disponible: {img['caption']}]\n"))

    # Pregunta del usuario
    parts.append(types.Part.from_text(
        text=f"\n## Pregunta del usuario:\n{query}\n\n"
        "Responde de forma clara, precisa y detallada basándote en el contexto "
        "y las imágenes proporcionadas. Si la información no está en el contexto, "
        "indícalo explícitamente."
    ))

    return parts


async def generate_response_stream(query: str, retrieval: dict):
    """
    Genera respuesta con streaming usando Gemini 2.5 Flash.
    Es un generador async que yield chunks de texto.
    """
    client = _get_client()
    parts = build_prompt_parts(query, retrieval)

    response = client.models.generate_content_stream(
        model=f"models/{settings.generation_model}",
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096,
        ),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text
```

---

### `api/websocket.py`

```python
"""
WebSocket handler para chat RAG en tiempo real.
Endpoint: ws://localhost:8000/ws/chat
"""
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

from core.ingestion.embedder import setup_genai
from core.retrieval.retriever import retrieve
from core.retrieval.generator import generate_response_stream

logger = logging.getLogger(__name__)


async def chat_websocket(websocket: WebSocket):
    """
    Protocolo de mensajes:

    Cliente → Servidor:
        {"query": "¿Qué dice el documento sobre X?"}

    Servidor → Cliente (streaming):
        {"type": "start"}
        {"type": "chunk", "content": "texto parcial..."}
        {"type": "sources", "text_sources": [...], "image_sources": [...]}
        {"type": "end"}

    En caso de error:
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info(f"🔌 WebSocket conectado: {websocket.client}")
    setup_genai()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                query = data.get("query", "").strip()
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "JSON inválido"})
                continue

            if not query:
                await websocket.send_json({"type": "error", "message": "Query vacía"})
                continue

            logger.info(f"❓ Query: {query[:80]}...")

            # Señal de inicio
            await websocket.send_json({"type": "start"})

            # Retrieval
            try:
                retrieval = retrieve(query)
            except Exception as e:
                logger.error(f"Error en retrieval: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": f"Error en búsqueda: {e}"})
                continue

            # Streaming de respuesta
            try:
                async for chunk in generate_response_stream(query, retrieval):
                    await websocket.send_json({"type": "chunk", "content": chunk})
            except Exception as e:
                logger.error(f"Error en generación: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": f"Error generando respuesta: {e}"})
                continue

            # Fuentes recuperadas
            await websocket.send_json({
                "type": "sources",
                "text_sources": [
                    {"source": c["source"], "score": round(c["score"], 3)}
                    for c in retrieval["text_chunks"]
                ],
                "image_sources": [
                    {
                        "source": img["source"],
                        "caption": img["caption"],
                        "score": round(img["score"], 3),
                        "page": img["page"],
                        "base64": img.get("base64", ""),
                    }
                    for img in retrieval["images"]
                ],
            })

            # Señal de fin
            await websocket.send_json({"type": "end"})

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket desconectado: {websocket.client}")
    except Exception as e:
        logger.error(f"Error inesperado en WebSocket: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
```

---

### `api/routes.py`

```python
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
    """Dispara la indexación de nuevos documentos en background."""
    folder = Path(request.docs_folder) if request.docs_folder else settings.docs_folder
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Carpeta no encontrada: {folder}")
    background_tasks.add_task(run_indexing_pipeline, folder)
    return {"message": "Indexación iniciada en background", "folder": str(folder)}


@router.get("/status")
async def get_status():
    """Estado del sistema: colecciones y documentos indexados."""
    client = get_qdrant_client()
    collections = {c.name: c.vectors_count for c in client.get_collections().collections}
    indexed_text = get_indexed_sources(settings.qdrant_collection_text)
    indexed_images = get_indexed_sources(settings.qdrant_collection_images)
    return {
        "status": "ok",
        "collections": collections,
        "indexed_documents": sorted(indexed_text | indexed_images),
        "total_documents": len(indexed_text | indexed_images),
    }


@router.delete("/collection/{name}")
async def clear_collection(name: str):
    """⚠️ Elimina todos los puntos de una colección (no la borra)."""
    client = get_qdrant_client()
    client.delete_collection(name)
    return {"message": f"Colección {name} eliminada"}
```

---

### `main.py`

```python
"""
Entry point de la aplicación FastAPI.
Al iniciar:
  1. Verifica colecciones en Qdrant
  2. Indexa SOLO documentos nuevos (incremental)
  3. Levanta WebSocket y REST API
"""
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core.ingestion.embedder import setup_genai
from core.vectorstore.indexer import run_indexing_pipeline
from api.websocket import chat_websocket
from api.routes import router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: indexación incremental automática.
    Shutdown: limpieza de recursos.
    """
    logger.info("🚀 Iniciando RAG Multimodal Backend...")
    setup_genai()

    # Indexar documentos nuevos en el arranque (no bloquea si todo está listo)
    logger.info("🔍 Verificando documentos pendientes de indexar...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_indexing_pipeline, settings.docs_folder)

    logger.info("✅ Backend listo. WebSocket disponible en ws://localhost:8000/ws/chat")
    yield

    logger.info("👋 Cerrando backend...")


app = FastAPI(
    title="RAG Multimodal API",
    description="Backend RAG multimodal con Docling, NVIDIA AI Endpoints, Gemini y Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(router)

# WebSocket
app.add_api_websocket_route("/ws/chat", chat_websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.generation_model}
```

---

## ▶️ Paso 5 — Ejecutar el Backend

```powershell
# 1. Asegúrate de que Qdrant esté corriendo
docker ps | findstr qdrant

# 2. Coloca tus PDFs en la carpeta filesRAG/
# mkdir filesRAG  (si no existe)

# 3. Arrancar el backend con UV
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# O con workers múltiples (producción, sin --reload):
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Al arrancar verás:

```
08:00:00 [INFO] 🚀 Iniciando RAG Multimodal Backend...
08:00:01 [INFO] Cliente Gemini inicializado
08:00:02 [INFO] ✅ Qdrant conectado en localhost:6333
08:00:02 [INFO] ✅ Colección ya existe: rag_text_chunks
08:00:02 [INFO] ✅ Colección ya existe: rag_image_chunks
08:00:02 [INFO] 5 PDFs totales | 2 nuevos a indexar
08:00:02 [INFO] � Parseando: nuevo_reporte.pdf
...
08:05:30 [INFO] Indexación completada.
08:05:30 [INFO] ✅ Backend listo. WebSocket disponible en ws://localhost:8000/ws/chat
```

---

## 🧪 Paso 6 — Probar el WebSocket

### Con Python (cliente de prueba):

```python
import asyncio
import websockets
import json

async def test_chat():
    uri = "ws://localhost:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        # Enviar pregunta
        await ws.send(json.dumps({"query": "¿Qué información hay sobre machine learning en los documentos?"}))

        full_response = ""
        async for message in ws:
            msg = json.loads(message)
            match msg["type"]:
                case "start":
                    print("🔄 Generando respuesta...\n")
                case "chunk":
                    print(msg["content"], end="", flush=True)
                    full_response += msg["content"]
                case "sources":
                    print(f"\n\n📚 Fuentes de texto: {msg['text_sources']}")
                    print(f"🖼️  Fuentes de imagen: {msg['image_sources']}")
                case "end":
                    print("\n✅ Respuesta completa")
                    break
                case "error":
                    print(f"\n❌ Error: {msg['message']}")
                    break

asyncio.run(test_chat())
```

### Con curl (REST endpoints):

```powershell
# Estado del sistema
curl http://localhost:8000/api/status

# Disparar indexación manual
curl -X POST http://localhost:8000/api/index `
  -H "Content-Type: application/json" `
  -d '{"docs_folder": "./docs"}'

# Health check
curl http://localhost:8000/health
```

---

## 🔄 Flujo de Indexación Incremental (detallado)

```
Al arrancar main.py:
│
├── ensure_collections()
│   ├── Si collection "rag_text_chunks" NO existe → la crea
│   └── Si ya existe → no toca nada (idempotente)
│
├── get_indexed_sources("rag_text_chunks")
│   └── Hace scroll en Qdrant, extrae todos los "source" únicos
│       → Ejemplo: {"manual_v1.pdf", "guia_tecnica.pdf"}
│
├── iter_pdfs(docs_folder)
│   └── Encuentra todos los .pdf en ./docs/ recursivamente
│       → Ejemplo: ["manual_v1.pdf", "guia_tecnica.pdf", "nuevo_reporte.pdf"]
│
├── new_pdfs = todos - ya_indexados
│   └── → ["nuevo_reporte.pdf"]
│
└── Si new_pdfs está vacío:
    └── "✅ Todos los documentos ya están indexados. Nada que hacer."
        (el backend arranca en segundos)
```

---

## 🏗️ Paso 7 — Agregar Nuevos Documentos

```powershell
# Simplemente copia los PDFs nuevos a la carpeta docs/
Copy-Item "C:\mis_docs\nuevo_informe.pdf" .\docs\

# Opción A: Reinicia el backend (indexará automáticamente al arrancar)
# Ctrl+C para detener, luego:
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Opción B: Llama al endpoint REST sin reiniciar
curl -X POST http://localhost:8000/api/index `
  -H "Content-Type: application/json" `
  -d '{}'
```

---

## 🔧 Consideraciones Técnicas

### NVIDIA AI Endpoints

- El modelo `nvidia/nemotron-nano-12b-v2-vl` es cloud-based, no requiere GPU local.
- Las imágenes se redimensionan a max 768px antes de enviar a la API para reducir payload.
- Timeout de 60 segundos por imagen captioning.

### Rate Limits Gemini Embedding

- El modelo `gemini-embedding-2-preview` tiene rate limits. El `embed_texts_batch` incluye `sleep_between=1.0s` entre batches de 50 textos.
- Si tienes muchos documentos, considera aumentar `sleep_between` a 2.0.

### Dimensión del embedding

- `gemini-embedding-2-preview` produce vectores de **3072 dimensiones**.
- Asegúrate que `EMBEDDING_DIMENSION=3072` en el `.env`.

### Chunking recomendado para PDFs técnicos

```env
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

Para documentos narrativos/legales:

```env
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

---

## 📊 Endpoints Disponibles

| Método   | Endpoint                      | Descripción                |
| -------- | ----------------------------- | -------------------------- |
| `WS`     | `ws://localhost:8000/ws/chat` | Chat RAG en tiempo real    |
| `GET`    | `/health`                     | Estado del servidor        |
| `GET`    | `/api/status`                 | Colecciones e índices      |
| `POST`   | `/api/index`                  | Disparar indexación manual |
| `DELETE` | `/api/collection/{name}`      | Limpiar colección          |
| `GET`    | `/docs`                       | Swagger UI automático      |

---

## 🐞 Troubleshooting

### Qdrant no responde

```powershell
docker logs qdrant_container
docker restart qdrant_container
```

### NVIDIA API captioning falla

- Verifica que `NVIDIA_API_KEY` esté configurada en `.env`
- Verifica que el modelo `nvidia/nemotron-nano-12b-v2-vl` esté disponible en NVIDIA AI Endpoints
- Si hay timeouts, las imágenes pueden ser muy grandes - el código las redimensiona automáticamente a 768px

### Rate limit Gemini

```
google.api_core.exceptions.ResourceExhausted: 429
```

Aumenta `sleep_between` en `embed_texts_batch()` o reduce `batch_size`.

### Docling OCR lento

Docling descarga modelos de OCR en la primera ejecución (~500MB). Es normal que tarde la primera vez.

### Colecciones con dimensión incorrecta

Si cambiaste `EMBEDDING_DIMENSION`, debes borrar y recrear las colecciones:

```powershell
curl -X DELETE http://localhost:8000/api/collection/rag_text_chunks
curl -X DELETE http://localhost:8000/api/collection/rag_image_chunks
# Luego reinicia el backend
```

---

---

_Generado para Windows 11 PowerShell · Python 3.11 · UV · Stack: Docling + Gemini Embedding 2 Preview + Qdrant + LangChain + FastAPI + Gemini 2.5 Flash_

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author / Autor

**Diego Ivan Perea Montealegre**

- GitHub: [@diegoperea20](https://github.com/diegoperea20)

---

Created by [Diego Ivan Perea Montealegre](https://github.com/diegoperea20)