"""
Entry point de la aplicacion FastAPI.
Al iniciar:
  1. Verifica colecciones en Qdrant
  2. Indexa SOLO documentos nuevos (incremental)
  3. Levanta WebSocket y REST API
"""
import logging
import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core.ingestion.embedder import setup_genai
from core.vectorstore.indexer import run_indexing_pipeline
from api.websocket import chat_websocket
from api.routes import router

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Silenciar loggers ruidosos de terceros
for _name in [
    "httpx", "httpcore", "hpack", "hyperframe",
    "docling", "docling.models", "docling.pipeline",
    "docling.utils", "docling.document_converter",
    "transformers", "PIL", "filelock",
    "rapidocr", "RapidOCR",
    "torch", "torchvision",
    "urllib3", "charset_normalizer",
    "watchfiles",
    "multipart",
    "google.genai", "huggingface_hub",
]:
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: indexacion incremental automatica.
    Shutdown: limpieza de recursos.
    """
    logger.info("Iniciando RAG Multimodal Backend...")
    setup_genai()

    # Indexar documentos nuevos en el arranque (no bloquea si todo esta listo)
    logger.info("Verificando documentos pendientes de indexar...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_indexing_pipeline, settings.docs_folder)

    logger.info("Backend listo. WebSocket disponible en ws://localhost:8000/ws/chat")
    yield

    logger.info("Cerrando backend...")


app = FastAPI(
    title="RAG Multimodal API",
    description="Backend RAG multimodal con Docling, Gemini y Qdrant",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
