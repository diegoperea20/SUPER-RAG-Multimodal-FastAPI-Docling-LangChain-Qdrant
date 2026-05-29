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

    Cliente -> Servidor:
        {"query": "¿Que dice el documento sobre X?"}

    Servidor -> Cliente (streaming):
        {"type": "start"}
        {"type": "chunk", "content": "texto parcial..."}
        {"type": "sources", "text_sources": [...], "image_sources": [...]}
        {"type": "end"}

    En caso de error:
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info(f"WebSocket conectado: {websocket.client}")
    setup_genai()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                query = data.get("query", "").strip()
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "JSON invalido"})
                continue

            if not query:
                await websocket.send_json({"type": "error", "message": "Query vacia"})
                continue

            logger.info(f"Query: {query[:80]}...")

            # Senal de inicio
            await websocket.send_json({"type": "start"})

            # Retrieval
            try:
                retrieval = retrieve(query)
            except Exception as e:
                logger.error(f"Error en retrieval: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": f"Error en busqueda: {e}"})
                continue

            # Streaming de respuesta
            try:
                async for chunk in generate_response_stream(query, retrieval):
                    await websocket.send_json({"type": "chunk", "content": chunk})
            except Exception as e:
                logger.error(f"Error en generacion: {e}", exc_info=True)
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

            # Senal de fin
            await websocket.send_json({"type": "end"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket desconectado: {websocket.client}")
    except Exception as e:
        logger.error(f"Error inesperado en WebSocket: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
