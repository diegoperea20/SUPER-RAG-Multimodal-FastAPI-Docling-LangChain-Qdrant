"""
Genera respuesta multimodal con Gemini 2.5 Flash.
Combina texto e imagenes recuperadas en el prompt.
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
    Usa types.Part para texto e imagenes.
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

    # Imagenes recuperadas
    if images:
        parts.append(types.Part.from_text(text="## Imagenes relevantes recuperadas:\n"))
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
        "Responde de forma clara, precisa y detallada basandote en el contexto "
        "y las imagenes proporcionadas. Si la informacion no esta en el contexto, "
        "indicalo explicitamente."
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
