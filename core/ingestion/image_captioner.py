"""
Genera descripciones semanticas ricas de imagenes usando NVIDIA AI Endpoints.
Usa el modelo de vision configurado (nvidia/nemotron-nano-12b-v2-vl)
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
    Genera un caption semantico rico usando NVIDIA vision model via API directa.
    Combina la imagen con el texto circundante del Markdown como contexto.

    Args:
        base64_image: Imagen en base64 (sin prefijo data:).
        surrounding_text: Texto del Markdown cercano a la imagen.
        mime_type: MIME type de la imagen.

    Returns:
        Descripcion enriquecida de la imagen.
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
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
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
