"""
Usa Docling para convertir PDFs a Markdown + imagenes en Base64.
Soporta OCR para PDFs escaneados.
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
    # images: [{"caption": str, "base64": str, "page": int, "index": int}]


def build_converter() -> DocumentConverter:
    """Construye el converter de Docling con OCR activado."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True  # Extrae imagenes del PDF

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_pdf(pdf_path: Path, converter: DocumentConverter) -> ParsedDocument:
    """
    Parsea un PDF y devuelve texto Markdown + lista de imagenes en Base64.
    """
    logger.info(f"Parseando: {pdf_path.name}")
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Exportar a Markdown
    markdown_text = doc.export_to_markdown()

    # Extraer imagenes con contexto textual cercano
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
            logger.warning(f"  No se pudo extraer imagen {idx}: {e}")

    logger.info(f"  {len(markdown_text)} chars, {len(images)} imagenes")
    return ParsedDocument(
        source_path=pdf_path,
        markdown_text=markdown_text,
        images=images,
    )


def iter_pdfs(docs_folder: Path) -> Generator[Path, None, None]:
    """Itera sobre todos los PDFs de la carpeta (recursivo)."""
    for pdf in docs_folder.rglob("*.pdf"):
        yield pdf
