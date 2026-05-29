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
