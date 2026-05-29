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
    docs_folder: Path = Path("./docs")

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    k_text: int = 5
    k_images: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
