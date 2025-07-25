from pathlib import Path
from typing_extensions import TypedDict
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]


class ChunkingConfig(TypedDict):
    max_chunk_size: int
    overlap_size: int
    separators: list[str]


class Settings(BaseSettings):
    """Application settings"""

    DB_PATH: str = str(BASE_DIR / "data" / "documents_meta.db")
    DOCUMENT_FOLDER: str = str(BASE_DIR / "data")
    VECTOR_FOLDER: str = str(BASE_DIR / "data" / "vector")

    CHUNKING_CONFIG: ChunkingConfig = {
        "max_chunk_size": 1000,
        "overlap_size": 100,
        "separators": ["\n\n", "\n", ".", "!", "?", ";", " "]
    }

    class Config:
        env_file = '.env'


settings = Settings()
