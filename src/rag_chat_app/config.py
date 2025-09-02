import logging

from pathlib import Path
from typing import List
from typing_extensions import TypedDict
from pydantic import Field
from pydantic_settings import BaseSettings

from .llm.enums import LLMProvider, OpenAIModel

BASE_DIR = Path(__file__).resolve().parents[2]


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Separate log files for different components
CHAT_LOG_FILE = LOG_DIR / "chat.log"
INGESTION_LOG_FILE = LOG_DIR / "ingestion.log"

# Chat/Retrieval logging configuration
CHAT_LOGGING_CONFIG = {
    "filename": CHAT_LOG_FILE,
    "filemode": "a",
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "encoding": "utf-8",
}

# Ingestion logging configuration
INGESTION_LOGGING_CONFIG = {
    "filename": INGESTION_LOG_FILE,
    "filemode": "a",
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "encoding": "utf-8",
}

# Backward compatibility - default to chat config
LOGGING_CONFIG = CHAT_LOGGING_CONFIG


class ChunkingConfig(TypedDict):
    max_chunk_size: int
    overlap_size: int
    separators: list[str]


class Settings(BaseSettings):
    """Application settings"""

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # Metadata store configuration
    DB_PATH: str = str(BASE_DIR / "data" / "documents_meta.db")
    DOCUMENT_FOLDER: str = str(BASE_DIR / "data")

    # Vector store configuration
    VECTOR_FOLDER: str = str(BASE_DIR / "data" / "vector")
    VECTOR_COLLECTION_NAME: str = "rag_documents"

    # Chunking configuration
    CHUNKING_CONFIG: ChunkingConfig = {
        "max_chunk_size": 1000,
        "overlap_size": 100,
        "separators": ["\n\n", "\n", ".", "!", "?", ";", " "],
    }

    # LLM configuration
    LLM_INTENTION_PROVIDER: LLMProvider = LLMProvider.OPENAI
    LLM_INTENTION_MODEL: OpenAIModel = OpenAIModel.GPT_4O_MINI
    LLM_CHAT_PROVIDER: LLMProvider = LLMProvider.OPENAI
    LLM_CHAT_MODEL: OpenAIModel = OpenAIModel.GPT_4O_MINI

    # Parser configuration
    ENABLED_PARSERS: List[str] = [
        "pdf",
    ]

    # Embedding configuration
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    HUGGINGFACE_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"


settings = Settings()
