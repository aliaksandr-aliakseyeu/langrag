from typing import Optional

from rag_chat_app.config import settings
from .sqlite_store import SQLiteMetadataStore
from .run_migrations import run_migrations
from .json_store import JsonMetadataStore


def create_sqlite_metadata_store(db_path: Optional[str] = None) -> SQLiteMetadataStore:
    """
    Create and initialize a SQLite metadata store.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Initialized SQLiteMetadataStore
    """
    db_path = db_path or settings.DB_PATH
    run_migrations(db_path)

    return SQLiteMetadataStore(db_path)


def create_json_metadata_store(json_path: Optional[str] = None) -> JsonMetadataStore:
    json_path = json_path or settings.JSON_PATH
    return JsonMetadataStore(json_path)
