from typing import Optional

from rag_chat_app.config import settings
from .sqlite_store import SQLiteMetadataStore
from .run_migrations import run_migrations


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
