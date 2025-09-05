import sqlite3
from datetime import datetime
from typing import List, Optional

from rag_chat_app.document_sources.metadata import DocumentMetadata, DocumentSourceType
from ..enums import VectorStatus
from .metadata_store import MetadataStore


class SQLiteMetadataStore(MetadataStore):
    """
    SQLite-based implementation of document metadata storage.

    Provides CRUD operations for document metadata using SQLite database
    with support for transactions, filtering, and soft deletes.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite metadata store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

    def _connection(self) -> sqlite3.Connection:
        """
        Create database connection with row factory for named access.

        Returns:
            SQLite connection with Row factory configured
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        """
        Save or update document metadata with UPSERT operation.

        Args:
            documents: List of document metadata to persist
        """
        with self._connection() as conn:
            try:
                cursor = conn.cursor()
                for doc in documents:
                    cursor.execute(
                        """
                        INSERT INTO documents(
                            file_hash, source_type, source_path, file_name,
                            file_extension, file_size, last_modified, chunk_count,
                            vector_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_path, source_type)
                        DO UPDATE SET
                            file_hash = excluded.file_hash,
                            file_size = excluded.file_size,
                            last_modified = excluded.last_modified,
                            chunk_count = 0,
                            updated_at = CURRENT_TIMESTAMP,
                            vector_status = excluded.vector_status,
                            is_deleted = 0
                    """,
                        (
                            doc.file_hash,
                            doc.source_type.value,
                            doc.source_path,
                            doc.file_name,
                            doc.file_extension,
                            doc.file_size,
                            doc.last_modified.isoformat(),
                            doc.chunk_count,
                            doc.vector_status.value,
                        ),
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    def load_documents_metadata(
        self,
        supported_extensions: List[str] = None,
        source_type: str = None,
        vector_status: str = None,
    ) -> List[DocumentMetadata]:

        query = """
            SELECT file_hash, source_type, source_path, file_name, file_extension, file_size,
            last_modified, chunk_count, vector_status FROM documents
        """

        conditions, params = self._build_filter_conditions(
            supported_extensions, source_type, vector_status
        )

        if conditions:
            query += " WHERE " + "AND ".join(conditions)

        documents = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                documents.append(
                    DocumentMetadata(
                        chunk_count=row["chunk_count"],
                        file_extension=row["file_extension"],
                        file_hash=row["file_hash"],
                        file_name=row["file_name"],
                        file_size=row["file_size"],
                        last_modified=datetime.fromisoformat(row["last_modified"]),
                        source_path=row["source_path"],
                        source_type=DocumentSourceType(row["source_type"]),
                        vector_status=VectorStatus.from_string(row["vector_status"]),
                    )
                )

        return documents

    def delete_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        with self._connection() as conn:
            try:
                cursor = conn.cursor()
                for doc in documents:
                    cursor.execute(
                        """
                        UPDATE documents
                        SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE source_type = ? AND source_path = ?
                    """,
                        (doc.source_type.value, doc.source_path),
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    def _build_filter_conditions(
        self,
        supported_extensions: List[str] = None,
        source_type: str = None,
        vector_status: str = None,
    ):
        condition = ["is_deleted = 0 "]
        params = []

        if vector_status:
            condition.append("vector_status = ?")
            params.append(vector_status)

        if source_type:
            condition.append("source_type = ?")
            params.append(source_type)

        if supported_extensions:
            placeholder = " ,".join(["?"] * len(supported_extensions))
            condition.append(f"file_extension in ({placeholder})")
            params.extend(supported_extensions)

        return condition, params

    def get_by_hash(self, file_hash: str) -> DocumentMetadata:
        pass

    def update_document_processing_status(
        self,
        document: DocumentMetadata,
        vector_status: VectorStatus,
        vector_error: str = "",
        chunk_count: Optional[int] = None,
    ) -> None:
        with self._connection() as conn:
            try:
                cursor = conn.cursor()
                if chunk_count:
                    cursor.execute(
                        """
                        UPDATE documents
                        SET vector_status = ?, vector_error = ?, chunk_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE source_type = ? AND source_path = ?
                    """,
                        (
                            vector_status.value,
                            vector_error,
                            chunk_count,
                            document.source_type.value,
                            document.source_path,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE documents
                        SET vector_status = ?, vector_error = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE source_type = ? AND source_path = ?
                    """,
                        (
                            vector_status.value,
                            vector_error,
                            document.source_type.value,
                            document.source_path,
                        ),
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
