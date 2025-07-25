import sqlite3
from typing import List, Optional

from rag_chat_app.document_sources.metadata import DocumentMetadata
from ..enums import VectorStatus
from .metadata_store import MetadataStore


class SQLiteMetadataStore(MetadataStore):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        with self._connection() as conn:
            try:
                cursor = conn.cursor()
                for doc in documents:
                    cursor.execute("""
                        INSERT INTO documents(
                            file_hash, sourse_type, source_path, file_name,
                            file_extension, file_size, last_modified, chunk_count,
                            vector_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_path, sourse_type)
                        DO UPDATE SET
                            file_hash = excluded.file_hash,
                            file_size = excluded.file_size,
                            last_modified = excluded.last_modified,
                            chunk_count = 0,
                            updated_at = CURRENT_TIMESTAMP,
                            vector_status = excluded.vector_status,
                            is_deleted = 0
                    """, (
                        doc.file_hash, doc.sourse_type, doc.source_path, doc.file_name,
                        doc.file_extension, doc.file_size, doc.last_modified.isoformat(), doc.chunk_count,
                        doc.vector_status.value
                    ))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    def load_documents_metadata(
        self,
        supported_extentions: List[str] = None,
        source_type: str = None,
        vector_status: Optional[VectorStatus] = None,
    ) -> List[DocumentMetadata]:

        query = """
            SELECT file_hash, sourse_type, source_path, file_name, file_extension, file_size,
            last_modified, chunk_count, vector_status FROM documents
        """

        conditions, params = self._build_filter_conditions(supported_extentions, source_type, vector_status)

        if conditions:
            query += ' WHERE ' + 'AND '.join(conditions)

        documents = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                documents.append(
                    DocumentMetadata(
                        chunk_count=row['chunk_count'],
                        file_extension=row['file_extension'],
                        file_hash=row['file_hash'],
                        file_name=row['file_name'],
                        file_size=row['file_size'],
                        last_modified=row['last_modified'],
                        source_path=row['source_path'],
                        sourse_type=row['sourse_type'],
                        vector_status=VectorStatus.from_string(row['vector_status'])
                    )
                )

        return documents

    def delete_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        with self._connection() as conn:
            try:
                cursor = conn.cursor()
                for doc in documents:
                    cursor.execute("""
                        UPDATE documents
                        SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE sourse_type = ? AND source_path = ?
                    """, (doc.sourse_type, doc.source_path))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    def _build_filter_conditions(
        self,
        supported_extentions: List[str] = None,
        source_type: str = None,
        vector_status: Optional[VectorStatus] = None,
    ):
        condition = ['is_deleted = 0 ']
        params = []

        if vector_status:
            condition.append('vector_status = ?')
            params.append(vector_status.value)

        if source_type:
            condition.append('sourse_type = ?')
            params.append(source_type)

        if supported_extentions:
            placeholder = ' ,'.join(['?'] * len(supported_extentions))
            condition.append(f'file_extension in ({placeholder})')
            params.extend(supported_extentions)

        return condition, params

    def get_by_hash(self, file_hash: str) -> DocumentMetadata:
        pass

    def update_document_processing_status(
        self,
        document: DocumentMetadata,
        vector_status: VectorStatus,
        vector_error: str = '',
        chunk_count: Optional[int] = None
    ) -> None:
        with self._connection() as conn:
            try:
                cursor = conn.cursor()
                if chunk_count:
                    cursor.execute("""
                        UPDATE documents
                        SET vector_status = ?, vector_error = ?, chunk_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE sourse_type = ? AND source_path = ?
                    """, (vector_status.value, vector_error, chunk_count, document.sourse_type, document.source_path))
                else:
                    cursor.execute("""
                        UPDATE documents
                        SET vector_status = ?, vector_error = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE sourse_type = ? AND source_path = ?
                    """, (vector_status.value, vector_error, document.sourse_type, document.source_path))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
