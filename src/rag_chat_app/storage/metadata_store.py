from abc import ABC, abstractmethod
from typing import List, Optional

from rag_chat_app.document_sources.metadata import DocumentMetadata
from rag_chat_app.enums import VectorStatus


class MetadataStore(ABC):
    """
    Abstract base class for document metadata persistence.

    Defines the interface for storing, retrieving, and managing document metadata
    across different storage backends (SQLite, PostgreSQL, etc.).
    """

    @abstractmethod
    def save_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        """
        Save or update document metadata.

        Args:
            documents: List of document metadata to persist
        """
        pass

    @abstractmethod
    def load_documents_metadata(
        self,
        vector_status: str = None,
        supported_extensions: List[str] = None,
        source_type: str = None,
    ) -> List[DocumentMetadata]:
        """
        Load document metadata with optional filtering.

        Args:
            vector_status: Filter by vectorization status
            supported_extensions: Filter by file extensions
            source_type: Filter by document source type

        Returns:
            List of matching document metadata
        """
        pass

    @abstractmethod
    def get_by_hash(self, file_hash: str) -> DocumentMetadata:
        """
        Retrieve document metadata by file hash.

        Args:
            file_hash: Hash of the document file

        Returns:
            Document metadata for the given hash
        """
        pass

    @abstractmethod
    def update_document_processing_status(
        self,
        document: DocumentMetadata,
        vector_status: VectorStatus,
        vector_error: str = "",
        chunk_count: Optional[int] = None,
    ) -> None:
        """
        Update document processing status and related fields.

        Args:
            metadata: Document metadata with updated status
        """
        pass

    @abstractmethod
    def delete_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        """
        Mark documents as deleted (soft delete).

        Args:
            documents: List of documents to mark as deleted
        """
        pass
