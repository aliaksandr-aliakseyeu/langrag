from abc import ABC, abstractmethod
from typing import List

from .metadata import DocumentMetadata


class DocumentSource(ABC):
    """
    Abstract base class for document sources.

    Document sources are responsible for:
    - Discovering available documents
    - Providing document metadata
    - Reading document content
    - Monitoring changes to documents
    """

    @abstractmethod
    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all available documents from this source.

        Returns:
            List of DocumentMetadata objects representing available documents
        """
        pass

    @abstractmethod
    def read_document(self, meta_data: DocumentMetadata) -> str:
        """
        Read the content of a specific document.

        Args:
            meta_data: Metadata of the document to read

        Returns:
            Raw text content of the document
        """
        pass
