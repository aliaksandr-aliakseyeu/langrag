from abc import ABC, abstractmethod
from typing import List

from rag_chat_app.document_sources.metadata import DocumentMetadata


class MetadataStore(ABC):
    @abstractmethod
    def save_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        pass

    @abstractmethod
    def load_documents_metadata(
        self,
        vector_status: str = None,
        supported_extentions: List[str] = None,
        source_type: str = None
    ) -> List[DocumentMetadata]:
        pass

    @abstractmethod
    def get_by_hash(self, file_hash: str) -> DocumentMetadata:
        pass

    @abstractmethod
    def update_document_processing_status(self, metadata: DocumentMetadata) -> None:
        pass

    @abstractmethod
    def delete_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        pass
