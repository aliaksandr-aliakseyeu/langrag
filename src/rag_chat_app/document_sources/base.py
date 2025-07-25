from abc import ABC, abstractmethod
from typing import List

from .metadata import DocumentMetadata


class DocumentSource(ABC):
    @abstractmethod
    def list_documents(self) -> List[DocumentMetadata]:
        pass

    @abstractmethod
    def read_document(self, meta_data: DocumentMetadata) -> str:
        pass

    @abstractmethod
    def watch_for_changes(self):
        pass
