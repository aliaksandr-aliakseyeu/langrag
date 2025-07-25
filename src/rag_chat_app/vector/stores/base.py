from abc import ABC, abstractmethod
from langchain.embeddings.base import Embeddings
from langchain.schema import Document


class VectorStoreError(Exception):
    pass


class VectorStore(ABC):

    def __init__(self, embedding_function: Embeddings):
        self.embedding_function = embedding_function
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def add_documents(self, documents: Document) -> None:
        pass

    @abstractmethod
    def document_exists(self, source_path: str) -> bool:
        pass

    @abstractmethod
    def delete_vectors_by_source(self, source_path: str) -> None:
        pass
