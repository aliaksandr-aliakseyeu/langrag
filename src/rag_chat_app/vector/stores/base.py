from abc import ABC, abstractmethod
from typing import Any, List
from langchain.embeddings.base import Embeddings
from langchain.schema import Document


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""

    pass


class VectorStore(ABC):
    """
    Abstract base class for vector storage backends.

    Defines the interface for storing, retrieving, and managing document
    embeddings across different vector database backends.
    """

    def __init__(self, embedding_function: Embeddings):
        """
        Initialize vector store with embedding function.

        Args:
            embedding_function: Embedding model for vectorizing documents
        """
        self.embedding_function = embedding_function
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store (create collections, connect to DB, etc.)."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: Document objects to embed and store
        """
        pass

    @abstractmethod
    def document_exists(self, source_path: str) -> bool:
        """
        Check if document already exists in the vector store.

        Args:
            source_path: Path to the source document

        Returns:
            True if document exists in the store
        """
        pass

    @abstractmethod
    def delete_vectors_by_source(self, source_path: str) -> None:
        """
        Delete all vectors associated with a source document.

        Args:
            source_path: Path to the source document
        """
        pass

    @abstractmethod
    def as_retriever(self, **kwargs) -> Any:
        """
        Get a LangChain retriever interface for this vector store.

        Args:
            **kwargs: Additional arguments for retriever configuration

        Returns:
            LangChain retriever object
        """
        pass
