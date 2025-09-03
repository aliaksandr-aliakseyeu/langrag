import logging
from pathlib import Path
from typing import List, Optional
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from chromadb import PersistentClient

from rag_chat_app.config import settings
from rag_chat_app.vector.stores.base import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based implementation of vector storage.

    Provides document embedding storage and similarity search using ChromaDB
    with persistent storage and collection management.
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str,
        persist_directory: str = settings.VECTOR_FOLDER,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            embedding_function: Embedding model for vectorizing documents
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        super().__init__(embedding_function)
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.vectorstore: Optional[Chroma] = None

    def initialize(self):
        """Initialize ChromaDB collection and prepare for operations."""
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_directory),
            )
            self._initialized = True
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}") from e

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """
        Add documents to the ChromaDB collection in batches.

        Args:
            documents: Document objects to embed and store
            batch_size: Maximum number of documents per batch to avoid token limits

        Raises:
            VectorStoreError: If store is not initialized or operation fails
        """
        if not self._initialized and not self.vectorstore:
            raise VectorStoreError("ChromaVectorStore is not initialized")

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                self.vectorstore.add_documents(batch)
                batch_num = i // batch_size + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                logger.debug(
                    f"Added batch {batch_num}/{total_batches} "
                    f"({len(batch)} documents)"
                )
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    def get_collection(self):
        """
        Get direct access to ChromaDB collection when LangChain API is insufficient.

        Used only for operations that LangChain doesn't provide (like delete by metadata).

        Returns:
            ChromaDB collection object for advanced operations
        """
        client = PersistentClient(path=str(self.persist_directory))
        try:
            return client.get_collection(name=self.collection_name)
        except Exception:
            return client.get_or_create_collection(name=self.collection_name)

    def document_exists(self, source_path: str = "") -> bool:
        """
        Check if document already exists in the vector store.

        Args:
            source_path: Path to the source document

        Returns:
            True if document exists in the store

        Raises:
            VectorStoreError: If store is not initialized
        """
        if not self._initialized or not self.vectorstore:
            raise VectorStoreError("ChromaVectorStore is not initialized")

        try:
            results = self.vectorstore.similarity_search(
                query="", k=1, filter={"source": source_path}
            )
            return len(results) > 0
        except Exception:
            return False

    def delete_vectors_by_source(self, source_path: str) -> None:
        """
        Delete all vectors associated with a source document.

        Args:
            source_path: Path to the source document

        Raises:
            VectorStoreError: If store is not initialized or operation fails
        """
        if not self._initialized or not self.vectorstore:
            raise VectorStoreError("ChromaVectorStore is not initialized")

        try:
            results = self.vectorstore.similarity_search(
                query="",
                k=10000,
                filter={"source": source_path},
            )

            if results:
                collection = self.get_collection()
                collection.delete(where={"source": source_path})
        except Exception as e:
            print(f"Failed to delete vectors from ChromaDB for document: {source_path}")
            raise VectorStoreError(f"Failed to delete vectors: {e}") from e

    def as_retriever(self, **kwargs):
        """
        Get a LangChain retriever interface for this vector store.

        Args:
            **kwargs: Additional arguments for retriever configuration
                     (e.g., search_type, search_kwargs)

        Returns:
            LangChain retriever object for similarity search

        Raises:
            VectorStoreError: If store is not initialized
        """
        if not self._initialized and not self.vectorstore:
            raise VectorStoreError("ChromaVectorStore is not initialized")

        return self.vectorstore.as_retriever(**kwargs)
