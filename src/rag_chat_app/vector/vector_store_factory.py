from typing import Optional
from langchain.embeddings.base import Embeddings

from rag_chat_app.config import settings
from .stores.base import VectorStore


def create_chroma_vector_store(
    embedding_function: Embeddings,
    collection_name: str = "documents",
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Create a ChromaDB vector store.

    Args:
        embedding_function: Embedding model to use
        collection_name: Name of the collection
        persist_directory: Directory to persist the database

    Returns:
        ChromaVectorStore instance
    """
    from .stores.chroma_store import ChromaVectorStore

    persist_dir = persist_directory or settings.VECTOR_FOLDER

    try:
        store = ChromaVectorStore(
            embedding_function=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        store.initialize()
        return store
    except Exception as e:
        raise ValueError(f"Failed to create ChromaDB vector store: {e}") from e


def create_pinecone_vector_store(
    embedding_function: Embeddings,
    index_name: str,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,
) -> VectorStore:
    """
    Create a Pinecone vector store.

    Args:
        embedding_function: Embedding model to use
        index_name: Name of the Pinecone index
        api_key: Pinecone API key
        environment: Pinecone environment

    Returns:
        Pinecone VectorStore instance

    Note:
        This is a placeholder. You would implement PineconeVectorStore
        following the same interface as ChromaVectorStore.
    """
    # TODO: Implement PineconeVectorStore
    raise NotImplementedError("Pinecone vector store not yet implemented")
