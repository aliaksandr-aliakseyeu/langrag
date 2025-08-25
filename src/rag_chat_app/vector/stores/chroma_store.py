from pathlib import Path
from typing import Optional
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from chromadb import PersistentClient

from rag_chat_app.config import settings
from rag_chat_app.vector import VectorStore, VectorStoreError


class ChromaVectorStore(VectorStore):
    def __init__(self,
                 embedding_function: Embeddings,
                 collection_name: str,
                 persist_directory: str = settings.VECTOR_FOLDER):
        super().__init__(embedding_function)
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.vectorstore: Optional[Chroma] = None

    def initialize(self):
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_directory),
            )
            self._initialized = True
        except Exception as e:
            print(f'Failed to initialized ChromaDB: {e}')
            raise VectorStoreError(f'Failed to initialized ChromaDB: {e}') from e

    def add_documents(self, documents: Document) -> None:
        if not self._initialized and not self.vectorstore:
            raise VectorStoreError('ChromaVectorStore is not initialized')

        try:
            self.vectorstore.add_documents(documents)
        except Exception as e:
            print(f"Failed to add documents to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    def get_collection(self):
        client = PersistentClient(path=str(self.persist_directory))
        return client.get_collection(name=self.collection_name)

    def document_exists(self, source_path: str = '') -> bool:
        if not self._initialized and not self.vectorstore:
            raise VectorStoreError('ChromaVectorStore is not initialized')

        collection = self.get_collection()
        result = collection.get(where={'source': source_path}, limit=1)
        return len(result['ids']) > 0

    def delete_vectors_by_source(self, source_path: str) -> None:
        if not self._initialized and not self.vectorstore:
            raise VectorStoreError('ChromaVectorStore is not initialized')

        try:
            collection = self.get_collection()
            collection.delete(where={'source': source_path})
        except Exception as e:
            print(f"Failed to delete vectors from ChromaDB for document: {source_path}")
            raise VectorStoreError(f"Failed to delete vectors: {e}") from e

    def as_retriever(self, **kwargs):
        if not self._initialized and not self.vectorstore:
            raise VectorStoreError('ChromaVectorStore is not initialized')

        return self.vectorstore.as_retriever(**kwargs)
