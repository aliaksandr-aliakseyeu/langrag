from typing import Any, Dict, List
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.runnables import Runnable, RunnableLambda

from rag_chat_app.intent.enums import UserIntent


class IntentRetrieverFactory:
    """
    Factory for creating intent-specific retrievers.

    This class provides retrievers optimized for different types of queries:
    - Document search (similarity search)
    - Document name listing (broader search)
    - Document summarization (comprehensive retrieval)
    """

    def __init__(self, vector_store):
        """
        Initialize the retriever factory.

        Args:
            vector_store: Vector store for document retrieval
        """
        self.vector_store = vector_store

    def get_retriever(
        self, intent: UserIntent, params: Dict[str, Any] = None
    ) -> BaseRetriever:
        """Get retriever optimized for the intent type.

        Args:
            intent: User intent determining retrieval strategy
            params: Additional parameters for retriever configuration

        Returns:
            Configured retriever for the intent
        """
        params = params or {}

        if intent == UserIntent.SEARCH_DOCUMENTS:
            return self._get_search_retriever(params)
        elif intent == UserIntent.GET_DOCUMENT_NAMES:
            return self._get_document_names_retriever(params)
        elif intent == UserIntent.SUMMARIZE_DOCUMENT:
            return self._get_summarize_document_retriever(params)
        else:
            return self._get_default_retriever(params)

    def _get_search_retriever(self, params: Dict[str, Any]) -> BaseRetriever:
        """Get retriever for document search (k=5).

        Args:
            params: Additional parameters (currently unused)

        Returns:
            BaseRetriever configured for document search
        """
        return self.vector_store.as_retriever(k=5, search_type="similarity")

    def _get_document_names_retriever(self, params: Dict[str, Any]) -> BaseRetriever:
        """Get retriever for finding document names (k=10).

        Args:
            params: Additional parameters (currently unused)

        Returns:
            BaseRetriever configured for document name discovery
        """
        return self.vector_store.as_retriever(k=10, search_type="similarity")

    def _get_summarize_document_retriever(self, params: Dict[str, Any]) -> Runnable:
        """Get retriever for document summarization.

        Args:
            params: Configuration parameters:
                   - document_name (str): Specific document to summarize
                   - search_term (str): Search term for topic-based summarization

        Returns:
            Runnable configured for comprehensive document retrieval

        Raises:
            ValueError: If neither document_name nor search_term is provided
        """
        document_name = params.get("document_name", None)
        search_term = params.get("search_term", None)

        if search_term:
            return self.vector_store.as_retriever(
                search_type="similarity",
                k=50,
                filter={"file_name": document_name} if document_name else None,
            )

        if document_name:

            def fetch_chunks_by_file_name(_query: str) -> List[Document]:
                """Fetch all chunks for a specific document."""
                return self.vector_store.vectorstore.similarity_search(
                    query="", k=100, filter={"file_name": document_name}
                )

            return RunnableLambda(fetch_chunks_by_file_name)

        raise ValueError(
            "Either 'document_name' or 'search_term' must be provided for summarization."
        )

    def _get_default_retriever(self, params: Dict[str, Any]) -> BaseRetriever:
        """Get default retriever for general queries (k=7).

        Args:
            params: Additional parameters (currently unused)

        Returns:
            BaseRetriever configured with balanced parameters
        """
        return self.vector_store.as_retriever(k=7, search_type="similarity")
