import logging
from typing import Any, Dict, List, Tuple

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
)

from rag_chat_app.intent.enums import UserIntent
from rag_chat_app.prompts.prompt_factory import create_prompt_builder
from rag_chat_app.llm.llm_service import LLMService
from .retrievers import IntentRetrieverFactory

logger = logging.getLogger(__name__)


class RetrievalManager:
    """
    Manager for orchestrating document retrieval and response generation.

    This class handles:
    - Intent-specific retrieval configuration
    - Document formatting with sources
    - Chain building and execution
    """

    def __init__(self, vector_store, llm_service: LLMService):
        """
        Initialize the retrieval manager.

        Args:
            vector_store: Vector store for document retrieval
            llm_service: LLM service for creating language models
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.chat_llm = llm_service.create_chat_llm()
        self.retriever_factory = IntentRetrieverFactory(vector_store=vector_store)

    def format_docs_with_sources(self, docs: List[Document]) -> str:
        """Format documents grouped by source to avoid duplication.

        Args:
            docs: List of Document objects from vector store retrieval

        Returns:
            Formatted string with documents grouped by source
        """
        if not docs:
            logger.warning("No documents retrieved for formatting")
            return "No relevant documents found."

        source_groups = {}
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc.page_content)

        logger.debug(
            f"Formatting {len(docs)} documents from {len(source_groups)} unique sources: {set(source_groups.keys())}"
        )

        formatted_docs = []
        for source, contents in source_groups.items():
            if len(contents) == 1:
                formatted_docs.append(f"[Source: {source}]\n{contents[0]}\n")
            else:
                combined_content = "\n\n--- Section Break ---\n\n".join(contents)
                formatted_docs.append(f"[Source: {source}]\n{combined_content}\n")

        return "\n".join(formatted_docs)

    def build_chain(
        self,
        intent: UserIntent,
        chat_history: List[Tuple[str, str]],
        params: Dict[str, Any] = None,
    ) -> RunnableSequence:
        """Build LangChain retrieval chain for the given intent.

        Args:
            intent: User intent determining retrieval strategy and prompt
            chat_history: Previous conversation context (list of Q&A tuples)
            params: Optional parameters for retriever configuration

        Returns:
            Configured LangChain RunnableSequence ready for execution
        """
        params = params or {}
        logger.debug(f"Building chain for intent: {intent.value}, params: {params}")

        retriever = self.retriever_factory.get_retriever(intent, params)
        prompt_builder = create_prompt_builder(intent)

        logger.debug(
            f"Using retriever: {type(retriever).__name__}, prompt builder: {type(prompt_builder).__name__}"
        )

        prompt = prompt_builder.build_prompt()

        formatted_history = prompt_builder.format_chat_history(chat_history)

        chain = (
            {
                "context": retriever | self.format_docs_with_sources,
                "input": RunnablePassthrough(),
                "chat_history": lambda _: formatted_history,
            }
            | prompt
            | self.chat_llm
            | StrOutputParser()
        )

        return chain

    def run(
        self,
        intent: UserIntent,
        chat_history: List[Tuple[str, str]],
        message: str,
        params: Dict[str, Any] = None,
    ) -> str:
        """Execute complete retrieval pipeline for the intent.

        Args:
            intent: User intent determining the type of response to generate
            chat_history: Previous conversation context for better responses
            message: User's current message/question
            params: Optional parameters for specific intents (e.g., document_name)

        Returns:
            Generated response as a string
        """
        logger.info(
            f"Executing retrieval for intent: {intent.value}, message length: {len(message)}"
        )

        try:
            chain = self.build_chain(intent, chat_history, params)
            result = chain.invoke(message)

            logger.info(f"Successfully generated response ({len(result)} chars)")
            return result

        except Exception as e:
            logger.error(
                f"Failed to execute retrieval chain for intent {intent.value}: {e}",
                exc_info=True,
            )
            raise
