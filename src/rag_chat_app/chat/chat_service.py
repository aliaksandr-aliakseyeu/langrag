"""
Chat service that orchestrates the complete RAG pipeline.

This service handles:
1. Intent classification (first chain)
2. Intent-specific retrieval and response (second chain)
"""

import logging
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass

from rag_chat_app.intent.enums import UserIntent
from rag_chat_app.prompts.intention_prompt import (
    IntentClassificationResult,
    IntentPromtManager,
)
from rag_chat_app.retrieval.retrieval_manager import RetrievalManager
from rag_chat_app.vector.stores.base import VectorStore
from rag_chat_app.llm.llm_service import LLMService
from rag_chat_app.intent.intent_manager import IntentManager


logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response from the chat service."""

    answer: str
    intent: str
    confidence: float
    reasoning: str
    sources: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatService:
    """
    Main chat service that orchestrates intent detection and retrieval.

    This service implements a two-stage RAG pipeline:
    1. Intent Classification: Determine what the user wants to do
    2. Intent-specific Retrieval: Execute the appropriate retrieval chain
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        intent_confidence_threshold: float = 0.7,
    ):
        """
        Initialize the chat service.

        Args:
            vector_store: Vector store for document retrieval
            llm_service: LLM service for inference
            intent_confidence_threshold: Minimum confidence for intent classification
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        intent_prompt_manager = self._create_intent_prompt_manager()
        self.intent_manager = IntentManager(
            llm_service=llm_service,
            intent_prompt_manager=intent_prompt_manager,
            confidence_threshold=intent_confidence_threshold,
        )
        self.retrieval_manager = RetrievalManager(
            vector_store=vector_store, llm_service=llm_service
        )

        logger.info("ChatService initialized successfully")

    def _create_intent_prompt_manager(self):
        """
        Create and customize intent prompt manager.

        This method can be overridden to customize intent examples
        and prompt behavior for specific use cases.

        Returns:
            Configured IntentPromtManager instance
        """
        # Create default prompt manager
        prompt_manager = IntentPromtManager()

        # Example: Add custom examples for better classification
        # prompt_manager.add_example(IntentExample(
        #     query="Show me all documents about legal requirements",
        #     intent=UserIntent.GET_DOCUMENT_NAMES,
        #     parameters={"search_term": "legal requirements"}
        # ))

        return prompt_manager

    def chat(
        self, message: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> ChatResponse:
        """
        Process a user message through the complete RAG pipeline.

        Args:
            message: User's message/question
            chat_history: Previous conversation history

        Returns:
            ChatResponse with answer and metadata
        """
        chat_history = chat_history or []

        try:
            intent_result = self.intent_manager.classify_intent(message, chat_history)

            if self.intent_manager.is_high_confidence(intent_result):
                answer = self._process_with_intent(message, intent_result, chat_history)
            else:
                logger.warning(
                    "Low intent confidence (%.2f), falling back to general chat",
                    intent_result.confidence,
                )
                answer = self._process_general_chat(message, chat_history)

            return ChatResponse(
                answer=answer,
                intent=intent_result.intent,
                confidence=intent_result.confidence,
                reasoning=intent_result.reasoning,
                parameters=intent_result.parameters,
            )

        except Exception as e:
            logger.error("Error processing chat message", exc_info=True)
            return ChatResponse(
                answer="I apologize, but I encountered an error processing your request. Please try again.",
                intent=UserIntent.UNKNOWN.value,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
            )

    def _process_with_intent(
        self,
        message: str,
        intent_result: IntentClassificationResult,
        chat_history: List[Tuple[str, str]],
    ) -> str:
        """Process message using intent-specific retrieval chain."""
        intent = self.intent_manager.get_intent_enum(intent_result)
        recent_history = chat_history[-3:] if chat_history else []
        return self.retrieval_manager.run(
            intent=intent,
            chat_history=recent_history,
            message=message,
            params=intent_result.parameters,
        )

    def _process_general_chat(
        self, message: str, chat_history: List[Tuple[str, str]]
    ) -> str:
        """Process message as general chat."""
        recent_history = chat_history[-3:] if chat_history else []

        return self.retrieval_manager.run(
            intent=UserIntent.CHAT_GENERAL, chat_history=recent_history, message=message
        )
