"""
Intent manager for handling user intent classification.

This module manages the complete intent classification pipeline:
- Intent prompt creation
- LLM chain construction
- Intent classification logic
"""

import logging
from typing import List, Tuple, Optional

from .enums import UserIntent
from rag_chat_app.prompts.intention_prompt import (
    IntentPromtManager,
    IntentClassificationResult,
)
from rag_chat_app.llm.llm_service import LLMService
from rag_chat_app.utils.utils import format_chat_history

logger = logging.getLogger(__name__)


class IntentManager:
    """
    Manages intent classification for user messages.

    This class encapsulates:
    - Intent prompt management
    - Intent classification chain
    - Confidence threshold handling
    """

    def __init__(
        self,
        llm_service: LLMService,
        intent_prompt_manager: IntentPromtManager,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the intent manager.

        Args:
            llm_service: LLM service for intent classification
            intent_prompt_manager: Manager for intent classification prompts
            confidence_threshold: Minimum confidence for intent classification
        """
        self.llm_service = llm_service
        self.intent_prompt_manager = intent_prompt_manager
        self.confidence_threshold = confidence_threshold

        self.intent_llm = self.llm_service.create_intent_llm()
        self.intent_prompt = self.intent_prompt_manager.create_intent_prompt()
        self.intent_parser = self.intent_prompt_manager.get_output_parser()

        self.intent_chain = self.intent_prompt | self.intent_llm | self.intent_parser

        logger.info(
            "IntentManager initialized with confidence threshold: %.2f",
            confidence_threshold,
        )

    def classify_intent(
        self, message: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> IntentClassificationResult:
        """
        Classify the intent of a user message.

        Args:
            message: User's message to classify
            chat_history: Previous conversation history for context

        Returns:
            IntentClassificationResult with intent, confidence, and parameters

        Raises:
            Exception: If intent classification fails
        """
        if not message.strip():
            return IntentClassificationResult(
                intent=UserIntent.CHAT_GENERAL.value,
                parameters={},
                confidence=1.0,
                reasoning="Empty message received",
            )

        chat_history = chat_history or []

        try:
            logger.info("Classifying intent for message: %s", message[:100])

            recent_history = chat_history[-2:] if chat_history else []
            inputs = {
                "query": message,
                "chat_history": format_chat_history(recent_history),
            }
            result = self.intent_chain.invoke(inputs)

            logger.info(
                "Intent classified: %s (confidence: %.2f, reasoning: %s)",
                result.intent,
                result.confidence,
                result.reasoning[:100],
            )

            return result

        except Exception as e:
            logger.error("Failed to classify intent", exc_info=True)

            return IntentClassificationResult(
                intent=UserIntent.UNKNOWN.value,
                parameters={},
                confidence=0.0,
                reasoning=f"Intent classification failed: {str(e)}",
            )

    def is_high_confidence(self, intent_result: IntentClassificationResult) -> bool:
        """
        Check if the intent classification has high enough confidence.

        Args:
            intent_result: Result from classify_intent()

        Returns:
            True if confidence is above threshold
        """
        return intent_result.confidence >= self.confidence_threshold

    def get_intent_enum(self, intent_result: IntentClassificationResult) -> UserIntent:
        """
        Convert intent string to enum, with fallback to CHAT_GENERAL.

        Args:
            intent_result: Result from classify_intent()

        Returns:
            UserIntent enum value
        """
        try:
            return UserIntent(intent_result.intent)
        except ValueError:
            logger.warning(
                "Unknown intent string: %s, falling back to CHAT_GENERAL",
                intent_result.intent,
            )
            return UserIntent.CHAT_GENERAL
