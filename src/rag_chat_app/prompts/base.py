from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate


class BasePromptBuilder(ABC):
    """
    Abstract base class for prompt builders.

    Each prompt builder creates LangChain prompt templates for specific
    user intents in the RAG system.
    """

    @abstractmethod
    def build_prompt(self) -> ChatPromptTemplate:
        """
        Build a LangChain prompt template for this specific intent.

        Returns:
            ChatPromptTemplate configured for the intended use case
        """
        pass

    def format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        """
        Format chat history for inclusion in prompts.

        Args:
            chat_history: List of (question, answer) tuples

        Returns:
            Formatted string representation of recent conversation
        """
        if not chat_history:
            return "No previous conversation."

        formatted = []

        for question, answer in chat_history[-3:]:  # Last 3 exchanges
            formatted.append(f"Human: {question}")
            formatted.append(f"Assistant: {answer}")

        return "\n".join(formatted)
