from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .enums import LLMProvider
from .llm_config import LLMConfig


class LLMService:
    """
    Factory service for creating LLM instances.

    This service provides a unified interface for creating different LLM models
    for various tasks like intent classification and chat responses.
    """

    def __init__(self, config: LLMConfig = None):
        """
        Initialize LLM service with configuration.

        Args:
            config: LLM configuration. Uses settings if not provided.
        """
        self.config = config or LLMConfig.from_settings()

    def create_llm(
        self, provider: LLMProvider, model: str, temperature: float = 0.0
    ) -> Runnable:
        """
        Universal method for creating LLM instances.

        Args:
            provider: LLM provider (OpenAI, Ollama, etc.)
            model: Model name/identifier
            temperature: Temperature for response generation (0.0 = deterministic)

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If provider is unsupported
        """
        if provider == LLMProvider.OPENAI:
            return ChatOpenAI(model=model, temperature=temperature)

        if provider == LLMProvider.OLLAMA:
            return Ollama(model=model)

        raise ValueError(f"Unsupported provider: {provider}")

    def create_intent_llm(self) -> Runnable:
        """
        Create LLM for intent classification.

        Uses configuration with temperature=0 for deterministic results.

        Returns:
            Configured LLM for intent classification
        """
        return self.create_llm(
            provider=self.config.intent_provider,
            model=self.config.get_intent_model_string(),
            temperature=self.config.intent_temperature,
        )

    def create_chat_llm(self) -> Runnable:
        """
        Create LLM for chat responses.

        Uses configuration with slightly higher temperature for
        more natural responses.

        Returns:
            Configured LLM for chat responses
        """
        return self.create_llm(
            provider=self.config.chat_provider,
            model=self.config.get_chat_model_string(),
            temperature=self.config.chat_temperature,
        )
