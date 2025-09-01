from dataclasses import dataclass
from typing import Union

from .enums import LLMProvider, OpenAIModel, OllamaModel, HuggingFaceModel
from .llm_registry import MODEL_ENUM_MAP
from rag_chat_app.config import settings


@dataclass
class LLMConfig:
    """
    Simple configuration for LLM services.

    Uses existing enum validation system. Create manually for custom settings:

    Example:
        # Default from settings
        config = LLMConfig.from_settings()

        # Custom user config
        config = LLMConfig(
            intent_model=OpenAIModel.GPT_4O,
            chat_model=OpenAIModel.GPT_3_5,
            chat_temperature=0.2
        )
    """

    # Intent classification settings
    intent_provider: LLMProvider = LLMProvider.OPENAI
    intent_model: Union[OpenAIModel, OllamaModel, HuggingFaceModel] = (
        OpenAIModel.GPT_4O_MINI
    )
    intent_temperature: float = 0.0

    # Chat response settings
    chat_provider: LLMProvider = LLMProvider.OPENAI
    chat_model: Union[OpenAIModel, OllamaModel, HuggingFaceModel] = (
        OpenAIModel.GPT_4O_MINI
    )
    chat_temperature: float = 0.1

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_provider_model_pair(self.intent_provider, self.intent_model)
        self._validate_provider_model_pair(self.chat_provider, self.chat_model)

    def _validate_provider_model_pair(self, provider: LLMProvider, model):
        """
        Validate that model enum matches provider using existing MODEL_ENUM_MAP.

        Args:
            provider: LLM provider enum
            model: Model enum

        Raises:
            ValueError: If provider and model don't match
        """
        expected_enum_class = MODEL_ENUM_MAP.get(provider)
        if not expected_enum_class:
            raise ValueError(f"Unsupported provider: {provider}")

        if not isinstance(model, expected_enum_class):
            raise ValueError(
                f"Model {model} is not valid for provider {provider}. "
                f"Expected: {expected_enum_class.__name__}"
            )

    @classmethod
    def from_settings(cls) -> "LLMConfig":
        """Create configuration from application settings."""
        return cls(
            intent_provider=settings.LLM_INTENTION_PROVIDER,
            intent_model=settings.LLM_INTENTION_MODEL,
            intent_temperature=0.0,
            chat_provider=settings.LLM_INTENTION_PROVIDER,  # Using same for now
            chat_model=settings.LLM_INTENTION_MODEL,  # Using same for now
            chat_temperature=0.1,
        )

    def get_intent_model_string(self) -> str:
        """Get intent model as string value."""
        return self.intent_model.value

    def get_chat_model_string(self) -> str:
        """Get chat model as string value."""
        return self.chat_model.value
