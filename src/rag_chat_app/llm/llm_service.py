from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from rag_chat_app.enums import LLMProvider
from rag_chat_app.llm.llm_registry import MODEL_ENUM_MAP
from rag_chat_app.config import settings


class LLMService:

    def get_user_llm(
        self,
        provider: LLMProvider,
        model: str,
        temperature: float = 0.0
    ) -> Runnable:
        self._validate_model_for_provider(provider, model)

        if provider == LLMProvider.OPENAI:
            return ChatOpenAI(model=model, temperature=temperature)

        if provider == LLMProvider.OLLAMA:
            return Ollama(model=model)

    def get_intent_llm(self):
        provider = settings.LLM_INTENTION_PROVIDER
        model = settings.LLM_INTENTION_MODEL

        self._validate_model_for_provider(provider, model)

        if provider == LLMProvider.OPENAI:
            return ChatOpenAI(model=model, temperature=0)

        if provider == LLMProvider.OLLAMA:
            return Ollama(model=model)

    def _validate_model_for_provider(self, provider: LLMProvider, model: str,) -> None:
        enum_cls = MODEL_ENUM_MAP.get(provider)

        if not enum_cls:
            raise ValueError(f"No enum defined for provider {provider}")
        try:
            enum_cls(model)
        except ValueError:
            raise ValueError(f"Model '{model}' is invalid for provider '{provider}'")
