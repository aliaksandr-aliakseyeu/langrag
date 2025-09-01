from enum import Enum


class LLMProvider(str, Enum):
    """Available LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class OpenAIModel(str, Enum):
    """Available OpenAI models."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3_5 = "gpt-3.5-turbo"


class OllamaModel(str, Enum):
    """Available Ollama models."""

    MISTRAL = "mistral"
    LLAMA3 = "llama3"
    GEMMA = "gemma"


class HuggingFaceModel(str, Enum):
    """Available HuggingFace models."""

    ALL_MINI_LM = "all-MiniLM-L6-v2"
