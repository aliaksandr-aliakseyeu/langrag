from enum import Enum


class VectorStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def choices(cls):
        return [status.value for status in cls]

    @classmethod
    def default(cls):
        return cls.PENDING

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            print(f"Warning: Invalid vector_status '{value}', using default")
            return cls.default()


class LLMProvider(str, Enum):
    OPENAI = 'openai'
    OLLAMA = 'ollama'
    HUGGINGFACE = 'huggingface'


class OpenAIModel(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3_5 = "gpt-3.5-turbo"


class OllamaModel(str, Enum):
    MISTRAL = "mistral"
    LLAMA3 = "llama3"
    GEMMA = "gemma"


class HuggingFaceModel(str, Enum):
    ALL_MINI_LM = "all-MiniLM-L6-v2"


class UserIntent(str, Enum):
    SEARCH_DOCUMENTS = "search_documents"
    GET_DOCUMENT_NAMES = "get_document_names"
    SUMMARIZE_DOCUMENT = "summarize_document"
    CHAT_GENERAL = "chat_general"
    UNKNOWN = 'unknown'

    def description(self) -> str:
        return {
            UserIntent.SEARCH_DOCUMENTS: 'User wants to find specific information within documents',
            UserIntent.GET_DOCUMENT_NAMES: 'User wants to know which document(s) contain certain information.',
            UserIntent.SUMMARIZE_DOCUMENT: 'User wants a summary of a specific document or documents about a topic',
            UserIntent.CHAT_GENERAL: 'chat_general: General conversation not related to documents',
            UserIntent.UNKNOWN: 'Unknown or unrecognized intention.',
        }.get(self, 'Unknown intent.')

    @classmethod
    def all_with_description(cls) -> str:
        return '\n'.join(f'{intent.value}: {intent.description()}' for intent in cls)
