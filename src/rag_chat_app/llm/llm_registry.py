from rag_chat_app.enums import LLMProvider, OpenAIModel, OllamaModel, HuggingFaceModel


MODEL_ENUM_MAP = {
    LLMProvider.OPENAI: OpenAIModel,
    LLMProvider.OLLAMA: OllamaModel,
    LLMProvider.HUGGINGFACE: HuggingFaceModel,
}
