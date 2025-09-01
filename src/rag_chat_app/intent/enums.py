from enum import Enum


class UserIntent(str, Enum):
    """User intent types for RAG chat application."""

    SEARCH_DOCUMENTS = "search_documents"
    GET_DOCUMENT_NAMES = "get_document_names"
    SUMMARIZE_DOCUMENT = "summarize_document"
    CHAT_GENERAL = "chat_general"
    UNKNOWN = "unknown"

    def description(self) -> str:
        """Get human-readable description of the intent."""
        return {
            UserIntent.SEARCH_DOCUMENTS: "User wants to find specific information within documents",
            UserIntent.GET_DOCUMENT_NAMES: "User wants to know which document(s) contain certain information.",
            UserIntent.SUMMARIZE_DOCUMENT: "User wants a summary of a specific document or documents about a topic",
            UserIntent.CHAT_GENERAL: "General conversation not related to documents",
            UserIntent.UNKNOWN: "Unknown or unrecognized intention.",
        }.get(self, "Unknown intent.")

    @classmethod
    def all_with_description(cls) -> str:
        """Get all intents with their descriptions."""
        return "\n".join(f"{intent.value}: {intent.description()}" for intent in cls)
