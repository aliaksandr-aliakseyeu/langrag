from rag_chat_app.intent.enums import UserIntent
from .base import BasePromptBuilder
from .prompt_builders import (
    SearchDocumentsPromptBuilder,
    GetDocumentNamesBuilder,
    SummarizeDocumentPromptBuilder,
    ChatGeneralPromptBuilder,
)


# Mapping of intents to their prompt builder classes
PROMPT_BUILDER_MAP = {
    UserIntent.SEARCH_DOCUMENTS: SearchDocumentsPromptBuilder,
    UserIntent.GET_DOCUMENT_NAMES: GetDocumentNamesBuilder,
    UserIntent.SUMMARIZE_DOCUMENT: SummarizeDocumentPromptBuilder,
    UserIntent.CHAT_GENERAL: ChatGeneralPromptBuilder,
    UserIntent.UNKNOWN: ChatGeneralPromptBuilder,
}


def create_prompt_builder(intent: UserIntent) -> BasePromptBuilder:
    """
    Create appropriate prompt builder for the given intent.

    Args:
        intent: User intent type

    Returns:
        Instance of the appropriate prompt builder

    Raises:
        ValueError: If intent is not supported (should not happen with complete mapping)
    """
    builder_class = PROMPT_BUILDER_MAP.get(intent)

    if not builder_class:
        raise ValueError(f"No prompt builder defined for intent: {intent}")

    return builder_class()


def get_supported_intents() -> list[UserIntent]:
    """
    Get list of intents that have prompt builders.

    Returns:
        List of supported UserIntent enums
    """
    return list(PROMPT_BUILDER_MAP.keys())


def is_intent_supported(intent: UserIntent) -> bool:
    """
    Check if intent has a prompt builder.

    Args:
        intent: User intent to check

    Returns:
        True if intent is supported
    """
    return intent in PROMPT_BUILDER_MAP
