from typing import List, Tuple


def format_chat_history(history: List[Tuple[str, str]]) -> str:
    """
    Format chat history into a readable string.

    Converts a list of (user_message, assistant_response) tuples into
    a formatted string for use in prompts or logging.

    Args:
        history: List of (user_message, assistant_response) tuples

    Returns:
        Formatted string with labeled user and assistant messages
    """
    return "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])
