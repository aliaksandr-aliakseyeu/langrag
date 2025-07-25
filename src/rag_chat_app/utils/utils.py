from typing import List, Tuple


def format_chat_history(history: List[Tuple[str, str]]) -> str:
    return "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])
