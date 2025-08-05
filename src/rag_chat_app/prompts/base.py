from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate


class BasePromptBuilder(ABC):

    @abstractmethod
    def build_prompt(self) -> ChatPromptTemplate:
        pass

    def format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        if not chat_history:
            return "No previous conversation."

        formated = []

        for question, answer in chat_history[-3:]:
            formated.append(f'Human: {question}')
            formated.append(f'Assistant: {answer}')

        return '\n'.join(formated)
