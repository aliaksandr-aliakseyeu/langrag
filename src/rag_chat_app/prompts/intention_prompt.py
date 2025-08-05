from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_chat_app.enums import UserIntent


class IntentClassificationResult(BaseModel):
    intent: str = Field(description='The classified intent type')
    parameters: Dict[str, Any] = Field(description='Extracted parameters from the query')
    confidence: float = Field(description='Confidence score between 0 and 1')
    reasoning: str = Field(description='Brief explanation of the classification')


@dataclass
class IntentExample:
    query: str
    intent: UserIntent
    parameters: Dict[str, Any]
    confidence: float = 1.0


class IntentPromtManager:

    def __init__(self):
        self.examples = self._get_default_examples()
        self.outputparser = PydanticOutputParser(pydantic_object=IntentClassificationResult)

    def _get_default_examples(self) -> List[IntentExample]:
        return [
            IntentExample(
                query="What is the main topic of document.pdf?",
                intent=UserIntent.SUMMARIZE_DOCUMENT,
                parameters={"document_name": "document.pdf"}
            ),
            IntentExample(
                query="Which documents mention artificial intelligence?",
                intent=UserIntent.GET_DOCUMENT_NAMES,
                parameters={"search_term": "artificial intelligence"}
            ),
            IntentExample(
                query="How does machine learning work according to the documents?",
                intent=UserIntent.SEARCH_DOCUMENTS,
                parameters={"search_term": "machine learning"}
            ),
            IntentExample(
                query="What are the key findings about neural networks?",
                intent=UserIntent.SEARCH_DOCUMENTS,
                parameters={"search_term": "neural networks"}
            ),
            IntentExample(
                query="Can you summarize the research paper on transformers?",
                intent=UserIntent.SUMMARIZE_DOCUMENT,
                parameters={"search_term": "transformers", "document_type": "research paper"}
            ),
            IntentExample(
                query="Hello, how are you?",
                intent=UserIntent.CHAT_GENERAL,
                parameters={}
            ),
        ]

    def add_example(self, example: IntentExample):
        self.examples.append(example)

    def create_intent_prompt(self) -> ChatPromptTemplate:

        examples_text = self._format_examples()
        format_instructions = self.outputparser.get_format_instructions()
        intent_types = UserIntent.all_with_description()

        system_template = """
            You are an intent classifier for a document search and retrieval system.
            Your job is to analyze user queries and determine what they want to do.

            Available intent types:
            {intent_types}

            Examples:
            {examples_text}

            Instructions:
            1. Analyze the user query carefully
            2. Determine the most appropriate intent
            3. Extract relevant parameters (search terms, document names, etc.)
            4. Provide a confidence score (0.0-1.0)
            5. Give a brief reasoning for your classification

            {format_instructions}
        """

        human_template = """
            {chat_history}
            User Query: {query}
        """

        return ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('human', human_template)
        ]).partial(
            intent_types=intent_types,
            examples_text=examples_text,
            format_instructions=format_instructions
        )

    def get_output_parser(self) -> PydanticOutputParser:
        return self.outputparser

    def _format_examples(self) -> str:
        formated_examples = []

        for example in self.examples:
            formated_examples.append(
                f'query: {example.query} \n'
                f'intent: {example.intent.value} \n'
                f'parameters: {example.parameters} \n'
                f'confidence: {example.confidence}'
            )

        return '\n\n'.join(formated_examples)
