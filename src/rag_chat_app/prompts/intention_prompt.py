from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_chat_app.intent.enums import UserIntent


class IntentClassificationResult(BaseModel):
    intent: str = Field(description="The classified intent type")
    parameters: Dict[str, Any] = Field(
        description="Extracted parameters from the query"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the classification")


@dataclass
class IntentExample:
    query: str
    intent: UserIntent
    parameters: Dict[str, Any]
    confidence: float = 1.0


class IntentPromtManager:
    """
    Manager for intent classification prompts and output parsing.

    This class handles the complete prompt-response cycle for intent classification:
    - Manages training examples for few-shot learning
    - Creates structured prompts with format instructions
    - Provides output parser for structured LLM responses

    Note: Output parser is fixed to PydanticOutputParser with IntentClassificationResult
    to ensure type safety and compatibility with the intent pipeline.
    """

    def __init__(self):
        """
        Initialize intent prompt manager with default examples and output parser.

        Creates a fixed PydanticOutputParser for IntentClassificationResult to ensure
        type safety and compatibility with the intent classification pipeline.
        """
        self.examples = self._get_default_examples()
        self.outputparser = PydanticOutputParser(
            pydantic_object=IntentClassificationResult
        )

    def _get_default_examples(self) -> List[IntentExample]:
        """
        Get default training examples for intent classification.

        Returns:
            List of IntentExample objects for few-shot learning
        """
        return [
            IntentExample(
                query="What is the main topic of document.pdf?",
                intent=UserIntent.SUMMARIZE_DOCUMENT,
                parameters={"document_name": "document.pdf"},
            ),
            IntentExample(
                query="Which documents mention artificial intelligence?",
                intent=UserIntent.GET_DOCUMENT_NAMES,
                parameters={"search_term": "artificial intelligence"},
            ),
            IntentExample(
                query="How does machine learning work according to the documents?",
                intent=UserIntent.SEARCH_DOCUMENTS,
                parameters={"search_term": "machine learning"},
            ),
            IntentExample(
                query="What are the key findings about neural networks?",
                intent=UserIntent.SEARCH_DOCUMENTS,
                parameters={"search_term": "neural networks"},
            ),
            IntentExample(
                query="Can you summarize the research paper on transformers?",
                intent=UserIntent.SUMMARIZE_DOCUMENT,
                parameters={
                    "search_term": "transformers",
                    "document_type": "research paper",
                },
            ),
            IntentExample(
                query="Hello, how are you?",
                intent=UserIntent.CHAT_GENERAL,
                parameters={},
            ),
        ]

    def add_example(self, example: IntentExample):
        """
        Add a new training example for intent classification.

        Args:
            example: IntentExample with query, intent, and parameters
        """
        self.examples.append(example)

    def create_intent_prompt(self) -> ChatPromptTemplate:
        """
        Create a structured prompt template for intent classification.

        Builds a comprehensive prompt that includes:
        - Available intent types with descriptions
        - Few-shot examples for better classification
        - Format instructions from the output parser
        - Clear instructions for the LLM

        Returns:
            ChatPromptTemplate ready for LangChain use
        """

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
            2. This is a RAG (Retrieval-Augmented Generation) system - DEFAULT to "search_documents" for most queries
            3. Use "search_documents" for ANY question that could be answered by searching through documents:
               - Requirements, procedures, processes (e.g., "what do I need for X")
               - How-to questions (e.g., "how to apply for Y")
               - Information requests (e.g., "tell me about Z")
               - Any factual questions that might have answers in documents
            4. Determine the most appropriate intent
            5. Extract relevant parameters (search terms, document names, etc.)
            6. Provide a confidence score (0.0-1.0)
            7. Give a brief reasoning for your classification

            {format_instructions}
        """

        human_template = """
            {chat_history}
            User Query: {query}
        """

        return ChatPromptTemplate.from_messages(
            [("system", system_template), ("human", human_template)]
        ).partial(
            intent_types=intent_types,
            examples_text=examples_text,
            format_instructions=format_instructions,
        )

    def get_output_parser(self) -> PydanticOutputParser:
        """
        Get the output parser for structured LLM responses.

        Returns:
            PydanticOutputParser configured for IntentClassificationResult
        """
        return self.outputparser

    def _format_examples(self) -> str:
        """
        Format training examples into text for the prompt.

        Returns:
            Formatted string with all examples for few-shot learning
        """
        formated_examples = []

        for example in self.examples:
            formated_examples.append(
                f"query: {example.query} \n"
                f"intent: {example.intent.value} \n"
                f"parameters: {example.parameters} \n"
                f"confidence: {example.confidence}"
            )

        return "\n\n".join(formated_examples)
