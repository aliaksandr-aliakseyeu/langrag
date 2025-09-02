from rag_chat_app.prompts.base import BasePromptBuilder

from langchain_core.prompts import ChatPromptTemplate


class SearchDocumentsPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for document search queries.

    Creates prompts optimized for answering questions by searching through
    document content and providing contextual answers with source citations.
    """

    def build_prompt(self):
        template = """
            You are a helpful assistant that searches through documents to answer questions.
            Use the following pieces of context to answer the user's question accurately.

            IMPORTANT RULES:
            - Only use information from the provided context
            - If the answer cannot be found in the context, say "I don't have enough information to answer that question"
            - Be specific and cite relevant details from the context
            - Present each document only once, even if multiple sections are relevant
            - CRITICAL: You MUST always end your answer with "Sources: [filename1, filename2, ...]"

            Context from documents (each section shows source):
            {context}

            Previous conversation:
            {chat_history}

            Current question: {input}

            Answer based on the context above and ALWAYS end with sources list:
        """

        return ChatPromptTemplate.from_template(template)


class GetDocumentNamesBuilder(BasePromptBuilder):
    """
    Prompt builder for document name/listing queries.

    Creates prompts for finding and listing relevant documents based on
    search criteria, focusing on document identification rather than content.
    """

    def build_prompt(self) -> ChatPromptTemplate:
        template = """
            You are a document finder assistant.
            Your job is to identify which documents contain information relevant to the user's query.

            Based on the search results below, list the document names/sources that contain relevant information.

            IMPORTANT RULES:
            - Focus on document names, file paths, and sources
            - Briefly explain what type of information each document contains
            - If no relevant documents are found, say so clearly
            - Present the information in a clear, organized format
            - Present each document only once, even if multiple sections are relevant


            Search results from documents:
            {context}

            Previous conversation:
            {chat_history}

            User is looking for documents about: {input}

            Documents that contain relevant information:
        """

        return ChatPromptTemplate.from_template(template)


class SummarizeDocumentPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for document summarization queries.

    Creates prompts for generating comprehensive summaries of document content,
    maintaining structure and highlighting key information.
    """

    def build_prompt(self) -> ChatPromptTemplate:
        template = """
            You are a document summarization assistant. Create a comprehensive summary of the provided document content.

            IMPORTANT RULES:
            - Create a well-structured summary with key points
            - Include main topics, important details, and conclusions
            - Maintain the logical flow of information
            - If the content is incomplete, mention what sections are covered
            - Use bullet points or sections for better readability
            - Present each document only once, even if multiple sections are relevant
            - CRITICAL: You MUST always end your summary with "Sources: [filename1, filename2, ...]"

            Document content to summarize:
            {context}

            Previous conversation:
            {chat_history}

            Summarization request: {input}

            Document Summary (remember to end with sources):
        """

        return ChatPromptTemplate.from_template(template)


class ChatGeneralPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for general chat conversations.

    Creates prompts for handling general conversations that don't require
    specific document operations, while still making context available if relevant.
    """

    def build_prompt(self) -> ChatPromptTemplate:
        template = """
            You are a helpful assistant.
            The user is having a general conversation that may not be directly related to document search.

            You can use the context if it's relevant, but focus on being helpful and conversational.

            Available context (may not be relevant):
            {context}

            Previous conversation:
            {chat_history}

            User message: {input}

            Response:
        """

        return ChatPromptTemplate.from_template(template)
