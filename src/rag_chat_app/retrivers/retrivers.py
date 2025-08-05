from typing import Any, Dict, List, Tuple

from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
)

from rag_chat_app.enums import UserIntent
from rag_chat_app.prompts.prompt_builders import (
    GetDocumentNamesBulder,
    SearchDocumentsPromptBuilder,
    SummarizeDocumentPromptBuilder,
    ChatGeneralPromptBuilder
)


class IntenBaseRetriver:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_retriver(self, intent: UserIntent, params: Dict[str, Any] = None) -> BaseRetriever:

        params = params or {}

        if intent == UserIntent.SEARCH_DOCUMENTS:
            return self._get_search_retriver(params)
        elif intent == UserIntent.GET_DOCUMENT_NAMES:
            return self._get_document_names_retriver(params)
        elif intent == UserIntent.SUMMARIZE_DOCUMENT:
            return self._sumarize_document_retriver(params)
        else:
            return self._default_retriver(params)

    def _get_search_retriver(self, params: Dict[str, Any]) -> BaseRetriever:
        return self.vector_store.as_retriever(k=5, search_type='similarity')

    def _get_document_names_retriver(self, params: Dict[str, Any]) -> BaseRetriever:
        return self.vector_store.as_retriever(k=10, search_type='similarity')

    def _sumarize_document_retriver(self, params: Dict[str, Any]) -> Runnable:
        document_name = params.get('document_name', None)
        search_term = params.get('search_term', None)

        if search_term:
            return self.vector_store.as_retriever(
                search_type='similarity',
                k=50,
                filter={"file_name": document_name} if document_name else None
            )

        if document_name:
            def fetch_chunk_by_file_name(_query: str) -> List[Document]:
                return self.vector_store.vectorstore.similarity_search(
                    query="",
                    k=100,
                    filter={"file_name": document_name}
                )

            return RunnableLambda(fetch_chunk_by_file_name)

        raise ValueError("Either 'document_name' or 'search_term' must be provided.")


class RetrievalManager:
    def __init__(self, vector_store, llm, ):
        self.vector_store = vector_store
        self.llm = llm
        self.intent_retiver = IntenBaseRetriver(vector_store=vector_store)
        self.prompt_builders = {
            UserIntent.SEARCH_DOCUMENTS: SearchDocumentsPromptBuilder(),
            UserIntent.GET_DOCUMENT_NAMES: GetDocumentNamesBulder(),
            UserIntent.SUMMARIZE_DOCUMENT: SummarizeDocumentPromptBuilder(),
            UserIntent.CHAT_GENERAL: ChatGeneralPromptBuilder,
        }

    def format_docs_with_sources(self, docs: List[Document]) -> str:
        formatted_docs = []

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            formatted_docs.append(f'[Source: {source}]\n{doc.page_content}\n')

        return '\n'.join(formatted_docs)

    def build_chain(
            self,
            intent: UserIntent,
            chat_history: List[Tuple[str, str]],
            params: Dict[str, Any] = None
    ) -> RunnableSequence:
        params = params or {}
        retriver = self.intent_retiver.get_retriver(intent, params)
        prompt_builder = self.prompt_builders.get(intent, self.prompt_builders[UserIntent.CHAT_GENERAL])

        prompt = prompt_builder.build_prompt()

        formatted_history = prompt_builder.format_chat_history(chat_history)

        chain = (
            {
                'context': retriver | self.format_docs_with_sources,
                'input': RunnablePassthrough(),
                'chat_history': lambda _: formatted_history,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def run(
        self,
        intent: UserIntent,
        chat_history: List[Tuple[str, str]],
        message: str,
        params: Dict[str, Any] = None,
    ) -> str:
        chain = self.build_chain(intent, chat_history, params)
        return chain.invoke(message)


if __name__ == "__main__":

    from rag_chat_app.vector import ChromaVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = ChromaVectorStore(embedding_function=embedding_function, collection_name='test_vector_db')
    vectorstore.initialize()
    manager = RetrievalManager(vector_store=vectorstore, llm=llm)
    chat_history: List[Tuple[str, str]] = []

    # 🟡 Question 1: Search for information
    user_input_1 = "Which document do I need for a resident card?"
    response_1 = manager.run(UserIntent.SEARCH_DOCUMENTS, chat_history, user_input_1)
    print(f"\n[SEARCH_DOCUMENTS]\n{response_1}")
    chat_history.append((user_input_1, response_1))

    # 🟡 Question 2: Which files contain the information
    user_input_2 = "In which files do we have information about which document is needed for a resident card?"
    response_2 = manager.run(UserIntent.GET_DOCUMENT_NAMES, chat_history, user_input_2)
    print(f"\n[GET_DOCUMENT_NAMES]\n{response_2}")
    chat_history.append((user_input_2, response_2))

    # 🟡 Question 3: Summarization
    user_input_3 = "provide summary of the last file"
    response_3 = manager.run(
        UserIntent.SUMMARIZE_DOCUMENT,
        chat_history, user_input_3,
        params={"document_name": "Confirmation.pdf"}
    )
    print(f"\n[SUMMARIZE_DOCUMENT]\n{response_3}")
    chat_history.append((user_input_3, response_3))
