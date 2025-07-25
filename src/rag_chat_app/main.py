from pathlib import Path
from pprint import pprint
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from rag_chat_app.config import settings
from rag_chat_app.vector import ChromaVectorStore
from rag_chat_app.document_sources import LocalfileSource
from rag_chat_app.storage import SQLiteMetadataStore
from rag_chat_app.storage import run_migartions
from rag_chat_app.storage import VectorStatus
from rag_chat_app.parsers import PdfParser, ParserProvider
from rag_chat_app.vector.chunker import LangChainChunker
from rag_chat_app.utils.files_clasificator import clasificator
from rag_chat_app.utils.utils import format_chat_history


load_dotenv()


def main():
    parsers = [PdfParser(), ]
    parser_provider = ParserProvider(parsers)

    supported_extentions = parser_provider.get_suported_extentions()
    print('------------------------------------------')
    print(f'Supported_extention: {supported_extentions}')
    print('------------------------------------------')
    run_migartions(settings.DB_PATH)
    meta_store = SQLiteMetadataStore(settings.DB_PATH)
    source = LocalfileSource(settings.DOCUMENT_FOLDER, supported_extensions=supported_extentions)
    documents = source.list_documents()
    print(f"Found {len(documents)} document(s).")
    print('------------------------------------------')
    documents_status_map = clasificator(
        metadata_db=meta_store, documents=documents, supported_extentions=supported_extentions
    )
    print('------------------------------------------')
    pprint(documents_status_map)
    meta_store.save_documents_metadata(documents_status_map['new']+documents_status_map['updated'])
    print('------------------------------------------')
    print("Metadata saved to SQLite.")
    meta_store.delete_documents_metadata(documents_status_map['deleted'])
    print('------------------------------------------')
    print("Metadata deleted from SQLite.")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    test_chroma_db = ChromaVectorStore(embedding_function=embedding_function, collection_name='test_vector_db')
    test_chroma_db.initialize()
    pending_documents = meta_store.load_documents_metadata(vector_status=VectorStatus.PENDING)
    print('------------------------------------------')
    for doc_metadata in pending_documents:
        try:
            meta_store.update_document_processing_status(document=doc_metadata, vector_status=VectorStatus.PROCESSING)

            if not Path(doc_metadata.source_path).exists():
                vector_error = f'File does not exist: {doc_metadata.source_path}'
                meta_store.update_document_processing_status(
                    document=doc_metadata,
                    vector_status=VectorStatus.FAILED,
                    vector_error=vector_error
                )
                continue

            if test_chroma_db.document_exists(source_path=doc_metadata.source_path):
                test_chroma_db.delete_vectors_by_source(source_path=doc_metadata.source_path)
                print('------------------------------------------')
                print(f'Deleted vectors for document: {doc_metadata.source_path}')

            parser = parser_provider.get_parser(doc_metadata)
            if parser:
                parsed = parser.parse(doc_metadata)
                chunker = LangChainChunker()
                chunk_documents = chunker.chunk_documents(parsed)
                if not len(chunk_documents):
                    vector_error = f'Document {doc_metadata.source_path} do not have any text'
                    print(vector_error)
                    meta_store.update_document_processing_status(
                        document=doc_metadata,
                        vector_status=VectorStatus.FAILED,
                        vector_error=vector_error
                    )
                    continue
                test_chroma_db.add_documents(chunk_documents)
                meta_store.update_document_processing_status(
                    document=doc_metadata, vector_status=VectorStatus.COMPLETED, chunk_count=len(chunk_documents)
                )
            else:
                print(f'Parser for {doc_metadata.file_name} is not provided')

        except Exception as e:
            vector_error = f'Failed to process {doc_metadata.file_name}: {e}'
            meta_store.update_document_processing_status(
                document=doc_metadata,
                vector_status=VectorStatus.FAILED,
                vector_error=vector_error
            )
            print(vector_error)

    print('retriver part')
    retriver = test_chroma_db.as_retriver()
    custom_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following pieces of context to answer the user's question.
    If the answer cannot be found in the context, say you don't know. DO NOT use prior knowledge.

    {context}

    Chat history:
    {chat_history}

    Question: {input}
    Answer:
    """)

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )
    chat_history: List[Tuple[str, str]] = []
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=custom_prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriver, combine_docs_chain=combine_docs_chain)

    while True:
        question = input('You: ')
        if question.lower() in ['exit', 'quit']:
            break

        inputs = {
            'input': question,
            'chat_history': format_chat_history(chat_history)
        }

        result = retrieval_chain.invoke(input=inputs)
        answer = result['answer']
        print(answer)

        chat_history.append((question, answer))


if __name__ == '__main__':
    main()
