from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.schema import Document

from rag_chat_app.config import settings
from rag_chat_app.config import ChunkingConfig


class LangChainChunker:
    def __init__(self, config: ChunkingConfig = None, text_splitter: TextSplitter = None) -> Document:
        self.config = config or settings.CHUNKING_CONFIG
        self.text_spliter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=self.config['max_chunk_size'],
            chunk_overlap=self.config['overlap_size'],
            separators=self.config['separators'],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_spliter.split_documents(documents)
