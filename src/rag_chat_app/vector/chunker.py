from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.schema import Document

from rag_chat_app.config import settings
from rag_chat_app.config import ChunkingConfig


class LangChainChunker:
    """
    Document chunker using LangChain text splitters.

    Provides configurable text chunking for preparing documents for vector
    embedding, with support for custom chunk sizes and overlap settings.
    """

    def __init__(
        self, config: ChunkingConfig = None, text_splitter: TextSplitter = None
    ):
        """
        Initialize document chunker.

        Args:
            config: Chunking configuration (uses default from settings if not provided)
            text_splitter: Custom text splitter (uses RecursiveCharacterTextSplitter if not provided)
        """
        self.config = config or settings.CHUNKING_CONFIG
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=self.config["max_chunk_size"],
            chunk_overlap=self.config["overlap_size"],
            separators=self.config["separators"],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for vector embedding.

        Args:
            documents: List of LangChain Document objects to chunk

        Returns:
            List of chunked Document objects ready for embedding
        """
        return self.text_splitter.split_documents(documents)
