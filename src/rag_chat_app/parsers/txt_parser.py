import logging

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List

from rag_chat_app.document_sources.metadata import DocumentMetadata
from rag_chat_app.parsers.base import Parser


logger = logging.getLogger(__name__)


class TxtParser(Parser):
    """
    Parser for TXT documents using TextLoader.

    Extracts text and converts
    them into LangChain Document objects with metadata.
    """

    supported_extensions = [".txt", ".log"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        """
        Parse a TXT file and extract text content.

        Args:
            metadata: Document metadata containing file path and info

        Returns:
            List of LangChain Document objects
        """
        documents = TextLoader(str(metadata.source_path), encoding="utf-8").load()
        for doc in documents:
            doc.metadata.update(
                {
                    "file_name": metadata.file_name,
                    "file_extension": metadata.file_extension,
                    "source_path": metadata.source_path,
                }
            )
        return documents
