import logging

from langchain_community.document_loaders import UnstructuredRTFLoader
from langchain_core.documents import Document
from typing import List

from rag_chat_app.document_sources.metadata import DocumentMetadata
from rag_chat_app.parsers.base import Parser


logger = logging.getLogger(__name__)


class RtfParser(Parser):
    """
    Parser for RTF documents using Unstructured.

    Extracts text and structure (titles, paragraphs, lists, tables) and converts
    them into LangChain Document objects with metadata.
    """

    supported_extensions = [".rtf"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        """
        Parse a RTF file and extract structured content.

        Args:
            metadata: Document metadata containing file path and info

        Returns:
            List of LangChain Document objects
        """
        documents = UnstructuredRTFLoader(str(metadata.source_path)).load()
        for doc in documents:
            doc.metadata.update(
                {
                    "file_name": metadata.file_name,
                    "file_extension": metadata.file_extension,
                    "source_path": metadata.source_path,
                }
            )
        return documents
