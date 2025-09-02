import logging

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from typing import List

from rag_chat_app.document_sources.metadata import DocumentMetadata
from rag_chat_app.parsers.base import Parser


logger = logging.getLogger(__name__)


class PdfParser(Parser):
    """
    Parser for PDF documents using PDFPlumber.

    Extracts text content from PDF files and creates LangChain Document objects
    with preserved metadata for further processing in the RAG pipeline.
    """

    supported_extensions = [".pdf"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        """
        Parse a PDF file and extract text content.

        Args:
            metadata: Document metadata containing file path and information

        Returns:
            List of LangChain Document objects, typically one per page
        """
        try:
            documents = PDFPlumberLoader(str(metadata.source_path)).load()
            for doc in documents:
                doc.metadata.update(
                    {
                        "file_name": metadata.file_name,
                        "file_extension": metadata.file_extension,
                        "source_path": metadata.source_path,
                    }
                )
            return documents
        except Exception as e:
            logger.error(f"Failed to parse PDF {metadata.source_path}: {e}")
            raise
