from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Optional

from rag_chat_app.document_sources.metadata import DocumentMetadata


class Parser(ABC):
    """
    Abstract base class for document parsers.

    Each parser implementation should handle specific file types and convert
    them into LangChain Document objects for further processing.
    """

    supported_extensions: List[str] = []

    @abstractmethod
    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        """
        Parse a document and return LangChain Document objects.

        Args:
            metadata: Document metadata containing file path and info

        Returns:
            List of LangChain Document objects with content and metadata

        """
        pass

    def is_applieble(self, metadata: DocumentMetadata) -> bool:
        """
        Check if this parser can handle the given document type.

        Args:
            metadata: Document metadata to check

        Returns:
            True if parser supports the file extension
        """
        return metadata.file_extension.lower() in [
            ext.lower() for ext in self.supported_extensions
        ]

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this parser.

        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx'])
        """
        return self.supported_extensions


class ParserProvider:
    """
    Manages multiple parsers and selects the appropriate one for each document.

    This class acts as a registry and factory for document parsers, automatically
    selecting the correct parser based on file extension.
    """

    def __init__(self, parsers: List[Parser]):
        """
        Initialize parser provider with a list of parsers.

        Args:
            parsers: List of parser instances to manage
        """
        self._parsers = parsers

    def get_parser(self, metadata: DocumentMetadata) -> Optional[Parser]:
        """
        Get the appropriate parser for a document.

        Args:
            metadata: Document metadata to find parser for

        Returns:
            Parser instance that can handle the file type, or None if no parser found
        """
        for parser in self._parsers:
            if parser.is_applieble(metadata):
                return parser
        return None

    def get_suported_extentions(self) -> List[str]:
        """
        Get all file extensions supported by registered parsers.

        Returns:
            Sorted list of all supported file extensions
        """
        extentions = set()
        for parser in self._parsers:
            extentions.update(parser.get_supported_extensions())
        return sorted(list(extentions))
