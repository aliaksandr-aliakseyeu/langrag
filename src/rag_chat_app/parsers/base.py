from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Optional

from rag_chat_app.document_sources import DocumentMetadata


class Parser(ABC):
    supported_extensions: List[str] = []

    @abstractmethod
    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        pass

    def is_applieble(self, metadata: DocumentMetadata) -> bool:
        return metadata.file_extension.lower() in [ext.lower() for ext in self.supported_extensions]

    def get_supported_extensions(self) -> List[str]:
        return self.supported_extensions


class ParserProvider:
    def __init__(self, parsers: List[Parser]):
        self._parsers = parsers

    def get_parser(self, metadata: DocumentMetadata) -> Optional[Parser]:
        for parser in self._parsers:
            if parser.is_applieble(metadata):
                return parser
        return None

    def get_suported_extentions(self) -> List[str]:
        extentions = set()
        for parser in self._parsers:
            extentions.update(parser.get_supported_extensions())
        return sorted(list(extentions))
