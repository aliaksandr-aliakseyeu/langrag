from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from typing import List

from rag_chat_app.document_sources import DocumentMetadata
from rag_chat_app.parsers.base import Parser


class PdfParser(Parser):

    supported_extensions = [".pdf"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
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
            print(f"Failed to parse PDF {metadata.source_path}: {e}")
            raise
