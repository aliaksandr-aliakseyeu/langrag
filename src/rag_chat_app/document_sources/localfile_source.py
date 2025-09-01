from datetime import datetime
from pathlib import Path
from typing import Iterator, List

from .base import DocumentSource
from .metadata import DocumentMetadata
from rag_chat_app.utils.file_utils import get_file_hash


class LocalfileSource(DocumentSource):
    def __init__(self, storage_path: str, supported_extensions: List[str] = None):
        self.storage_path = Path(storage_path)
        self.supported_extensions = supported_extensions or [
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".doc",
            ".html",
            ".htm",
            ".csv",
            ".xlsx",
            ".rtf",
        ]

    def list_documents(self) -> List[DocumentMetadata]:
        documents = []

        if not self.storage_path.exists():
            print("Storadge path does not exist")
            return documents

        for file_path in self._get_files():
            try:
                file_stat = file_path.stat()
                file_hash = get_file_hash(file_path)
                metadata = DocumentMetadata(
                    sourse_type="local_file",
                    source_path=str(file_path),
                    file_name=file_path.name,
                    file_extension=file_path.suffix.lower(),
                    file_size=file_stat.st_size,
                    last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                    file_hash=file_hash,
                )
                documents.append(metadata)
            except (OSError, IOError) as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            except Exception as e:
                print(
                    f"Oops! Unexpected error occurred while processing {file_path}: {e}"
                )
                continue

        return documents

    def read_document(self, meta_data: DocumentMetadata) -> str: ...

    def watch_for_changes(self): ...

    def _get_files(self, recursive: bool = True) -> Iterator[Path]:
        files = (
            self.storage_path.rglob("*") if recursive else self.storage_path.glob("*")
        )

        for file_path in files:
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                yield file_path
