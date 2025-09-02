import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

from .base import DocumentSource
from .metadata import DocumentMetadata, DocumentSourceType
from rag_chat_app.utils.file_utils import get_file_hash

logger = logging.getLogger(__name__)


class LocalfileSource(DocumentSource):
    """
    Document source for local file system.

    This class implements the DocumentSource interface to provide documents
    from a local directory. It supports various file types and can recursively
    scan directories for supported documents.

    Attributes:
        storage_path: Path to the directory containing documents
        supported_extensions: List of supported file extensions
    """

    def __init__(self, storage_path: str, supported_extensions: List[str] = None):
        """
        Initialize the local file source.

        Args:
            storage_path: Path to the directory containing documents
            supported_extensions: List of supported file extensions.
                                 If None, uses default list including common formats.
        """
        if not storage_path:
            raise ValueError("storage_path cannot be empty")

        self.storage_path = Path(storage_path).resolve()
        if not self.storage_path.exists():
            raise FileNotFoundError(f"Storage path does not exist: {self.storage_path}")

        if not os.access(self.storage_path, os.R_OK):
            raise PermissionError(f"No read permission for: {self.storage_path}")

        self.supported_extensions = supported_extensions or [
            ".pdf",
        ]

    def list_documents(self) -> List[DocumentMetadata]:
        """
        Scan the storage directory and return metadata for all supported documents.

        Returns:
            List of DocumentMetadata objects for all discovered documents.
            Returns empty list if storage path doesn't exist.
        """
        documents = []
        total_files = 0
        error_count = 0

        for file_path in self._get_files():
            try:
                total_files += 1
                file_stat = file_path.stat()
                file_hash = get_file_hash(file_path)
                metadata = DocumentMetadata(
                    source_type=DocumentSourceType.LOCAL_FILE,
                    source_path=str(file_path),
                    file_name=file_path.name,
                    file_extension=file_path.suffix.lower(),
                    file_size=file_stat.st_size,
                    last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                    file_hash=file_hash,
                )
                documents.append(metadata)
            except (OSError, IOError) as e:
                error_count += 1
                logger.warning(f"METADATA_SCAN: Error reading file {file_path}: {e}")
                continue
            except Exception as e:
                error_count += 1
                logger.error(
                    f"METADATA_SCAN: Unexpected error occurred while processing {file_path}: {e}"
                )
                continue

        logger.info(
            f"METADATA_SCAN: Completed - found {len(documents)}/{total_files} valid documents, {error_count} errors"
        )

        return documents

    def read_document(self, meta_data: DocumentMetadata) -> str:
        """
        Read the content of a document.

        Args:
            meta_data: DocumentMetadata object containing document information

        Returns:
            Raw text content of the document
        """
        ...

    def _get_files(self, recursive: bool = True) -> Iterator[Path]:
        """
        Get iterator over supported files in the storage directory.

        Args:
            recursive: If True, search subdirectories recursively

        Yields:
            Path objects for files matching supported extensions
        """
        files = (
            self.storage_path.rglob("*") if recursive else self.storage_path.glob("*")
        )

        for file_path in files:
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                yield file_path
