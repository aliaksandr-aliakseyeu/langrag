from typing import List
from .base import DocumentSource
from .localfile_source import LocalfileSource


def create_localfile_source(
    document_folder: str, supported_extensions: List[str]
) -> DocumentSource:
    """
    Create a local file system document source.

    Args:
        document_folder: Path to the folder containing documents
        supported_extensions: List of supported file extensions

    Returns:
        LocalfileSource instance
    """
    return LocalfileSource(document_folder, supported_extensions)
