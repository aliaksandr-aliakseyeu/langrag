from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from rag_chat_app.enums import VectorStatus


class DocumentSourceType(str, Enum):
    """Types of document sources supported by the system."""

    LOCAL_FILE = "local_file"
    ONE_DRIVE = "onedrive"
    GOOGLE_DRIVE = "google_drive"
    SHAREPOINT = "sharepoint"
    WEB_CRAWLER = "web_crawler"
    API_SOURCE = "api_source"
    FTP_SOURCE = "ftp"


@dataclass
class DocumentMetadata:
    """
    Metadata information for a document in the RAG system.

    This dataclass stores essential information about documents including
    file details, processing status, and vectorization state.

    Attributes:
        source_type: Type of document source (DocumentSourceType enum)
        source_path: Full path or URL to the source document
        file_name: Name of the file including extension
        file_extension: File extension (e.g., '.pdf', '.txt')
        file_size: Size of the file in bytes
        last_modified: Timestamp when the document was last modified
        file_hash: Hash of the file content for change detection
        chunk_count: Number of chunks this document was split into (default: 0)
        vector_status: Current vectorization status (default: VectorStatus.default())
    """

    source_type: DocumentSourceType
    source_path: str
    file_name: str
    file_extension: str
    file_size: int
    last_modified: datetime
    file_hash: str
    chunk_count: int = 0
    vector_status: VectorStatus = VectorStatus.default()
