from datetime import datetime
from dataclasses import dataclass

from rag_chat_app.enums import VectorStatus


@dataclass
class DocumentMetadata:
    sourse_type: str
    source_path: str
    file_name: str
    file_extension: str
    file_size: int
    last_modified: datetime
    file_hash: str
    chunk_count: int = 0
    vector_status: VectorStatus = VectorStatus.default()
