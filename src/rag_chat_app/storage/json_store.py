import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from rag_chat_app.document_sources.metadata import DocumentMetadata, DocumentSourceType
from rag_chat_app.enums import VectorStatus
from .metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class JsonMetadataStore(MetadataStore):
    """
    JSON-based implementation of document metadata storage.

    Stores metadata in a single JSON file as a dictionary:
    { "source_type:source_path": { ...metadata... } }
    """

    def __init__(self, json_path: str):
        """
        Initialize JSON metadata store.

        Args:
            json_path: Path to JSON file for storing metadata
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            self._write_json({})
            logger.info("Created new JSON metadata store at %s", self.json_path)

    def _read_json(self) -> dict:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(
                "JSON metadata file not found or corrupted, initializing empty store."
            )
            return {}

    def _write_json(self, data: dict) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _make_key(self, doc: DocumentMetadata) -> str:
        return f"{doc.source_type.value}:{doc.source_path}"

    def _to_dict(self, doc: DocumentMetadata) -> dict:
        return {
            "file_hash": doc.file_hash,
            "source_type": doc.source_type.value,
            "source_path": doc.source_path,
            "file_name": doc.file_name,
            "file_extension": doc.file_extension,
            "file_size": doc.file_size,
            "last_modified": doc.last_modified.isoformat(),
            "chunk_count": doc.chunk_count,
            "vector_status": doc.vector_status.value,
            "vector_error": getattr(doc, "vector_error", ""),
            "created_at": getattr(doc, "created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "is_deleted": getattr(doc, "is_deleted", 0),
        }

    def _from_dict(self, data: dict) -> DocumentMetadata:
        return DocumentMetadata(
            file_hash=data["file_hash"],
            source_type=DocumentSourceType(data["source_type"]),
            source_path=data["source_path"],
            file_name=data["file_name"],
            file_extension=data["file_extension"],
            file_size=data["file_size"],
            last_modified=datetime.fromisoformat(data["last_modified"]),
            chunk_count=data["chunk_count"],
            vector_status=VectorStatus.from_string(data["vector_status"]),
        )

    def save_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        data = self._read_json()
        for doc in documents:
            key = self._make_key(doc)
            data[key] = self._to_dict(doc)
            logger.info("Saved metadata for document %s", key)
        self._write_json(data)

    def load_documents_metadata(
        self,
        vector_status: str = None,
        supported_extensions: List[str] = None,
        source_type: str = None,
    ) -> List[DocumentMetadata]:
        data = self._read_json()
        results = []
        for key, item in data.items():
            if item.get("is_deleted", 0):
                continue
            if vector_status and item["vector_status"] != vector_status:
                continue
            if source_type and item["source_type"] != source_type:
                continue
            if (
                supported_extensions
                and item["file_extension"] not in supported_extensions
            ):
                continue
            results.append(self._from_dict(item))
        logger.info("Loaded %d documents from JSON metadata store", len(results))
        return results

    def get_by_hash(self, file_hash: str) -> Optional[DocumentMetadata]: ...

    def update_document_processing_status(
        self,
        document: DocumentMetadata,
        vector_status: VectorStatus,
        vector_error: str = "",
        chunk_count: Optional[int] = None,
    ) -> None:
        data = self._read_json()
        key = self._make_key(document)
        if key in data:
            item = data[key]
            item["vector_status"] = vector_status.value
            item["vector_error"] = vector_error
            if chunk_count is not None:
                item["chunk_count"] = chunk_count
            item["updated_at"] = datetime.now().isoformat()
            self._write_json(data)
            logger.info(
                "Updated processing status for %s → %s", key, vector_status.value
            )
        else:
            logger.warning("Tried to update status for missing document %s", key)

    def delete_documents_metadata(self, documents: List[DocumentMetadata]) -> None:
        data = self._read_json()
        for doc in documents:
            key = self._make_key(doc)
            if key in data:
                data[key]["is_deleted"] = 1
                data[key]["updated_at"] = datetime.now().isoformat()
                logger.info("Marked document %s as deleted", key)
        self._write_json(data)
