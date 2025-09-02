from datetime import datetime
from typing import Dict, List
from rag_chat_app.document_sources.metadata import DocumentMetadata
from rag_chat_app.storage.metadata_store import MetadataStore


def classifier(
    metadata_db: MetadataStore,
    documents: List[DocumentMetadata],
    supported_extensions: List[str],
    source_type: str = None,
) -> Dict[str, List[DocumentMetadata]]:
    """
    Classify documents by comparing filesystem state with database records.

    Compares current document list from filesystem with stored metadata
    to determine which documents are new, updated, unchanged, or deleted.

    Args:
        metadata_db: Metadata store to query existing documents
        documents: Current list of documents from filesystem
        supported_extensions: List of supported file extensions to filter
        source_type: Optional source type filter

    Returns:
        Dictionary with keys 'new', 'updated', 'unchanged', 'deleted' containing
        lists of DocumentMetadata objects for each category
    """
    exist_documents = metadata_db.load_documents_metadata(
        source_type=source_type, supported_extensions=supported_extensions
    )

    result = {"unchanged": [], "new": [], "updated": [], "deleted": []}

    source_map = {doc.source_path: doc for doc in documents}
    db_map = {doc.source_path: doc for doc in exist_documents}

    source_path = set(source_map.keys())
    db_path = set(db_map.keys())

    new_path = source_path - db_path
    result["new"] = [source_map[path] for path in new_path]

    deleted_path = db_path - source_path
    result["deleted"] = [db_map[path] for path in deleted_path]

    for path in source_path & db_path:
        source_doc = source_map[path]
        db_doc = db_map[path]

        if source_doc.last_modified > datetime.fromisoformat(db_doc.last_modified):
            result["updated"].append(source_doc)
        else:
            result["unchanged"].append(source_doc)

    return result
