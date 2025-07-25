from datetime import datetime
from typing import Dict, List
from rag_chat_app.document_sources import DocumentMetadata
from rag_chat_app.storage import MetadataStore


def clasificator(
    metadata_db: MetadataStore,
    documents: List[DocumentMetadata],
    supported_extentions: List[str],
    source_type: str = None,
) -> Dict[str, List[DocumentMetadata]]:

    exist_documents = metadata_db.load_documents_metadata(
        source_type=source_type,
        supported_extentions=supported_extentions
    )

    result = {
        'unchaned': [],
        'new': [],
        'updated': [],
        'deleted': []
    }

    source_map = {doc.source_path: doc for doc in documents}
    db_map = {doc.source_path: doc for doc in exist_documents}

    source_path = set(source_map.keys())
    db_path = set(db_map.keys())

    new_path = source_path - db_path
    result['new'] = [source_map[path] for path in new_path]

    deleted_path = db_path - source_path
    result['deleted'] = [db_map[path] for path in deleted_path]

    for path in source_path & db_path:
        source_doc = source_map[path]
        db_doc = db_map[path]

        if source_doc.last_modified > datetime.fromisoformat(db_doc.last_modified):
            result['updated'].append(source_doc)
        else:
            result['unchaned'].append(source_doc)

    return result
