import sqlite3
from pathlib import Path

from ..enums import VectorStatus


def run_migartions(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    default_status = VectorStatus.default().value
    allowed_statuses = "', '".join(VectorStatus.choices())

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS
        documents(
            file_hash TEXT,
            sourse_type TEXT,
            source_path TEXT,
            file_name TEXT,
            file_extension TEXT,
            file_size INTEGER,
            last_modified TEXT,
            chunk_count INTEGER DEFAULT 0,
            vector_status TEXT DEAFULT '{default_status}'
                CHECK (vector_status IN ('{allowed_statuses}')),
            vector_error TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_deleted BOOLEAN DEFAULT 0,
            PRIMARY KEY (source_path, sourse_type)
        )
        """)
        conn.commit()
