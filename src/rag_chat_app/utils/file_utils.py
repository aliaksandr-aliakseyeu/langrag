import hashlib
from pathlib import Path


def get_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.

    Reads file in chunks to handle large files efficiently without
    loading the entire file into memory.

    Args:
        file_path: Path to the file to hash

    Returns:
        SHA256 hex digest of the file content
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()
