"""File version checks for separately opened workflow stages."""
from pathlib import Path
SelectionSignature = tuple[int, int, int] | None

def file_signature(path: Path) -> SelectionSignature:
    """Return the current file version, or None before its first save."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
