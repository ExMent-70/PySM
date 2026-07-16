"""Безопасная запись JSON-файлов редактора."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping


def atomic_write_bundle(writers: Mapping[Path, Callable[[Path], None]]) -> None:
    """Best-effort transactional replacement of several files.

    Every payload is prepared beside its target first. Existing targets are
    backed up before replacement and restored if one of the replacements
    fails. This prevents the editor from publishing JSON and embedding files
    from different save generations.
    """

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        for raw_target, writer in writers.items():
            target = Path(raw_target)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            os.close(fd)
            temp_path = Path(temp_name)
            staged[target] = temp_path
            writer(temp_path)

        for target in staged:
            if not target.exists():
                backups[target] = None
                continue
            fd, backup_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".rollback",
            )
            os.close(fd)
            backup_path = Path(backup_name)
            shutil.copy2(target, backup_path)
            backups[target] = backup_path

        for target, temp_path in staged.items():
            os.replace(temp_path, target)
            replaced.append(target)
    except Exception as exc:
        rollback_errors = []
        for target in reversed(replaced):
            backup = backups.get(target)
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                elif backup.exists():
                    os.replace(backup, target)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "Ошибка rollback после неудачного сохранения: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)


def json_writer(data: Any, *, ensure_ascii: bool = False) -> Callable[[Path], None]:
    """Return a bundle writer for a JSON payload."""

    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=ensure_ascii, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    return write
