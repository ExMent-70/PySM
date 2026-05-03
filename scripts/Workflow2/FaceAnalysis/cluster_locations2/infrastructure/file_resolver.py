import logging
from pathlib import Path
from typing import List

from ..domain.models import ResolvedImage

logger = logging.getLogger(__name__)


class FileResolver:
    def __init__(self, mask_suffix: str):
        self.mask_suffix = mask_suffix

    def resolve(self, paths: List[Path], input_is_mask: bool) -> List[ResolvedImage]:
        resolved = []

        for p in paths:
            if not input_is_mask:
                resolved.append(ResolvedImage(p, p))
                continue

            name = p.name
            if self.mask_suffix not in name:
                logger.warning(f"Invalid mask filename: {name}")
                continue

            base = name.replace(self.mask_suffix, "")
            parent = p.parent.parent

            candidates = [
                parent / f"{base}.jpg",
                parent / f"{base}.jpeg",
            ]

            original = next((c for c in candidates if c.exists()), None)

            if not original:
                logger.warning(f"Original not found for mask: {name}")
                continue

            resolved.append(ResolvedImage(p, original))

        return resolved