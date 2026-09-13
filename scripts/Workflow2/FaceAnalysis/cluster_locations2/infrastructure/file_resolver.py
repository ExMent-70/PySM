import logging
from pathlib import Path
from typing import List

from ..domain.models import ResolvedImage

logger = logging.getLogger(__name__)


class FileResolver:
    def __init__(self, mask_suffix: str):
        self.mask_suffix = mask_suffix
        self.original_exts = (".jpg", ".jpeg", ".png")

    def resolve(self, paths: List[Path], input_is_mask: bool) -> List[ResolvedImage]:
        """Map analysis input files to the original photos used for JSON updates."""
        resolved = []

        for p in paths:
            if not input_is_mask:
                resolved.append(ResolvedImage(p, p))
                continue

            mask_name = self._strip_mask_extension_wrapper(p)
            if not mask_name.endswith(self.mask_suffix):
                logger.warning(f"Invalid mask filename: {p.name}")
                continue

            base = mask_name[: -len(self.mask_suffix)]
            analysis_dir = p.parent.parent.parent
            candidates = [
                analysis_dir / "JPG" / f"{base}{ext}"
                for ext in self.original_exts
            ]

            original = next((c for c in candidates if c.exists()), None)

            if not original:
                logger.warning(f"Original not found for mask: {p.name}")
                continue

            resolved.append(ResolvedImage(p, original))

        return resolved

    def _strip_mask_extension_wrapper(self, path: Path) -> str:
        """Return a mask name that can be checked against mask_suffix.

        Cutout masks can be saved as files like
        ``IMG_0001_BiRefNet-portrait_output.jpg.png``. In that case the final
        ``.png`` is only a wrapper extension and the configured suffix still
        includes ``.jpg``.
        """
        if path.stem.endswith(self.mask_suffix):
            return path.stem
        return path.name
