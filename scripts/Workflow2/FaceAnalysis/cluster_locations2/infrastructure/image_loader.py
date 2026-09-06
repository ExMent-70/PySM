from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _strip_png_iccp_chunks(data: bytes) -> bytes:
    """Remove embedded ICC metadata from a PNG copy without touching pixels."""
    if not data.startswith(PNG_SIGNATURE):
        return data

    cursor = len(PNG_SIGNATURE)
    removed_ranges: list[tuple[int, int]] = []

    while cursor + 12 <= len(data):
        chunk_length = int.from_bytes(data[cursor : cursor + 4], "big")
        chunk_end = cursor + 12 + chunk_length
        if chunk_end > len(data):
            return data

        chunk_type = data[cursor + 4 : cursor + 8]
        if chunk_type == b"iCCP":
            removed_ranges.append((cursor, chunk_end))

        cursor = chunk_end
        # The PNG specification requires iCCP to precede the first IDAT chunk.
        if chunk_type in {b"IDAT", b"IEND"}:
            break

    if not removed_ranges:
        return data

    parts: list[bytes] = []
    previous_end = 0
    for chunk_start, chunk_end in removed_ranges:
        parts.append(data[previous_end:chunk_start])
        previous_end = chunk_end
    parts.append(data[previous_end:])
    return b"".join(parts)


class ImageLoader:
    def load(self, path: Path) -> Optional[np.ndarray]:
        try:
            data = path.read_bytes()
            if path.suffix.lower() == ".png":
                # OpenCV does not use the embedded profile for this BGR input,
                # while some legacy iCCP profiles trigger a libpng warning.
                data = _strip_png_iccp_chunks(data)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
