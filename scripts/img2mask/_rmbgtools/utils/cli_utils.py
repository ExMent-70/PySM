# _rmbgtools/utils/cli_utils.py

# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
import os
from typing import List
from .. import logger

# ======================================================================================
# Блок 2: Функция expand_paths
# ======================================================================================
def expand_paths(input_paths: List[str]) -> List[str]:
    """
    Раскрывает директории в списки файлов изображений.
    """
    expanded_paths = []
    seen = set()
    supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    for path in input_paths:
        path = os.path.normpath(path)
        if path in seen:
            continue

        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                if path not in seen:
                    expanded_paths.append(path)
                    seen.add(path)
            else:
                logger.warning(f"Skipping non-image file: {os.path.basename(path)}")
        elif os.path.isdir(path):
            logger.info(f"Scanning directory: {path}")
            for item in sorted(os.listdir(path)):
                full_path = os.path.join(path, item)
                if full_path in seen:
                    continue
                if os.path.isfile(full_path) and os.path.splitext(item)[1].lower() in supported_extensions:
                    expanded_paths.append(full_path)
                    seen.add(full_path)
        else:
            logger.warning(f"Path not found, skipping: {path}")
        
        seen.add(path)
            
    return expanded_paths