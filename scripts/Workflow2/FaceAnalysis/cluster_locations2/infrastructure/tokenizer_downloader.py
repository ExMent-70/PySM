from pathlib import Path
from transformers import CLIPTokenizer
import logging

logger = logging.getLogger(__name__)


class TokenizerDownloader:

    def ensure(self, path: Path, model_name: str):
        if path.exists() and any(path.iterdir()):
            logger.info("<b>Tokenizer найден локально</b>")
            return

        logger.info("<b>Скачивание tokenizer...</b>")

        path.mkdir(parents=True, exist_ok=True)

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(path)

        logger.info("<b>Tokenizer скачан</b> ✓")