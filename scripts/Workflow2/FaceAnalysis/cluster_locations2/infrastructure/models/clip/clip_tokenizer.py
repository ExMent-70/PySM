import logging
from pathlib import Path

from .clip_light_tokenizer import ClipLightTokenizer

logger = logging.getLogger(__name__)


class ClipTokenizerWrapper:
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = Path(tokenizer_path)
        self.hf_model_name = "openai/clip-vit-large-patch14"
        self.tokenizer = self._load_or_download()

    def _has_light_tokenizer_files(self) -> bool:
        return (
            (self.tokenizer_path / "vocab.json").exists()
            and (self.tokenizer_path / "merges.txt").exists()
        )

    def _load_or_download(self) -> ClipLightTokenizer:
        if self._has_light_tokenizer_files():
            logger.info(f"<b>Загрузка CLIP tokenizer из кеша:</b> {self.tokenizer_path}")
            return ClipLightTokenizer(self.tokenizer_path)

        logger.info(f"<b>Скачивание CLIP tokenizer:</b> {self.hf_model_name}")

        try:
            from transformers import CLIPTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers is required to download CLIP tokenizer files"
            ) from e

        tokenizer = CLIPTokenizer.from_pretrained(self.hf_model_name)
        self.tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(self.tokenizer_path))

        logger.info(f"<b>CLIP tokenizer сохранён в:</b> {self.tokenizer_path}")
        return ClipLightTokenizer(self.tokenizer_path)

    def tokenize(self, texts: list[str]):
        return self.tokenizer.tokenize(texts)
