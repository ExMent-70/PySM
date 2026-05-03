from pathlib import Path
from transformers import CLIPTokenizer
import logging

logger = logging.getLogger(__name__)


class ClipTokenizerWrapper:
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = Path(tokenizer_path)
        self.hf_model_name = "openai/clip-vit-large-patch14"

        self.tokenizer = self._load_or_download()

    def _load_or_download(self) -> CLIPTokenizer:
        if self.tokenizer_path.exists() and any(self.tokenizer_path.iterdir()):
            logger.info(f"<b>Загрузка токенизатора из кеша:</b> {self.tokenizer_path}")
            return CLIPTokenizer.from_pretrained(str(self.tokenizer_path))

        logger.info(f"<b>Скачивание токенизатора:</b> {self.hf_model_name}")

        tokenizer = CLIPTokenizer.from_pretrained(self.hf_model_name)

        self.tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(self.tokenizer_path))

        logger.info(f"<b>Tokenizer сохранён в:</b> {self.tokenizer_path} ✓")

        return tokenizer

    def tokenize(self, texts: list[str]):
        out = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        )

        return out["input_ids"], out["attention_mask"]