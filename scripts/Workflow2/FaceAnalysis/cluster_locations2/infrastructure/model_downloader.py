import requests
from pathlib import Path
import time
import logging

from pysm_lib.pysm_progress_reporter import tqdm

logger = logging.getLogger(__name__)


MODEL_URLS = {
    "ViT-L-14.onnx": "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/Rqfo48sIngLSFg"
}


class ModelDownloader:
    def ensure(self, path: Path):
        if path.exists():
            logger.info("<i>модель найдена локально</i><br>")
            return

        url = MODEL_URLS.get(path.name)
        if not url:
            raise RuntimeError(f"Нет URL для модели: {path.name}")

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"<i>скачивание модели:</i> {path.name}<br>")

        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()

                    total = int(response.headers.get("content-length", 0))

                    with open(path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc="Загрузка модели"
                    ) as pbar:

                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                logger.info("<i>Модель успешно скачана</i>")
                return

            except Exception as e:
                logger.warning(f"Ошибка скачивания (попытка {attempt + 1}): {e}")

                if path.exists():
                    path.unlink()

                time.sleep(2)

        raise RuntimeError("Не удалось скачать модель после 3 попыток")