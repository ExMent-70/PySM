# fc_lib/fc_image_loader.py

from pathlib import Path
import logging
import cv2
import numpy as np
from typing import Optional

# --- Новые импорты ---
try:
    import rawpy
    from psd_tools import PSDImage
    from PIL import Image  # Pillow нужен для конвертации из psd-tools
except ImportError as e:
    print(
        f"Ошибка импорта библиотек для ImageLoader: {e}. Установите: rawpy, psd-tools, Pillow"
    )
    # Можно выйти, если критично
    # import sys
    # sys.exit(1)
# --- Конец новых импортов ---

from .fc_config import ConfigManager
from .fc_messages import get_message

logger = logging.getLogger(__name__)


class ImageLoader:
    """Класс для загрузки и предварительной обработки изображений (RAW, JPG, PSD)."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.folder_path = Path(self.config.get("paths", "folder_path"))
        # Используем output_path из конфига для базовой папки
        self.output_path = Path(self.config.get("paths", "output_path"))
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Папка для сохранения извлеченных/обработанных JPEG
        self.jpeg_output_path = self.output_path / "_JPG"
        self.should_save_jpeg = self.config.get("processing", "save_jpeg", False)

        if self.should_save_jpeg:
            try:
                self.jpeg_output_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Сохранение обработанных JPG включено в: {self.jpeg_output_path}"
                )
            except OSError as e:
                logger.error(f"Не удалось создать папку {self.jpeg_output_path}: {e}")
                self.should_save_jpeg = False  # Отключаем, если не можем создать папку

    def load_image(self, file_path: Path, file_type: str) -> Optional[np.ndarray]:
        """
        Извлекает изображение (RAW/JPG/PSD), масштабирует и возвращает массив NumPy (BGR).
        """
        img = None
        source_description = ""
        img_to_process = None
        try:
            if file_type == "raw":
                # ... (загрузка из RAW без изменений) ...
                source_description = "RAW (встроенное превью)"
                logger.debug(f"Загрузка {source_description} из: {file_path.name}")
                with rawpy.imread(str(file_path)) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = cv2.imdecode(
                                np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR
                            )
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            img = cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)
                        else:
                            logger.warning(
                                get_message(
                                    "WARNING_NO_JPEG_IN_RAW", filename=file_path.name
                                )
                            )
                    except rawpy.LibRawNoThumbnailError:
                        logger.warning(f"В RAW нет превью: {file_path.name}")
                    except rawpy.LibRawUnsupportedThumbnailError:
                        logger.warning(
                            f"Формат превью в RAW не поддерживается: {file_path.name}"
                        )
                    except Exception as thumb_err:
                        logger.error(f"Ошибка извлечения превью из RAW: {thumb_err}")

            elif file_type == "psd":
                source_description = "PSD (композитное изображение)"
                logger.debug(f"Загрузка {source_description} из: {file_path.name}")
                try:
                    # Открываем PSD
                    psd = PSDImage.open(str(file_path))
                    # --- Используем topil() для получения Pillow Image ---
                    pil_image = psd.topil()
                    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                    if pil_image:
                        # Конвертируем в RGB (на случай RGBA) и затем в BGR для OpenCV
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")
                            logger.debug("PSD-композит конвертирован в RGB")
                        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        logger.debug(
                            f"Композитное изображение PSD получено и конвертировано в OpenCV формат (shape: {img.shape})"
                        )
                    else:
                        logger.warning(
                            f"Не удалось получить композитное изображение из PSD: {file_path.name} (topil() вернул None)"
                        )
                except Exception as psd_err:
                    logger.error(
                        f"Ошибка загрузки или обработки PSD (psd-tools): {file_path.name} - {psd_err}",
                        exc_info=True,
                    )
                    img = None

            elif file_type == "jpg":
                # ... (загрузка JPG без изменений) ...
                source_description = "JPEG"
                logger.debug(f"Загрузка {source_description} из: {file_path.name}")
                with open(file_path, "rb") as f:
                    img_data = f.read()
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

            else:
                logger.error(
                    f"Неподдерживаемый тип файла для загрузки: {file_type} ({file_path.name})"
                )
                logger.warning(
                    f"load_image для {file_path.name} ЗАВЕРШЕН С НЕИЗВЕСТНЫМ ТИПОМ, возвращается None"
                )
                return None

            # Проверка загрузки
            if img is None:
                logger.error(
                    f"Не удалось декодировать/получить изображение ({source_description}) из файла: {file_path.name}"
                )
                logger.warning(
                    f"load_image для {file_path.name} ЗАВЕРШЕН С ОШИБКОЙ ДЕКОДИРОВАНИЯ, возвращается None"
                )
                return None

            # Масштабирование
            height, width = img.shape[:2]
            long_side = max(height, width)
            min_preview_size = self.config.get("processing", "min_preview_size", 2048)
            img_to_process = img
            if long_side > min_preview_size:
                scale = min_preview_size / long_side
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
                img_to_process = cv2.resize(
                    img, (new_width, new_height), interpolation=cv2.INTER_AREA
                )
                logger.debug(
                    get_message(
                        "DEBUG_IMAGE_RESIZED",
                        filename=file_path.name,
                        original_shape=(height, width),
                        new_shape=(new_height, new_width),
                    )
                )
            else:
                logger.debug(
                    get_message(
                        "DEBUG_IMAGE_PROCESSED",
                        filename=file_path.name,
                        face_count=0,
                        shape=(height, width),
                    )
                )

            # Сохранение JPEG (если нужно)
            if self.should_save_jpeg:
                output_jpeg = self.jpeg_output_path / f"{file_path.stem}.jpg"
                jpeg_quality = 100  # Качество JPEG (0-100)
                target_dpi = (300, 300)  # Желаемый DPI (X, Y)

                logger.debug(
                    f"Подготовка к сохранению JPEG для {file_path.name} в {output_jpeg}..."
                )

                # --- ИЗМЕНЕНИЕ: Сохраняем ОДИН РАЗ через Pillow ---
                try:
                    # 1. Конвертируем BGR NumPy массив в RGB Pillow Image
                    # Проверяем, есть ли что конвертировать
                    if img_to_process is None or img_to_process.size == 0:
                        raise ValueError("Изображение для сохранения пустое")

                    # Конвертация цвета BGR (OpenCV) -> RGB (Pillow)
                    pil_image_to_save = Image.fromarray(
                        cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
                    )

                    # 2. Сохраняем Pillow Image с нужными параметрами
                    pil_image_to_save.save(
                        output_jpeg,
                        format="JPEG",  # Указываем формат явно
                        quality=jpeg_quality,
                        dpi=target_dpi,
                        icc_profile=pil_image_to_save.info.get(
                            "icc_profile"
                        ),  # Сохраняем ICC профиль, если он был
                    )
                    pil_image_to_save.close()  # Закрываем объект изображения

                    logger.debug(
                        get_message(
                            "DEBUG_JPEG_SAVED",
                            filename=file_path.name,
                            output_file=output_jpeg,
                        )
                        + f" (Q={jpeg_quality}, DPI={target_dpi})"
                    )

                except Exception as save_err:
                    logger.error(
                        f"Ошибка сохранения JPEG для {file_path.name} через Pillow в {output_jpeg}: {save_err}",
                        exc_info=True,
                    )
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            return img_to_process

        except Exception as e:
            logger.error(
                get_message("ERROR_PROCESSING_IMAGE", file_path=file_path, exc=e),
                exc_info=True,
            )
            logger.warning(
                f"load_image для {file_path.name} ЗАВЕРШЕН С ИСКЛЮЧЕНИЕМ, возвращается None"
            )
            return None
