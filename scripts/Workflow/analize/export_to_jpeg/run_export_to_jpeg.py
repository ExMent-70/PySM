# analize/export_to_jpeg/run_export_to_jpeg.py

# --- Блок 1: Импорты стандартных и сторонних библиотек ---
# ==============================================================================
import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Попытка импорта необходимых библиотек для обработки изображений
try:
    import cv2
    import numpy as np
    import rawpy
    from PIL import Image
    from psd_tools import PSDImage
except ImportError as e:
    print(f"Критическая ошибка: Необходимая библиотека не найдена. {e}", file=sys.stderr)
    print("Пожалуйста, установите зависимости: pip install opencv-python numpy rawpy Pillow psd-tools", file=sys.stderr)
    sys.exit(1)


# --- Блок 2: Настройка системного пути и импорт PySM ---
# ==============================================================================
try:
    current_script_path = Path(__file__).resolve()
    # Предполагаем, что структура папок .../analize/export_to_jpeg/
    project_root = current_script_path.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else None

# --- Блок 3: Константы и настройка логирования ---
# ==============================================================================
RAW_EXTENSIONS = [".arw", ".cr2", ".cr3", ".nef", ".dng"]
PSD_EXTENSIONS = [".psd", ".psb"]

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- Блок 4: Вспомогательные функции ---
# ==============================================================================
def construct_analysis_paths() -> Dict[str, Optional[Path]]:
    """
    Формирует пути для анализа на основе переменных контекста PySM.
    """
    if not IS_MANAGED_RUN or not pysm_context:
        logger.critical("Ошибка: Скрипт запущен без окружения PySM, автоматическое формирование путей невозможно.")
        return {"input": None, "output": None}

    photo_session = pysm_context.get("wf_photo_session")
    session_name = pysm_context.get("wf_session_name")
    session_path_str = pysm_context.get("wf_session_path")

    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста (wf_session_path, wf_session_name, wf_photo_session) не найдены.")
        return {"input": None, "output": None}

    base_path = Path(session_path_str) / session_name
    input_dir = base_path / "Capture" / photo_session
    output_dir = base_path / "Output" / f"Analysis_{photo_session}" / "JPG"

    return {"input": input_dir, "output": output_dir}


# --- Блок 5: Класс ImageExporter ---
# ==============================================================================
class ImageExporter:
    """
    Инкапсулирует логику конвертации и сохранения изображений в формат JPEG.
    """
    def __init__(self, jpeg_quality: int, preview_size: int):
        if not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality должно быть в диапазоне от 0 до 100.")
        self.jpeg_quality = jpeg_quality
        self.preview_size = preview_size
        logger.info(f"Качество JPEG=<b>{self.jpeg_quality}</b>, Размер=<b>{self.preview_size}</b>px.")

    def export_file(self, source_path: Path, output_dir: Path) -> Tuple[bool, str]:
        output_jpeg_path = output_dir / f"{source_path.stem}.jpg"
        img_np = None
        source_type = "unknown"
        try:
            suffix = source_path.suffix.lower()
            if suffix in RAW_EXTENSIONS:
                source_type = "RAW Preview"
                try:
                    with rawpy.imread(str(source_path)) as raw:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img_np = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            img_np = cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)
                        else:
                            return False, f"Неподдерживаемый формат превью в {source_path.name}."
                except rawpy.LibRawNoThumbnailError:
                    return False, f"В RAW-файле {source_path.name} отсутствует превью."
                except rawpy.LibRawUnsupportedThumbnailError:
                    return False, f"Формат превью в {source_path.name} не поддерживается rawpy."
                except Exception as raw_exc:
                    return False, f"Ошибка извлечения превью из {source_path.name}: {raw_exc}"
            elif suffix in PSD_EXTENSIONS:
                source_type = "PSD"
                psd = PSDImage.open(str(source_path))
                pil_image = psd.topil()
                if pil_image:
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if img_np is None:
                return False, f"Не удалось извлечь изображение из {source_path.name} ({source_type})."

            height, width = img_np.shape[:2]
            long_side = max(height, width)
            if long_side > self.preview_size:
                scale = self.preview_size / long_side
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
                img_to_save = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                img_to_save = img_np

            pil_image_rgb = Image.fromarray(cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB))
            pil_image_rgb.save(
                output_jpeg_path,
                format="JPEG",
                quality=self.jpeg_quality,
                dpi=(300, 300),
                icc_profile=pil_image_rgb.info.get("icc_profile"),
            )
            return True, f"{source_path.name} -> {output_jpeg_path.name}"
        except Exception as e:
            return False, f"Ошибка при обработке {source_path.name}: {e}"


# --- Блок 6: Конфигурация и выполнение скрипта ---
# ==============================================================================
def get_config() -> argparse.Namespace:
    """
    Определяет аргументы командной строки и разрешает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(description="Конвертация RAW/PSD файлов в JPEG.")
    parser.add_argument(
        "--a_r2j_image_type", type=str, dest="a_r2j_image_type", default="raw",
        choices=["raw", "psd", "all"], help="Тип файлов для обработки."
    )
    parser.add_argument(
        "--a_r2j_preview_size", type=int, dest="a_r2j_preview_size", default=4096,
        help="Максимальный размер длинной стороны для итогового JPEG."
    )
    parser.add_argument(
        "--a_r2j_jpeg_quality", type=int, dest="a_r2j_jpeg_quality", default=95,
        help="Качество сохранения JPEG (0-100)."
    )
    parser.add_argument(
        "--all_threads", type=int, dest="all_threads", default=0,
        help="Количество потоков для обработки (0 - авто)."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        # При автономном запуске потребуется передать пути вручную (не реализовано в этой версии)
        return parser.parse_args()

def main():
    """
    Основная функция-оркестратор.
    """
    logger.info("<b>ЭКСПОРТ RAW/PSD В JPEG</b><br>")

    # 1. Получение конфигурации и путей
    config = get_config()
    paths = construct_analysis_paths()
    input_dir = paths.get("input")
    output_dir = paths.get("output")

    # 2. Валидация путей (критически важно после их формирования)
    if not input_dir or not output_dir:
        # Сообщение об ошибке уже было выведено в construct_analysis_paths
        sys.exit(1)

    if not input_dir.is_dir():
        logger.critical(f"Ошибка: Папка с исходниками не найдена или не является директорией: {input_dir}")
        sys.exit(1)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Ошибка: Не удалось создать выходную папку {output_dir}: {e}")
        sys.exit(1)

    logger.info(f"Исходная папка: <i>{input_dir.resolve()}</i>")
    logger.info(f"Выходная папка: <i>{output_dir.resolve()}</i><br>")

    # 3. Определение списка файлов для обработки
    allowed_extensions = []
    if config.a_r2j_image_type in ["raw", "all"]:
        allowed_extensions.extend(RAW_EXTENSIONS)
    if config.a_r2j_image_type in ["psd", "all"]:
        allowed_extensions.extend(PSD_EXTENSIONS)
    
    files_to_process = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in allowed_extensions
    ]

    if not files_to_process:
        logger.info("Не найдено файлов для обработки с указанными расширениями. Завершение работы.")
        return

    logger.info(f"Найдено <b>{len(files_to_process)} {config.a_r2j_image_type}-файла(ов)</b> для конвертации.")

    # 4. Запуск обработки
    exporter = ImageExporter(
        jpeg_quality=config.a_r2j_jpeg_quality,
        preview_size=config.a_r2j_preview_size
    )

    num_workers = config.all_threads
    if not num_workers or num_workers <= 0:
        num_workers = os.cpu_count() or 4
        logger.info(f"Используется <b>{num_workers}</b> потоков для обработки (авто).")
    else:
        logger.info(f"Используется <b>{num_workers}</b> потоков для обработки.")

    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(exporter.export_file, file_path, output_dir): file_path
            for file_path in files_to_process
        }

        progress_bar = tqdm(futures.items(), total=len(files_to_process), desc="Экспорт в JPEG")
        for future, path in progress_bar:
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    logger.warning(message)
                progress_bar.set_postfix(ok=success_count, failed=error_count)
            except Exception as e:
                error_count += 1
                logger.error(f"Критическая ошибка в потоке для файла {path.name}: {e}")
                progress_bar.set_postfix(ok=success_count, failed=error_count)

    if IS_MANAGED_RUN and pysm_context:
        pysm_context.log_link(url_or_path=str(output_dir), text="Открыть папку с файлами <i>JPG</i>")

    logger.info(f"<br>Экспорт завершен")
    logger.info(f"- успешно: <b>{success_count}</b>")
    logger.info(f"- ошибок: <b>{error_count}</b><br>")

    if error_count > 0:
        sys.exit(1)


# --- Блок 7: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    if not IS_MANAGED_RUN:
        logger.info("Скрипт запущен автономно. Автоматическое определение путей невозможно.")
        # Для автономного запуска потребуется доработка (например, передача путей через аргументы)
    main()