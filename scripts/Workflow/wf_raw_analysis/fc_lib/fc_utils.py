# fc_lib/fc_utils.py
"""Утилитные функции для обработки лиц и файлов."""

import logging
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict  # Добавили Optional, Dict
import numpy as np
import cv2
import onnxruntime as ort
import json
from urllib.parse import quote

from .fc_messages import get_message

logger = logging.getLogger(__name__)


# В файле fc_lib/fc_utils.py
import logging
from typing import Union, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ... (остальные функции) ...


# --- НОВАЯ ФУНКЦИЯ: Загрузка эмбеддингов и индексов ---
def load_embeddings_and_indices(
    embeddings_dir: Path,
    data_type: str,
    calling_module_name: str = "Unknown",  # Для логирования
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Загружает массив эмбеддингов из .npy файла и индексный словарь из .json файла.

    Args:
        embeddings_dir: Путь к папке _Embeddings.
        data_type: 'portrait' или 'group'.
        calling_module_name: Имя модуля, вызывающего функцию (для логов).

    Returns:
        Кортеж: (массив_эмбеддингов | None, индексный_словарь | None).
                Для 'group' ключи индексного словаря будут преобразованы обратно в кортежи.
    """
    embeddings_array = None
    index_dict = None
    npy_filename = f"{data_type}_embeddings.npy"
    idx_filename = f"{data_type}_index.json"
    npy_path = embeddings_dir / npy_filename
    idx_path = embeddings_dir / idx_filename

    logger.debug(
        f"Загрузка эмбеддингов и индекса типа '{data_type}' из {embeddings_dir.resolve()}..."
    )

    # Загрузка эмбеддингов
    if npy_path.exists() and npy_path.is_file():
        try:
            embeddings_array = np.load(npy_path)
            logger.debug(
                f"Загружено {embeddings_array.shape[0]} эмбеддингов из {npy_filename}"
            )
        except Exception as e:
            logger.error(
                f"Ошибка загрузки файла эмбеддингов {npy_path.resolve()}: {e}",
                exc_info=True,
            )
            embeddings_array = None  # Сбрасываем в случае ошибки
    else:
        logger.warning(
            f"Файл эмбеддингов {npy_path.resolve()} не найден."
        )

    # Загрузка индекса
    if idx_path.exists() and idx_path.is_file():
        try:
            with idx_path.open("r", encoding="utf-8") as f:
                raw_index_dict = json.load(f)

            if data_type == "group":
                # Преобразуем строковые ключи обратно в кортежи (filename, face_idx)
                index_dict = {}
                for key_str, idx_val in raw_index_dict.items():
                    parts = key_str.split("::")
                    if len(parts) == 2:
                        try:
                            index_dict[(parts[0], int(parts[1]))] = int(
                                idx_val
                            )  # Убедимся что индекс тоже int
                        except ValueError:
                            logger.warning(
                                f"Некорректный ключ или значение '{key_str}': {idx_val} в файле индекса {idx_filename}. Пропуск."
                            )
                    else:
                        logger.warning(
                            f"Некорректный формат ключа '{key_str}' в файле индекса {idx_filename}. Пропуск."
                        )
            else:  # Для portrait ключи уже строки
                # Преобразуем значения индекса в int на всякий случай
                try:
                    index_dict = {k: int(v) for k, v in raw_index_dict.items()}
                except ValueError:
                    logger.error(
                        f"Некорректные значения индексов в файле {idx_filename}. Загрузка индекса не удалась."
                    )
                    index_dict = None

            if index_dict is not None:
                logger.debug(
                    f"Загружен индекс ({len(index_dict)} записей) из {idx_filename}"
                )

        except json.JSONDecodeError as e:
            logger.error(
                f"Ошибка декодирования JSON индекса {idx_path.resolve()}: {e}"
            )
            index_dict = None
        except Exception as e:
            logger.error(
                f"Ошибка загрузки файла индекса {idx_path.resolve()}: {e}",
                exc_info=True,
            )
            index_dict = None
    else:
        logger.warning(
            f"[{calling_module_name}] Файл индекса {idx_path.resolve()} не найден."
        )

    # Проверка согласованности
    if embeddings_array is not None and index_dict is not None:
        if embeddings_array.shape[0] != len(index_dict):
            logger.error(
                f"Несоответствие количества эмбеддингов ({embeddings_array.shape[0]}) и записей в индексе ({len(index_dict)}) для типа '{data_type}'. Данные могут быть некорректны!"
            )

    logger.info(
        f"Загружено {embeddings_array.shape[0]} эмбеддинга(ов) и {len(index_dict)} индекса(ов) типа '{data_type}' из {embeddings_dir.resolve()}"
    )

    return embeddings_array, index_dict


# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---


def transform_coords(
    coords_ref: Union[List[List[float]], List[float], None],
    shape_ref: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Union[List[List[float]], List[float], None]:
    """
    Преобразует координаты из эталонной системы (shape_ref) в целевую (target_shape).
    Предполагается, что обе системы имеют одинаковое соотношение сторон.

    Args:
        coords_ref: Координаты [[x,y,...],...] или [x1,y1,x2,y2] в эталонной системе.
                    Поддерживаются точки с доп. измерениями (например, Z).
        shape_ref: Размеры эталонного изображения (Href, Wref), к которому относятся coords_ref.
        target_shape: Размеры целевого изображения (Htarget, Wtarget).

    Returns:
        Преобразованные координаты в целевой системе или None при ошибке.
        Координаты позы (3 элемента) возвращаются без изменений.
    """
    if coords_ref is None:
        return None

    href, wref = shape_ref
    htarget, wtarget = target_shape

    # Проверка корректности размеров
    if href <= 0 or wref <= 0 or htarget <= 0 or wtarget <= 0:
        logger.warning(
            "Некорректные размеры изображения (<=0) переданы в transform_coords."
        )
        # Возвращаем копию, чтобы избежать модификации исходных данных, если они изменяемы
        return (
            [p[:] for p in coords_ref]
            if isinstance(coords_ref, list)
            else coords_ref[:]
        )

    # Проверка соотношения сторон (с допуском на ошибки округления)
    ref_aspect = wref / href
    target_aspect = wtarget / htarget
    if abs(ref_aspect - target_aspect) > 1e-5:
        logger.warning(
            f"Соотношения сторон эталонного ({ref_aspect:.4f}) и целевого ({target_aspect:.4f}) изображений не совпадают. Результат масштабирования может быть искажен."
        )
        # Продолжаем с предупреждением, но результат может быть неточным

    # Масштабные коэффициенты
    scale_x = wtarget / wref
    scale_y = htarget / href

    # Если масштаб не изменился, возвращаем копию исходных координат
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return (
            [p[:] for p in coords_ref]
            if isinstance(coords_ref, list)
            else coords_ref[:]
        )

    transformed_coords = []
    try:
        # Определяем тип координат
        is_list_of_points = (
            isinstance(coords_ref, list)
            and len(coords_ref) > 0
            and isinstance(coords_ref[0], (list, tuple))
        )

        if is_list_of_points:
            # Обрабатываем список точек [[x, y, (z)], ...]
            for point in coords_ref:
                if len(point) >= 2:
                    x_new = point[0] * scale_x
                    y_new = point[1] * scale_y
                    # Копируем остальные элементы (например, Z), если они есть
                    new_point = [x_new, y_new] + point[2:]
                    transformed_coords.append(new_point)
                else:
                    # Оставляем некорректные точки как есть (или можно их отфильтровать)
                    logger.warning(
                        f"Некорректная точка ({point}) пропущена при трансформации."
                    )
                    transformed_coords.append(point[:])  # Добавляем копию
        elif isinstance(coords_ref, (list, tuple)) and len(coords_ref) == 4:
            # Обрабатываем bbox [x1, y1, x2, y2]
            transformed_coords = [
                coords_ref[0] * scale_x,
                coords_ref[1] * scale_y,
                coords_ref[2] * scale_x,
                coords_ref[3] * scale_y,
            ]
        elif isinstance(coords_ref, (list, tuple)) and len(coords_ref) == 3:
            # Предполагаем, что это pose [yaw, pitch, roll] - она не масштабируется
            return coords_ref[:]  # Возвращаем копию
        else:
            logger.warning(
                f"Неподдерживаемый формат координат для transform_coords: {type(coords_ref)}, len={len(coords_ref) if isinstance(coords_ref, (list, tuple)) else 'N/A'}"
            )
            return (
                [p[:] for p in coords_ref]
                if isinstance(coords_ref, list)
                else coords_ref[:]
            )  # Возвращаем копию

        return transformed_coords

    except Exception as e:
        logger.error(f"Ошибка при преобразовании координат: {e}", exc_info=True)
        return (
            [p[:] for p in coords_ref]
            if isinstance(coords_ref, list)
            else coords_ref[:]
        )  # Возвращаем копию


# Не забудьте добавить transform_coords в __all__ в fc_lib/__init__.py
# Например: __all__ = [ ..., "fc_utils", "transform_coords", ... ] # Или просто экспортируйте fc_utils


def get_max_workers() -> int:
    """Определяет оптимальное число воркеров с учетом GPU и CPU."""
    import os

    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 16)  # Ограничение для I/O задач


def get_best_provider(config: Dict[str, Any], cache_path_str: str) -> Tuple[str, List[dict]]:
    """
    Определяет наилучший провайдер для выполнения модели на основе секции [provider] конфига.

    Args:
        config: Словарь с конфигурацией провайдера (секция [provider] из face_config.toml).

    Returns:
        Кортеж: (имя_выбранного_провайдера, список_опций_провайдера).
    """
    logger.debug(get_message("INFO_PROVIDER_DETECTION"))
    try:
        available_providers = ort.get_available_providers()
        logger.info(
            get_message("INFO_AVAILABLE_PROVIDERS", providers=available_providers)
        )
    except Exception as e:
        logger.error(
            f"Не удалось получить список доступных провайдеров ONNX Runtime: {e}"
        )
        return "CPUExecutionProvider", [{}]  # Fallback to CPU

    # Получаем путь к кэшу TensorRT
    #cache_path_str = config.get("tensorRT_cache_path", "TensorRT_cache")
    cache_path = Path(cache_path_str)

    if not cache_path.is_absolute():
        cache_path = (Path.cwd() / cache_path).resolve()
    else:
        cache_path = cache_path.resolve()
    logger.info(f"Путь к кэшу TensorRT: {cache_path}")

    # Получаем предпочтительный провайдер
    preferred_provider = config.get("provider_name")

    if preferred_provider and preferred_provider in available_providers:
        selected_provider = preferred_provider
        logger.debug(
            f"Выбран предпочтительный провайдер из конфигурации: {selected_provider}"
        )
    else:
        preferred_order = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected_provider = next(
            (p for p in preferred_order if p in available_providers),
            "CPUExecutionProvider",
        )
        logger.info(
            f"Предпочтительный провайдер не указан/недоступен. Автовыбран: {selected_provider}"
        )

    # Формируем опции
    provider_options = []
    if selected_provider == "TensorrtExecutionProvider":
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            options = {
                "device_id": str(config.get("device_id", "0")),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_path),
                "trt_fp16_enable": config.get("trt_fp16_enable", True),
                "trt_max_workspace_size": str(
                    config.get("trt_max_workspace_size", "1073741824")
                ),
            }
            provider_options.append(options)
            logger.debug(f"Опции для TensorrtExecutionProvider: {options}")
        except Exception as e:
            logger.error(
                f"Ошибка при создании кэша TensorRT ({cache_path}) или опций: {e}"
            )
            logger.warning("Откат к CPUExecutionProvider.")
            selected_provider = "CPUExecutionProvider"
            provider_options = [{}]

    elif selected_provider == "CUDAExecutionProvider":
        options = {"device_id": str(config.get("device_id", "0"))}
        # Добавляем gpu_mem_limit, если он указан и > 0
        gpu_mem_limit = config.get("gpu_mem_limit")
        if (
            gpu_mem_limit is not None
            and isinstance(gpu_mem_limit, int)
            and gpu_mem_limit > 0
        ):
            options["gpu_mem_limit"] = str(gpu_mem_limit)  # ONNX Runtime ожидает строку
            logger.info(f"Установлен лимит памяти GPU для CUDA: {gpu_mem_limit} байт")
        elif gpu_mem_limit is not None:
            logger.warning(
                f"Некорректное значение gpu_mem_limit: {gpu_mem_limit}. Лимит не установлен."
            )

        provider_options.append(options)
        logger.debug(f"Опции для CUDAExecutionProvider: {options}")
    else:  # CPUExecutionProvider или другой
        provider_options.append({})
        logger.debug(f"Опции для {selected_provider}: {{}}")

    return selected_provider, provider_options


def preprocess_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Преобразует изображение для анализа лиц (пример)."""
    # Этот препроцессинг может быть не универсален, зависит от модели
    try:
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        # Пример: нормализация 0-1
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Пример: BGR -> RGB и HWC -> CHW
        # img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        # img_chw = np.transpose(img_rgb, (2, 0, 1))
        # return np.expand_dims(img_chw, axis=0)
        return img_normalized  # Возвращаем просто нормализованный ресайз для примера
    except Exception as e:
        logger.error(f"Ошибка preprocess_image: {e}")
        # Вернуть пустой массив или обработать ошибку иначе
        return np.array([])


def rescale_keypoints(
    keypoints: List[List[float]],
    original_shape: Tuple[int, int],
    target_size: Tuple[int, int],
    expect_3d: bool = False,
) -> List[List[float]]:
    """Пересчитывает координаты ключевых точек ИЗ целевого В исходный."""
    # Эта функция может быть не нужна или требовать пересмотра
    orig_h, orig_w = original_shape
    target_h, target_w = target_size
    scale = min(
        target_h / orig_h if orig_h > 0 else 1, target_w / orig_w if orig_w > 0 else 1
    )
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    rescaled_keypoints = []
    inv_scale = 1.0 / scale if scale > 1e-6 else 1.0  # Обратный масштаб
    for kp in keypoints:
        if len(kp) >= 2:
            x_target, y_target = kp[0], kp[1]
            # Обратное преобразование
            x_orig = (x_target - left) * inv_scale
            y_orig = (y_target - top) * inv_scale
            if len(kp) == 3 and expect_3d:
                rescaled_keypoints.append([x_orig, y_orig, kp[2]])
            else:
                rescaled_keypoints.append([x_orig, y_orig])
        else:
            logger.warning(get_message("WARNING_INVALID_KEYPOINT", kp=kp))
            rescaled_keypoints.append([0.0, 0.0, 0.0] if expect_3d else [0.0, 0.0])
    return rescaled_keypoints


def rescale_to_target(
    keypoints: List[List[float]],
    current_shape: Tuple[int, int],
    target_size: Tuple[int, int],
    expect_3d: bool = False,
) -> List[List[float]]:
    """Пересчитывает координаты ключевых точек ИЗ текущего В целевой."""
    # Эта функция может быть не нужна или требовать пересмотра
    curr_h, curr_w = current_shape
    target_h, target_w = target_size
    scale = min(
        target_h / curr_h if curr_h > 0 else 1, target_w / curr_w if curr_w > 0 else 1
    )
    new_h, new_w = int(curr_h * scale), int(curr_w * scale)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    rescaled_keypoints = []
    for kp in keypoints:
        if len(kp) >= 2:
            x_curr, y_curr = kp[0], kp[1]
            x_target = left + x_curr * scale
            y_target = top + y_curr * scale
            if len(kp) == 3 and expect_3d:
                rescaled_keypoints.append([x_target, y_target, kp[2]])
            else:
                rescaled_keypoints.append([x_target, y_target])
        else:
            rescaled_keypoints.append([0.0, 0.0, 0.0] if expect_3d else [0.0, 0.0])
    return rescaled_keypoints


def save_json(data: Any, output_file: Path, success_message_key: str) -> None:
    """Сохраняет данные в JSON-файл."""
    try:
        output_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # Создаем директорию, если нужно
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # Используем indent=2
        logger.debug(get_message(success_message_key, output_file=output_file.resolve()))
    except Exception as e:
        logger.error(
            get_message("ERROR_SAVING_JSON", output_file=output_file.resolve(), exc=e),
            exc_info=True,
        )


def normalize_path(path: Path, base_dir: Path) -> str:
    """Преобразует путь в относительный URL-совместимый формат."""
    try:
        # Используем Path.relative_to для получения относительного пути
        relative_p = path.relative_to(base_dir)
        # Заменяем разделители и кодируем для URL
        return quote(str(relative_p).replace("\\", "/"))
    except ValueError:
        # Если путь не находится внутри base_dir, возвращаем абсолютный путь как URI
        logger.debug(
            f"Путь {path} не является дочерним для {base_dir}. Используется file URI."
        )
        return path.as_uri()
    except Exception as e:
        logger.error(f"Ошибка нормализации пути {path} относительно {base_dir}: {e}")
        # Возвращаем абсолютный путь как URI в случае других ошибок
        return path.as_uri()


def pad_description(description: str, length: int = 30) -> str:
    """Дополняет описание пробелами до указанной длины."""
    return description.ljust(length)
