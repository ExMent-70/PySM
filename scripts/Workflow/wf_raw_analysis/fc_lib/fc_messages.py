# fc_lib/fc_messages.py
"""Модуль для хранения текстовых сообщений проекта."""

import logging  # Добавим импорт logging
from typing import Dict

# Используем print для ранних ошибок, когда логгер может быть не настроен
# logger = logging.getLogger(__name__) # Можно инициализировать, но осторожно

MESSAGES: Dict[str, str] = {
    # Информация (INFO)
    "INFO_JSON_PORTRAIT_UPDATE": "Обновлено {update_count} портретных записей (валидные кластеры), ошибок: {fail_count}.",
    "INFO_MOVE_FILES_TO_CLUSTERS_DISABLED": "Перемещение/копирование файлов в папки кластеров отключено в конфигурации ('task.move_files_to_claster' = false).",
    "INFO_CONFIG_LOADED": "Конфигурация успешно загружена из {config_path}",
    "INFO_JSON_PORTRAIT_SAVED": "JSON файл портретных данных сохранен: {output_file}",
    "INFO_JSON_GROUP_SAVED": "JSON файл групповых данных сохранен: {output_file}",
    "INFO_EMBEDDING_INDEX_SAVED": "Файл индекса эмбеддингов сохранен: {output_file}",
    "INFO_GROUP_TO_PORTRAIT_SAVED": "JSON сопоставлений (группы -> портреты) сохранен: {output_file}",
    "INFO_PORTRAIT_TO_GROUP_SAVED": "JSON сопоставлений (портреты -> группы) сохранен: {output_file}",
    "INFO_PROCESSING_START": "Обработка папки: {folder_path}",
    "INFO_IMAGES_FOUND": "Режим обработки: {image_type_selected}{sorted_list}, Найдено  {count} изображений для обработки",
    "INFO_PROCESSING_DONE": "Обработка завершена. Результаты: {portrait_file}, {group_file}",
    "INFO_CLUSTERS_FOUND": "Получено {n_clusters} кластеров и {n_noise} шумовых точек",
    "INFO_STATISTICS_SAVED": "Статистика сохранена в {output_file}",
    "INFO_PROVIDER_DETECTION": "Определение доступных провайдеров ONNX Runtime...",
    "INFO_AVAILABLE_PROVIDERS": "Доступные провайдеры ONNX: {providers}",
    "INFO_DISTANCE_PLOT_SAVED": "График расстояний сохранен в {output_file}",
    "INFO_MATCHING_THRESHOLD": "Сопоставление с порогом {metric}: {threshold:.4f}",
    "INFO_MATCHES_FOUND": "Найдено {count} сопоставлений портретных и групповых кластеров",
    "INFO_MATCHES_SAVED": "Файл сопоставлений сохранен ({count} кластеров): {output_file}",
    "INFO_KEYPOINTS_DISABLED": "Анализ ключевых точек отключён в конфигурации ('task.keypoint_analysis' = false).",
    "INFO_FILE_MOVED": "{action} файл {filename} в {cluster_name}",
    "INFO_GROUP_PHOTO_MOVED": "{action} групповая фотография {filename}",
    "INFO_REPORT_DISABLED": "Генерация HTML-отчёта отключена в конфигурации ('task.generate_html' = false).",
    "INFO_LOADING_JSON_FOR_REPORT": "Загрузка JSON-файлов для отчёта...",
    "INFO_PREPARING_PORTRAIT_CLUSTERS": "Подготовка данных портретных кластеров...",
    "INFO_PREPARING_MATCHES": "Подготовка данных сопоставлений...",
    "INFO_GENERATING_VISUALIZATION": "Визуализация эмбеддингов...",
    "INFO_REPORT_SAVED": "HTML-отчёт сохранён в {output_file}",
    "INFO_VISUALIZATION_SAVED": "Визуализация эмбеддингов сохранена: {output_file}",  # Уточнить, сохраняется ли отдельно
    "INFO_FACE_ANALYSIS_INITIALIZED": "Модель FaceAnalysis успешно инициализирована.",
    "INFO_NO_FACES_FOR_XMP": "Нет данных о лицах для создания XMP-файлов.",
    "INFO_NO_PORTRAIT_CLUSTERS": "Нет портретных кластеров для сопоставления.",
    "INFO_NO_GROUP_PHOTOS": "Нет групповых фотографий для обработки.",
    "INFO_EMOTION_MODEL_LOADED": "Модель эмоций загружена: {model_path}",  # Не используется?
    "INFO_KEYPOINT_ANALYSIS_START": "Запуск анализа ключевых точек...",
    "INFO_KEYPOINT_ANALYSIS_COMPLETE": "Анализ ключевых точек завершён.",
    "INFO_IMAGE_PROCESSING_START": "Начинаем обработку изображений...",
    "INFO_IMAGE_PROCESSING_COMPLETE": "Обработка изображений завершена.",
    "INFO_HTML_REPORT_GENERATION_START": "Генерация HTML-отчета...",
    "INFO_HTML_REPORT_GENERATED": "HTML-отчет сгенерирован.",
    "INFO_MEMORY_USAGE": "Использование памяти: {percent}% (Доступно: {available:.1f} MB, Всего: {total:.1f} MB)",
    "INFO_JSON_UPDATED": "JSON-файл успешно обновлён: {file_path}",  # Добавлено в пред. итерации
    "INFO_JSON_SAVED_NEW": "Создан новый JSON-файл: {file_path}",  # Добавлено в пред. итерации
    # Отладка (DEBUG)
    "DEBUG_IMAGE_PROCESSED": "Обработано {filename}: {face_count} лиц, shape={shape}",
    "DEBUG_IMAGE_RESIZED": "Изображение {filename} ({original_shape}) -> ({new_shape})",
    "DEBUG_XMP_CREATED": "XMP создан для {filename}: {xmp_file}",
    "DEBUG_XMP_FILE_COPIED": "XMP скопирован: {src_xmp}",  # Упрощено
    "DEBUG_GROUP_XMP_FILE_COPIED": "Групповой XMP скопирован: {src_xmp}",  # Упрощено
    "DEBUG_EYE_STATE_ANALYSIS": "Глаз: EAR={ear:.3f}, NormEAR={normalized_ear:.3f}, ZDiff={z_diff:.3f}",
    "DEBUG_EMOTION_ANALYSIS": "Эмоция: {emotion}, Увер.: {confidence:.2f}",  # Не используется?
    "DEBUG_IMAGE_SCALE": "Shape: {shape}, Scale: {scale:.4f}",  # Упрощено
    "DEBUG_POINT_RADIUS": "Радиус точки: {radius}",  # Не используется?
    "DEBUG_KEYPOINTS_SAMPLE": "Пример KPS: {keypoints}",
    "DEBUG_POINT_OUT_OF_BOUNDS": "Точка ({x},{y}) вне границ {shape}",  # Не используется?
    "DEBUG_IMAGE_PROCESSING": "Обработка {image_path}: Shape={shape}, Target={target_size}, KPS={keypoints}",  # Упрощено
    "DEBUG_IMAGE_OVERWRITTEN": "Изображение перезаписано: {image_path}",  # Не используется?
    "DEBUG_BASE_DIR_SELECTION": "Базовая дир. для отчета: {base_dir} (move_files={move_files})",
    "DEBUG_RELATIVE_PATH": "Отн. путь для {filename}: {rel_path}",
    "DEBUG_JPEG_SAVED": "JPEG из {filename} сохранен в {output_file}",
    "DEBUG_APPLYING_GREEN_KPS": "Нанесение KPS на {filename}, kps: {kps}",  # Не используется?
    "DEBUG_RECALCULATED_KPS_ORIG": "Пересчитанные KPS для {filename}: {kps_orig}",  # Не используется?
    "DEBUG_SAVED_GREEN_KPS_IMAGE": "Сохранено отладочное KPS изображение: {filename}",  # Не используется?
    "DEBUG_FACE_PROCESSING": "Обработка лица: {face_id}, KPS: {keypoints}",
    "DEBUG_HEAD_POSE_ANALYSIS": "Поза головы: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f} -> {result}",
    "DEBUG_INSUFFICIENT_CHILD_NAMES": "Не хватило имен для кластера {label}. Присвоено временное.",
    # Предупреждения (WARNING)
    "WARNING_KPS_OUT_OF_BOUNDS": "Координаты KPS вне изображения для {filename}: ({kp_x}, {kp_y})",
    "WARNING_NO_IMAGES": "Не найдено изображений для обработки в {folder_path}",
    "WARNING_NO_FACES": "Лица не найдены в {filename}",
    "WARNING_IMAGE_NOT_FOUND": "Изображение не найдено: {image_path}",
    "WARNING_INVALID_KEYPOINT": "Некорректный формат ключевой точки: {kp}",
    "WARNING_EMOTION_MODEL_NOT_FOUND": "Модель эмоций {model_path} не найдена",  # Не используется?
    "WARNING_FILE_NOT_FOUND": "Файл {filename} не найден для {action}",
    "WARNING_FILE_ACTION_FAILED": "Ошибка {action} файла {filename} в {destination}: {exc}",
    "WARNING_EMBEDDINGS_NOT_FOUND": "Файл эмбеддингов {file_path} не найден",
    "WARNING_INSUFFICIENT_DATA": "Недостаточно данных для визуализации (требуется >= 2 точек)",
    "WARNING_NO_PORTRAIT_EMBEDDINGS": "Нет эмбеддингов для портретных фотографий.",
    "WARNING_EMOTION_MODEL_NOT_LOADED": "Модель эмоций не загружена.",  # Не используется?
    "WARNING_FONT_NOT_FOUND": "Шрифт Arial не найден, используется шрифт по умолчанию.",  # Не используется?
    "WARNING_IMAGE_PROCESSING_FAILED": "Не удалось обработать изображение {image_path}: {exc}",
    "WARNING_NO_2D_KEYPOINTS": "Ключевые точки 2D отсутствуют для {item}",
    "WARNING_NO_JPEG_IN_RAW": "В RAW-файле {filename} отсутствует встроенный JPEG-превью.",
    "WARNING_PREVIEW_TOO_SMALL": "Превью в {filename} ({long_side}px) меньше мин. размера ({min_size}px).",
    "WARNING_NO_POSE_DATA": "Отсутствуют данные о позе головы.",
    # Ошибки (ERROR)
    "ERROR_CONFIG_LOAD": "Ошибка загрузки конфигурации из {config_path}: {exc}",
    "ERROR_PROCESSING_IMAGE": "Ошибка обработки файла {file_path}: {exc}",
    "ERROR_SAVING_JSON": "Ошибка сохранения JSON в {output_file}: {exc}",
    "ERROR_SAVING_XMP": "Ошибка сохранения XMP для {file_path}: {exc}",
    "ERROR_LOADING_EMBEDDINGS": "Ошибка при загрузке эмбеддингов: {exc}",
    "ERROR_LOADING_JSON": "Ошибка загрузки JSON из {file_path}: {exc}",
    "ERROR_ANALYZING_EYES": "Ошибка анализа глаз: {exc}",  # Не используется?
    "ERROR_ANALYZING_MOUTH": "Ошибка анализа рта: {exc}",  # Не используется?
    "ERROR_ANALYZING_HEAD": "Ошибка анализа головы: {exc}",  # Не используется?
    "ERROR_ANALYZING_EMOTIONS": "Ошибка анализа эмоций: {exc}",  # Не используется?
    "ERROR_UPDATING_XMP": "Ошибка обновления XMP {file_path}: {exc}",
    "ERROR_SAVING_REPORT": "Ошибка сохранения HTML-отчета: {exc}",
    "ERROR_IMAGE_DECODE_FAILED": "Не удалось декодировать изображение.",  # Не используется?
    "ERROR_XMP_CREATION_FAILED": "Ошибка при создании XMP-файла: {exc}",  # Не используется?
    "ERROR_UNSUPPORTED_CLUSTERING_ALGORITHM": "Неподдерживаемый алгоритм кластеризации: {algorithm}",
    "ERROR_EMOTION_MODEL_LOAD_FAILED": "Не удалось загрузить модель эмоций: {exc}",  # Не используется?
    "ERROR_KEYPOINT_ANALYSIS_FAILED": "Ошибка при анализе ключевых точек: {exc}",
    "ERROR_PROCESSING_FAILED": "Ошибка в процессе обработки: {exc}",  # Не используется?
    "ERROR_LOADING_JSON_FILES": "Ошибка загрузки JSON-файлов: {exc}",  # Не используется?
    "ERROR_INVALID_FACE_TYPE": "Некорректный тип данных лица для {key}: {type}, данные: {face}",
}


def get_message(key: str, **kwargs) -> str:
    """Форматирует сообщение с подстановкой параметров."""
    message = MESSAGES.get(key)
    if message is None:
        # Логируем или возвращаем плейсхолдер, если ключ не найден
        # Используем print, т.к. logger может быть еще не настроен
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: Ключ сообщения '{key}' не найден в fc_messages.MESSAGES."
        )
        return f"[Сообщение '{key}' не найдено]"
    try:
        return message.format(**kwargs)
    except KeyError as e:
        # Ошибка: не хватает аргумента для форматирования
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: Отсутствует аргумент '{e}' для форматирования сообщения '{key}'."
        )
        return message  # Возвращаем неформатированное сообщение
    except Exception as e:
        # Другие ошибки форматирования
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка форматирования сообщения '{key}': {e}")
        return message
