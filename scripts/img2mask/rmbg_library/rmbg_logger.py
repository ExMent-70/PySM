# rmbg_library/rmbg_logger.py
import logging
import sys
from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    # Prevent circular import error during type checking
    try:
        from .rmbg_config import LoggingConfig
    except ImportError:
        # Fallback if used outside of the main structure during development
        class LoggingConfig:
            logging_level: str
            log_file: str


# --- Вспомогательный класс для безопасного форматирования ---
class SafeDict(dict):
    """
    Словарь, который возвращает '{key}' вместо выброса KeyError,
    если ключ отсутствует. Используется для безопасного format_map.
    """

    def __missing__(self, key):
        return "{" + str(key) + "}"


# --- Словарь сообщений ---
# (Оставляем ваш словарь MESSAGES без изменений)
MESSAGES: Dict[str, str] = {
    "INFO_CONFIG_LOADED": "Конфигурация успешно загружена из {config_path}",
    "INFO_START_PROCESSING": "Начало обработки файла: {file_path}",
    "INFO_END_PROCESSING": "Завершение обработки файла: {file_path}",
    "INFO_START_MODEL_LOAD": "Загрузка модели {model_name} для процессора {processor}...",
    "INFO_MODEL_LOADED": "Модель {model_name} загружена за {time:.2f} секунд",
    "INFO_PROCESSING_DONE": "Обработка файла {filename} процессором {processor} завершена за {time:.2f} секунд",
    "INFO_SAVING_OUTPUT": "Сохранение результата: {output_path}",
    "DEBUG_IMAGE_SAVED": "Изображение сохранено: {output_path}",
    "DEBUG_MASK_SAVED": "Маска сохранена: {output_path}",
    "DEBUG_PROCESSING_DETAILS": "Детали обработки: {details}",
    "INFO_TOTAL_TIME": "Общее время выполнения: {time:.2f} секунд",
    "INFO_DOWNLOADING_MODEL": "Загрузка/проверка файлов модели {model_name} в {cache_dir}...",  # Изменено сообщение для ясности
    "INFO_MODEL_FILES_DOWNLOADED": "Файлы модели {model_name} успешно загружены/проверены.",  # Изменено сообщение
    "INFO_MODEL_FOUND_CACHE": "Файлы модели {model_name} найдены в кэше: {cache_dir}",
    "INFO_MODEL_CLEARED": "Модель {model_name} очищена из памяти",
    "INFO_NO_IMAGES_FOUND": "В папке {input_dir} не найдено подходящих изображений.",
    "WARNING_SKIPPING_FILE": "Пропуск файла (возможно, не изображение): {file_path}",
    "WARNING_NO_OBJECTS_DETECTED": "Объекты не найдены для промпта '{prompt}' в файле {filename}, результат не сохраняется.",
    "WARNING_PROCESSOR_FALLBACK": "Процессор '{processor_name}' не найден или не реализован.",
    "WARNING_MODEL_SCRIPT_IMPORT": "Не удалось импортировать скрипт модели '{script_path}': {exc}. Функциональность модели '{model_name}' может быть недоступна.",
    "ERROR_CONFIG_LOAD": "Ошибка загрузки конфигурации из {config_path}: {exc}",
    "ERROR_CONFIG_VALIDATION": "Ошибка валидации конфигурации: {exc}",
    "ERROR_MODEL_LOAD": "Ошибка загрузки модели {model_name} для процессора {processor}: {exc}",
    "ERROR_PROCESSING_FILE": "Ошибка обработки файла {file_path} процессором {processor}: {exc}",
    "ERROR_NO_IMAGES": "В папке {input_dir} не найдено изображений",
    "ERROR_UNSUPPORTED_MODEL": "Неподдерживаемая или ненайденная модель: {model_name} для процессора {processor}.",
    "ERROR_UNKNOWN_MODEL_TYPE": "Неизвестный тип модели '{model_type}' для '{model_name}'.",
    "ERROR_UNSUPPORTED_BACKGROUND": "Неподдерживаемый фон: {background}. Поддерживаемые варианты: {options}",
    "ERROR_CONFIG_FILE_NOT_FOUND": "Файл конфигурации '{config_path}' не найден. Пожалуйста, создайте его или укажите правильный путь.",
    "ERROR_DOWNLOADING_MODEL": "Ошибка загрузки файлов модели {model_name}: {exc}",
    "ERROR_SAVING_OUTPUT": "Ошибка сохранения файла {output_path}: {exc}",
    "ERROR_MASK_GENERATION": "Не удалось сгенерировать маску для промпта: {prompt} в файле {filename}",
    "ERROR_GENERIC": "Произошла ошибка: {exc}",
    "ERROR_PROCESSOR_INIT": "Ошибка инициализации процессора '{processor_name}': {exc}",
    "ERROR_MISSING_MODEL_FILES": "Отсутствуют необходимые файлы для модели '{model_name}' в {cache_dir}.",
    "ERROR_MODEL_INFERENCE": "Ошибка выполнения модели '{model_name}' процессором {processor}: {exc}",
}


# --- Исправленная функция get_message ---
def get_message(key: str, **kwargs) -> str:
    """
    Получить сообщение из словаря MESSAGES и подставить параметры.
    Если параметр не найден в kwargs, он останется в фигурных скобках {key}.
    """
    message_template = MESSAGES.get(key, f"Неизвестный ключ сообщения: {key}")
    try:
        # Создаем экземпляр SafeDict из kwargs и передаем его в format_map
        mapping = SafeDict(kwargs)
        return message_template.format_map(mapping)
    except Exception as e:
        # Обработка других возможных ошибок форматирования
        logging.getLogger(__name__).error(
            f"Критическая ошибка форматирования сообщения '{key}': {e}. Шаблон: '{message_template}', Аргументы: {kwargs}"
        )
        # Возвращаем шаблон с указанием на ошибку
        return message_template + f" [ОШИБКА ФОРМАТИРОВАНИЯ: {e}]"


# --- Функция setup_logging ---
# (Остается без изменений)
def setup_logging(config: "LoggingConfig", log_dir: Path = Path(".")):
    """
    Настройка базового логирования.
    :param config: Объект конфигурации секции [logging].
    :param log_dir: Директория для сохранения лог-файла.
    """
    logger = logging.getLogger()  # Корневой логгер

    try:
        log_level_str = config.logging_level.upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
    except AttributeError:
        log_level = logging.INFO
        print(
            f"WARNING: Некорректный уровень логирования '{config.logging_level}', используется INFO.",
            file=sys.stderr,
        )

    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file_path = log_dir / config.log_file
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
    except Exception as e:
        print(
            f"ERROR: Не удалось создать обработчик файла логов {log_file_path}: {e}",
            file=sys.stderr,
        )  # Log before logger ready
        file_handler = None

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    logger.handlers = []
    logger.setLevel(log_level)
    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Логгируем сообщение о настройке после добавления обработчиков
    logger.debug(
        f"Логирование настроено. Уровень: {log_level_str}. Файл: {log_file_path if file_handler else 'НЕ СОЗДАН'}"
    )

    return logger
