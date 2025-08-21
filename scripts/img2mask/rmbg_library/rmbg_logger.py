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
    "INFO_MODEL_FILES_DOWNLOADED": "Файлы модели {model_name} успешно загружены/проверены:",  # Изменено сообщение
    "INFO_MODEL_FOUND_CACHE": "Файлы модели {model_name} найдены в кэше: {cache_dir}",
    "INFO_MODEL_CLEARED": "Модель {model_name} выгружена из памяти",
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
def setup_logging(config: "LoggingConfig", log_dir: Path = Path(".")):
    """
    Настройка базового логирования.
    :param config: Объект конфигурации секции [logging].
    :param log_dir: Директория для сохранения лог-файла.
    """
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Импортируем переменные окружения здесь, чтобы функция была самодостаточной
    try:
        # Импорт PySM
        from pysm_lib import pysm_context
        IS_MANAGED_RUN = True
    except ImportError as e:
        IS_MANAGED_RUN = False
        pysm_context = None

    # 1. Определяем уровень логирования с приоритетом PySM
    log_level_str_from_config = config.logging_level.upper()
    log_level_str = log_level_str_from_config

    if IS_MANAGED_RUN and pysm_context:
        # У PySM высший приоритет
        log_level_str = pysm_context.get("sys_log_level", log_level_str_from_config).upper()
    
    # Преобразуем строку в объект уровня логирования
    log_level = getattr(logging, log_level_str, logging.INFO)

    # 2. Создаем форматтеры в зависимости от уровня
    # Исправлены опечатки (%% и ss)
    if log_level <= logging.INFO:
        stream_formatter = logging.Formatter("%(message)s")
    else: # DEBUG и выше
        stream_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    
    # Для файла всегда используем подробный формат
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 3. Настраиваем корневой логгер, НЕ ИСПОЛЬЗУЯ basicConfig
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Очищаем все предыдущие обработчики, чтобы избежать дублирования вывода
    logger.handlers = []

    # 4. Создаем и добавляем обработчик для вывода в консоль (stdout/stderr)
    # PySM обычно лучше работает с stderr
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # 5. Создаем и добавляем обработчик для вывода в файл
    log_file_path = log_dir / config.log_file
    try:
        # mode='w' перезаписывает лог при каждом запуске, 'a' - дописывает в конец
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8", mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Теперь сообщение об ошибке будет выведено в уже настроенный stream_handler
        logger.error(f"Не удалось создать обработчик файла логов {log_file_path}: {e}")
        file_handler = None # Убедимся, что переменная существует

    # Логгируем сообщение о настройке после добавления обработчиков
    # Используем .debug(), чтобы это сообщение не мешало при обычном запуске (INFO)
    logger.debug(
        f"Логирование настроено. Уровень: {log_level_str}. Файл: {log_file_path if file_handler else 'НЕ СОЗДАН'}"
    )

    return logger

