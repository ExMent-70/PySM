# fc_lib/__init__.py
# --- ИЗМЕНЕНИЙ НЕ ТРЕБУЕТСЯ ---

from .fc_config import ConfigManager
from .fc_json_data_manager import JsonDataManager
from .fc_face_processor import FaceProcessor
from .fc_cluster_manager import ClusterManager
from .fc_image_loader import ImageLoader
from .fc_face_detector import FaceDetector

# Импортируем сначала модуль, потом берем из него
from . import fc_file_manager
from . import fc_keypoints
from . import fc_report
from . import fc_xmp_utils
from . import fc_utils  # Импортируем сам модуль fc_utils
from . import fc_messages

# Импорты для обработчиков и интерфейса
from .face_data_processor_interface import FaceDataProcessorInterface
from .processor_registry import initialize_face_data_processors
from .fc_attribute_analyzer import AttributeAnalyzer


# Экспортируем нужные классы и функции
FileManager = fc_file_manager.FileManager
run_file_moving = fc_file_manager.run_file_moving

KeypointAnalyzer = fc_keypoints.KeypointAnalyzer
run_keypoint_analysis = fc_keypoints.run_keypoint_analysis

ReportGenerator = fc_report.ReportGenerator
run_html_report_generation = fc_report.run_html_report_generation

run_xmp_creation = fc_xmp_utils.run_xmp_creation

# Экспортируем конкретную функцию из fc_utils
transform_coords = fc_utils.transform_coords
# --- ИЗМЕНЕНИЕ: Экспортируем новую функцию ---
load_embeddings_and_indices = fc_utils.load_embeddings_and_indices
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

# Список того, что экспортируется при 'from fc_lib import *'
# Также используется инструментами статического анализа
__all__ = [
    # Основные классы
    "ConfigManager",
    "JsonDataManager",
    "FaceProcessor",
    "ClusterManager",
    "FileManager",
    "KeypointAnalyzer",
    "ReportGenerator",
    "ImageLoader",
    "FaceDetector",
    "AttributeAnalyzer",  # Добавим сюда, если нужно импортировать напрямую
    # Интерфейсы и реестры
    "FaceDataProcessorInterface",
    "initialize_face_data_processors",
    # Функции-обертки
    "run_file_moving",
    "run_keypoint_analysis",
    "run_html_report_generation",
    "run_xmp_creation",
    # Модули и утилиты
    "fc_utils",
    "transform_coords",  # Явно экспортируем для удобства
    "load_embeddings_and_indices", # <-- ДОБАВИТЬ ЭТУ СТРОКУ
    "fc_xmp_utils",  # Экспортируем модуль, если нужны XmpManager и т.д.
    "fc_messages",
    "get_message",  # Явно экспортируем get_message
]

# Добавим get_message в __all__ для удобства
get_message = fc_messages.get_message
