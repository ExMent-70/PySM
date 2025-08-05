# analize/analyze_faces/face_lib/__init__.py
"""
Локальная библиотека для этапа анализа лиц.

Этот файл собирает ключевые классы из модулей этой библиотеки,
чтобы их можно было удобно импортировать одним оператором:
`from analyze_faces.face_lib import ConfigManager, FaceAnalyzer`
"""

from .config_loader import ConfigManager
from .onnx_manager import ONNXModelManager
from .coordinate_transformer import CoordinateTransformer
from .face_data_processor_interface import FaceDataProcessorInterface
from .attribute_analyzer import AttributeAnalyzer
from .face_analyzer import FaceAnalyzer

__all__ = [
    "ConfigManager",
    "ONNXModelManager",
    "CoordinateTransformer",
    "FaceDataProcessorInterface",
    "AttributeAnalyzer",
    "FaceAnalyzer",
]