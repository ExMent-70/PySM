# analize/analyze_faces/face_lib/face_data_processor_interface.py
"""
Определяет интерфейс (протокол) для всех классов, которые могут
обрабатывать данные одного лица и добавлять к ним свою информацию.
"""

from typing import Protocol, Dict, Any

# Импортируем наши новые классы для тайп-хинтинга
from .config_loader import ConfigManager
from .onnx_manager import ONNXModelManager


class FaceDataProcessorInterface(Protocol):
    """
    Интерфейс для модулей, которые анализируют данные одного лица
    и добавляют результаты в словарь метаданных.
    """

    def __init__(self, config_manager: ConfigManager, onnx_manager: ONNXModelManager):
        """
        Инициализирует обработчик с явным доступом к менеджеру конфигурации
        и менеджеру ONNX-сессий.
        """
        self.config_manager = config_manager
        self.onnx_manager = onnx_manager
        ...

    @property
    def is_enabled(self) -> bool:
        """Свойство, указывающее, включен ли этот обработчик в конфиге."""
        ...

    def process_face_data(
        self,
        face_data_bundle: Dict[str, Any],
        face_data_dict: Dict[str, Any]
    ) -> None:
        """
        Обрабатывает данные одного лица и ДОБАВЛЯЕТ/ИЗМЕНЯЕТ поля в face_data_dict.

        Args:
            face_data_bundle (Dict[str, Any]):
                Словарь с временными данными, необходимыми для анализа.
                Содержит 'full_image', 'face_crop' и т.д.
                Этот словарь не сохраняется.

            face_data_dict (Dict[str, Any]):
                Словарь с метаданными лица, который будет сохранен в JSON.
                Этот метод должен добавлять в него свои результаты
                (например, 'gender_faceonnx', 'age_faceonnx').
        """
        ...

# --- Документация по структурам данных (для ясности) ---

# ПРИМЕРНЫЕ КЛЮЧИ в 'face_data_bundle' (передаются из FaceAnalyzer):
# 'full_image': np.ndarray          # Оригинальное полное изображение (BGR)
# 'face_crop': np.ndarray           # Финальный кроп лица, переданный в модели
# 'filename': str                   # Имя исходного файла
# 'face_index': int                 # Индекс лица в файле

# ПРИМЕРНЫЕ КЛЮЧИ, которые УЖЕ ЕСТЬ в 'face_data_dict' (перед вызовом):
# 'bbox': Optional[List[float]]
# 'kps': Optional[List[List[float]]]
# 'det_score': float
# 'landmark_3d_68': Optional[List[List[float]]]
# 'pose': Optional[List[float]]
# 'landmark_2d_106': Optional[List[List[float]]]
# 'gender_insight': Optional[int]
# 'age_insight': Optional[int]