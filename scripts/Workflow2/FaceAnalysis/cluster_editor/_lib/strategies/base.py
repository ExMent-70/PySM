# analize/cluster_editor/_lib/strategies/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

# Импорт моделей (предполагается, что они уровнем выше)
from ..data_models import ImageRecord, Face

class EditorStrategy(ABC):
    """
    Абстрактный базовый класс для стратегий режимов редактора.
    Определяет интерфейс взаимодействия DataManager и UI с бизнес-логикой.
    """

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Уникальный идентификатор режима (face, location, cleaning, matches)."""
        pass

    @abstractmethod
    def get_window_title(self, session_name: str) -> str:
        """Формирует заголовок главного окна."""
        pass

    @abstractmethod
    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        """
        Группирует записи в кластеры согласно логике режима.
        Возвращает: {cluster_id: [List of Face objects]}
        """
        pass

    @abstractmethod
    def get_files_for_cluster(self, cluster_id: str, records: Dict[str, ImageRecord]) -> List[str]:
        """Возвращает список файлов для кластера."""
        pass

    @abstractmethod
    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None: # <--- Добавлено
        """Обрабатывает Drag & Drop перемещение."""
        pass

    @abstractmethod
    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        """Переименовывает кластер."""
        pass

    @abstractmethod
    def save(self, records: Dict[str, ImageRecord], paths_config: Dict[str, Path]) -> bool:
        """Сохраняет изменения на диск."""
        pass

    # --- UI Hooks (Optional) ---

    def show_face_details_panel(self) -> bool:
        """Нужно ли показывать правую панель с лицами."""
        return True

    def get_preview_image(self, cluster_id: str, faces: List[Face], records: Dict[str, ImageRecord]) -> Optional[str]:
        """Определяет обложку кластера для списка."""
        if faces:
            return faces[0].filename
        return None

    def get_name_prefix(self, cluster_id: str) -> str:
        """Возвращает префикс отображения имени (например '01-')."""
        return ""
        
    def _strip_name_prefix(self, name: str) -> str:
        """
        Утилита для очистки имени от префикса (используется UI).
        По умолчанию возвращает имя как есть.
        """
        return name        