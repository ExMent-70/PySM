
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..data_models import ImageRecord, Face
from ..student_roster import StudentRoster

def natural_sort_key(text: str) -> List[Any]:
    """Утилита для натуральной сортировки строк (например, 'photo-2' встанет перед 'photo-10')."""
    return[int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

class EditorStrategy(ABC):
    """
    Абстрактный базовый класс для стратегий режимов редактора.
    Обеспечивает кэширование выборок для O(1) отклика UI.
    """
    
    def __init__(self):
        self._files_cache: Optional[Dict[str, List[str]]] = None
        self.student_roster: Optional[StudentRoster] = None

    def set_student_roster(self, roster: Optional[StudentRoster]) -> None:
        """Подключает реестр, используемый только для вычисляемых подписей."""
        self.student_roster = roster
        self.invalidate_cache()

    def student_name(self, student_id: Optional[str]) -> str:
        if not student_id:
            return ""
        if not self.student_roster:
            return f"Неизвестный ID [{student_id}]"
        return self.student_roster.name_for(student_id)

    def student_label(self, student_id: Optional[str]) -> str:
        if not student_id:
            return ""
        if not self.student_roster:
            return f"Неизвестный ID [{student_id}]"
        return self.student_roster.label_for(student_id)

    def invalidate_cache(self):
        """Сбрасывает кэш (вызывается при перемещении или переименовании данных)."""
        self._files_cache = None

    @property
    @abstractmethod
    def mode_name(self) -> str:
        pass

    @abstractmethod
    def get_window_title(self, session_name: str) -> str:
        pass

    @abstractmethod
    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        pass

    def get_files_for_cluster(self, cluster_id: str, records: Dict[str, ImageRecord]) -> List[str]:
        """
        Универсальный метод с O(1) доступом к списку файлов.
        Автоматически перестраивает кэш, если он был сброшен.
        """
        if self._files_cache is None:
            self._files_cache = self._build_files_cache(records)
        return self._files_cache.get(cluster_id, list())

    @abstractmethod
    def _build_files_cache(self, records: Dict[str, ImageRecord]) -> Dict[str, List[str]]:
        """Внутренний метод для построения обратного индекса файлов (словарь id -> файлы)."""
        pass

    @abstractmethod
    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        pass

    def build_save_outputs(
        self,
        records: Dict[str, ImageRecord],
        paths_config: Dict[str, Path],
    ) -> Dict[Path, Any]:
        """Return additional JSON payloads that must commit with the main file."""

        return {}

    def show_face_details_panel(self) -> bool:
        return True

    def get_preview_image(self, cluster_id: str, faces: List[Face], records: Dict[str, ImageRecord]) -> Optional[str]:
        if faces:
            return faces[0].filename
        return None

    def get_name_prefix(self, cluster_id: str) -> str:
        return ""
        
    def normalize_cluster_name(self, name: str) -> str:
        """Return the persisted cluster name without UI-only prefixes."""

        return name
