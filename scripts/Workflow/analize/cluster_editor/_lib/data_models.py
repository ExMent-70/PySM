# analize/cluster_editor/_lib/data_models.py
"""
Модуль моделей данных для редактора кластеров.
Определяет структуры данных для Лица (Face) и Изображения (ImageRecord).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class Face:
    """
    Представляет данные одного лица на фотографии.
    
    Attributes:
        bbox: Координаты лица [top, right, bottom, left].
        cluster_label: ID кластера (число), к которому привязано лицо.
        child_name: Имя человека, привязанного к этому лицу.
        extra_data: Дополнительные данные (например, дистанция совпадения).
        embedding_key: Уникальный ключ для связи с вектором в кэше (не сохраняется в JSON).
    """
    bbox: List[float]
    cluster_label: Optional[int] = None
    child_name: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    # Поля для хранения исходного состояния (для отслеживания изменений)
    original_cluster_label: Optional[int] = None
    original_child_name: Optional[str] = None

    # Поля для UI (заполняются динамически)
    filename: str = ""
    effective_name: str = ""
    
    # Временный ключ для связи с numpy-массивом векторов в памяти.
    # Формат: 'filename' (портреты) или 'filename::face_index' (группы).
    embedding_key: Optional[str] = None 

    @property
    def is_changed(self) -> bool:
        """Возвращает True, если cluster_label или child_name были изменены."""
        return (self.cluster_label != self.original_cluster_label or
                self.child_name != self.original_child_name)

    def commit_changes(self):
        """Фиксирует текущее состояние как исходное (сбрасывает флаг is_changed)."""
        self.original_cluster_label = self.cluster_label
        self.original_child_name = self.child_name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Face":
        """Создает объект Face из словаря."""
        known_fields = {'bbox', 'cluster_label', 'child_name'}
        light_data = {k: data[k] for k in known_fields if k in data}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        instance = cls(**light_data)
        instance.extra_data = extra_data
        
        # При загрузке текущее состояние считается исходным
        instance.original_cluster_label = instance.cluster_label
        instance.original_child_name = instance.child_name
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует объект в словарь для сохранения в JSON."""
        data = self.extra_data.copy()
        data.update({
            'bbox': self.bbox,
            'cluster_label': self.cluster_label,
            'child_name': self.child_name
        })
        return data


@dataclass
class ImageRecord:
    """
    Представляет запись об изображении (файле).
    Может быть типа 'portrait' (одиночное) или 'group' (групповое).
    """
    filename: str
    image_type: str
    faces: List[Face]
    original_shape: Tuple[int, int]
    location_cluster: Optional[int] = None
    location_name: Optional[str] = None
    
    # Поля для хранения исходного состояния
    original_image_type: str = ""
    original_location_cluster: Optional[int] = None
    original_location_name: Optional[str] = None
    
    # Список для хранения лиц, скрытых при конвертации Группа -> Портрет.
    # Позволяет восстановить данные при обратном переносе.
    removed_faces: List[Face] = field(default_factory=list)

    @property
    def is_changed(self) -> bool:
        """Вычисляет, было ли изменено состояние записи (тип, локация или любое из лиц)."""
        if (self.image_type != self.original_image_type or
                self.location_cluster != self.original_location_cluster or
                self.location_name != self.original_location_name):
            return True
            
# --- НАЧАЛО ИЗМЕНЕНИЯ: Удалена ошибочная проверка ---
        # Было: if self.removed_faces: return True
        # Эта проверка вызывала ложное срабатывание при загрузке файла с историей.
        # Реальные изменения отслеживаются через image_type и face.is_changed.
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        return any(face.is_changed for face in self.faces)
    
    def commit_changes(self):
        """Фиксирует текущее состояние как исходное."""
        self.original_image_type = self.image_type
        self.original_location_cluster = self.location_cluster
        self.original_location_name = self.location_name
        
        # При сохранении на диск мы фиксируем изменения лиц.
        # removed_faces остаются в списке, чтобы их можно было сохранить в JSON.
        for face in self.faces:
            face.commit_changes()

    @classmethod
    def from_dict(cls, filename: str, image_type: str, data: Dict[str, Any]) -> "ImageRecord":
        """Создает объект ImageRecord из словаря."""
        raw_faces_data = data.get("faces", [])
        parsed_faces = [Face.from_dict(face_data) for face_data in raw_faces_data]
        
        # Загружаем скрытые лица, если они есть в JSON
        raw_removed = data.get("removed_faces", [])
        parsed_removed = [Face.from_dict(face_data) for face_data in raw_removed]
        
        instance = cls(
            filename=filename,
            image_type=image_type,
            faces=parsed_faces,
            location_cluster=data.get("location_cluster"),
            location_name=data.get("location_name"),
            original_shape=tuple(data.get("original_shape", [0, 0])),
            removed_faces=parsed_removed
        )
        # Запоминаем исходное состояние
        instance.original_image_type = instance.image_type
        instance.original_location_cluster = instance.location_cluster
        instance.original_location_name = instance.location_name
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует объект в словарь."""
        result = {
            "faces": [face.to_dict() for face in self.faces],
            "location_cluster": self.location_cluster,
            "location_name": self.location_name,
            "original_shape": self.original_shape
        }
        # Сохраняем скрытые лица, чтобы возможность восстановления
        # сохранялась даже после перезапуска приложения
        if self.removed_faces:
            result["removed_faces"] = [face.to_dict() for face in self.removed_faces]
            
        return result