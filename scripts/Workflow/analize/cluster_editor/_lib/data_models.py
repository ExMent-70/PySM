# analize/cluster_editor/_lib/data_models.py

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class Face:
    """Представляет данные одного лица с отслеживанием исходного состояния."""
    bbox: List[float]
    cluster_label: Optional[int] = None
    child_name: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    # --- ВОССТАНОВЛЕНО: Поля для хранения исходного состояния ---
    original_cluster_label: Optional[int] = None
    original_child_name: Optional[str] = None

    # Поля для UI
    filename: str = ""
    effective_name: str = ""

    @property
    def is_changed(self) -> bool:
        """Вычисляет, было ли изменено состояние лица (принадлежность или имя)."""
        return (self.cluster_label != self.original_cluster_label or
                self.child_name != self.original_child_name)

    def commit_changes(self):
        """Фиксирует текущее состояние как исходное (после сохранения)."""
        self.original_cluster_label = self.cluster_label
        self.original_child_name = self.child_name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Face":
        known_fields = {'bbox', 'cluster_label', 'child_name'}
        light_data = {k: data[k] for k in known_fields if k in data}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        instance = cls(**light_data)
        instance.extra_data = extra_data
        # Запоминаем исходное состояние при загрузке
        instance.original_cluster_label = instance.cluster_label
        instance.original_child_name = instance.child_name
        return instance

    def to_dict(self) -> Dict[str, Any]:
        data = self.extra_data.copy()
        data.update({
            'bbox': self.bbox,
            'cluster_label': self.cluster_label,
            'child_name': self.child_name
        })
        return data


@dataclass
class ImageRecord:
    """Представляет запись для изображения с отслеживанием исходного состояния."""
    filename: str
    image_type: str
    faces: List[Face]
    original_shape: Tuple[int, int]
    location_cluster: Optional[int] = None
    location_name: Optional[str] = None
    
    # --- ВОССТАНОВЛЕНО: Поля для хранения полного исходного состояния ---
    original_image_type: str = ""
    original_location_cluster: Optional[int] = None
    original_location_name: Optional[str] = None

    @property
    def is_changed(self) -> bool:
        """Вычисляет, было ли изменено состояние записи (тип, локация или любое из лиц)."""
        if (self.image_type != self.original_image_type or
                self.location_cluster != self.original_location_cluster or
                self.location_name != self.original_location_name):
            return True
        return any(face.is_changed for face in self.faces)
    
    def commit_changes(self):
        """Фиксирует текущее состояние как исходное."""
        self.original_image_type = self.image_type
        self.original_location_cluster = self.location_cluster
        self.original_location_name = self.location_name
        for face in self.faces:
            face.commit_changes()

    @classmethod
    def from_dict(cls, filename: str, image_type: str, data: Dict[str, Any]) -> "ImageRecord":
        raw_faces_data = data.get("faces", [])
        parsed_faces = [Face.from_dict(face_data) for face_data in raw_faces_data]
        
        instance = cls(
            filename=filename,
            image_type=image_type,
            faces=parsed_faces,
            location_cluster=data.get("location_cluster"),
            location_name=data.get("location_name"),
            original_shape=tuple(data.get("original_shape", [0, 0]))
        )
        # Запоминаем полное исходное состояние
        instance.original_image_type = instance.image_type
        instance.original_location_cluster = instance.location_cluster
        instance.original_location_name = instance.location_name
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faces": [face.to_dict() for face in self.faces],
            "location_cluster": self.location_cluster,
            "location_name": self.location_name,
            "original_shape": self.original_shape
        }