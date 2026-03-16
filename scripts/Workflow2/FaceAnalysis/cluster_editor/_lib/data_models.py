# analize/cluster_editor/_lib/data_models.py
"""
Модуль моделей данных для редактора кластеров.
Реализация Способа Б: Immutable Index (face_index).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class Face:
    """
    Представляет данные одного лица на фотографии.
    """
    bbox: List[float]
    cluster_label: Optional[int] = None
    child_name: Optional[str] = None
    
    # --- НОВОЕ ПОЛЕ: Неизменяемый индекс ---
    # Хранит порядковый номер лица в исходном списке (и в файле векторов).
    # Позволяет восстанавливать порядок после объединения списков.
    face_index: Optional[int] = None 
    # ---------------------------------------

    # Поля для Cleaning
    quality_status: str = "ok"
    temp_cluster_label: Optional[int] = None
    temp_child_name: Optional[str] = None

    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    # Исходное состояние
    original_cluster_label: Optional[int] = None
    original_child_name: Optional[str] = None
    original_quality_status: str = "ok"
    original_temp_label: Optional[int] = None

    original_matched_label: Optional[int] = None

    # UI поля
    filename: str = ""
    effective_name: str = ""
    embedding_key: Optional[str] = None 

    @property
    def is_changed(self) -> bool:
        # Проверка стандартных полей
        if (self.cluster_label != self.original_cluster_label or
            self.child_name != self.original_child_name or
            self.quality_status != self.original_quality_status or
            self.temp_cluster_label != self.original_temp_label):
            return True
            
        # --- НОВАЯ ПРОВЕРКА: Matches ---
        # Проверяем, изменился ли лейбл сопоставления в extra_data
        current_match = self.extra_data.get('matched_portrait_cluster_label')
        if current_match != self.original_matched_label:
            return True
            
        return False

    @property
    def is_trash(self) -> bool:
        return self.quality_status in ("trash", "technical_trash")

    def commit_changes(self):
        self.original_cluster_label = self.cluster_label
        self.original_child_name = self.child_name
        self.original_quality_status = self.quality_status
        self.original_temp_label = self.temp_cluster_label
        # Фиксируем состояние матчинга
        self.original_matched_label = self.extra_data.get('matched_portrait_cluster_label')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Face":
        known_fields = {'bbox', 'cluster_label', 'child_name', 'face_index', 
                        'quality_status', 'temp_cluster_label', 'temp_child_name'}
        
        kwargs = {k: data[k] for k in known_fields if k in data}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        instance = cls(**kwargs)
        instance.extra_data = extra_data
        
        instance.original_cluster_label = instance.cluster_label
        instance.original_child_name = instance.child_name
        instance.original_quality_status = instance.quality_status
        instance.original_temp_label = instance.temp_cluster_label
        
        # Инициализируем исходное состояние матчинга из extra_data
        instance.original_matched_label = extra_data.get('matched_portrait_cluster_label')
        
        return instance

    def to_dict(self) -> Dict[str, Any]:
        data = self.extra_data.copy()
        data.update({
            'bbox': self.bbox,
            'face_index': self.face_index,
            'cluster_label': self.cluster_label,
            'child_name': self.child_name,
            'quality_status': self.quality_status,
            'temp_cluster_label': self.temp_cluster_label,
            'temp_child_name': self.temp_child_name
        })
        return data


@dataclass
class ImageRecord:
    """
    Представляет запись об изображении.
    """
    filename: str
    faces: List[Face]
    original_shape: Tuple[int, int]
    
    location_cluster: Optional[int] = None
    location_name: Optional[str] = None
    
    original_location_cluster: Optional[int] = None
    original_location_name: Optional[str] = None
    
    image_type: str = "portrait"
    original_image_type: str = "portrait"
    
    face_count: int = 0
    
    removed_faces: List[Face] = field(default_factory=list)

    @property
    def is_changed(self) -> bool:
        if (self.location_cluster != self.original_location_cluster or
            self.location_name != self.original_location_name or
            self.image_type != self.original_image_type):
            return True
        if any(face.is_changed for face in self.faces):
            return True
        return False
    
    def commit_changes(self):
        self.original_location_cluster = self.location_cluster
        self.original_location_name = self.location_name
        self.original_image_type = self.image_type
        for face in self.faces:
            face.commit_changes()
        for face in self.removed_faces:
            face.commit_changes()

    @classmethod
    def from_dict(cls, filename: str, data: Dict[str, Any]) -> "ImageRecord":
        raw_faces_data = data.get("faces", [])
        parsed_faces = [Face.from_dict(f) for f in raw_faces_data]
        
        raw_removed = data.get("removed_faces", [])
        parsed_removed = [Face.from_dict(f) for f in raw_removed]
        
        face_count = data.get("face_count", len(parsed_faces))
        
        instance = cls(
            filename=filename,
            faces=parsed_faces,
            location_cluster=data.get("location_cluster"),
            location_name=data.get("location_name"),
            original_shape=tuple(data.get("original_shape", [0, 0])),
            face_count=face_count,
            removed_faces=parsed_removed,
            image_type='portrait' if face_count == 1 else 'group'
        )
        instance.original_location_cluster = instance.location_cluster
        instance.original_location_name = instance.location_name
        instance.original_image_type = instance.image_type
        return instance

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "filename": self.filename,
            "face_count": len(self.faces),
            "original_shape": self.original_shape,
            "location_cluster": self.location_cluster,
            "location_name": self.location_name,
            "faces": [face.to_dict() for face in self.faces]
        }
        if self.removed_faces:
            res["removed_faces"] = [face.to_dict() for face in self.removed_faces]
        return res