"""
Модуль моделей данных для редактора кластеров.
Реализация Способа Б: Immutable Index (face_index).
"""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


def _optional_integer(value: Any, field_name: str) -> Optional[int]:
    """Normalize a JSON integer without silently truncating floats or bools."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Поле {field_name} должно быть целым числом или null.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value) or not value.is_integer():
            raise ValueError(f"Поле {field_name} должно быть целым числом или null.")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Поле {field_name} должно быть целым числом или null."
            ) from exc
    raise ValueError(f"Поле {field_name} должно быть целым числом или null.")


def _validate_filename(filename: str) -> str:
    """Accept only flat file names stored directly below a session JPG folder."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Имя файла должно быть непустой строкой.")
    path = Path(filename)
    if path.is_absolute() or path.name != filename or filename in {".", ".."}:
        raise ValueError(f"Недопустимое имя файла: {filename!r}")
    return filename

@dataclass
class Face:
    """
    Представляет данные одного лица на фотографии.
    """
    bbox: List[float]
    cluster_label: Optional[int] = None
    student_id: Optional[str] = None
    
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
    original_student_id: Optional[str] = None
    original_quality_status: str = "ok"
    original_temp_label: Optional[int] = None

    original_matched_label: Optional[int] = None
    original_gender_faceonnx: Optional[str] = None

    # UI поля
    filename: str = ""
    effective_name: str = ""
    embedding_key: Optional[str] = None 

    @property
    def is_changed(self) -> bool:
        # Проверка стандартных полей
        if (self.cluster_label != self.original_cluster_label or
            self.student_id != self.original_student_id or
            self.quality_status != self.original_quality_status or
            self.temp_cluster_label != self.original_temp_label):
            return True
            
        # --- ПРОВЕРКА: Matches ---
        # Проверяем, изменился ли лейбл сопоставления в extra_data
        current_match = self.extra_data.get('matched_portrait_cluster_label')
        if current_match != self.original_matched_label:
            return True
        # --- ПРОВЕРКА ИЗМЕНЕНИЯ ПОЛА ---
        if self.extra_data.get('gender_faceonnx') != self.original_gender_faceonnx:
            return True
            
        return False

    @property
    def is_trash(self) -> bool:
        return self.quality_status in ("trash", "technical_trash")

    def commit_changes(self):
        self.original_cluster_label = self.cluster_label
        self.original_student_id = self.student_id
        self.original_quality_status = self.quality_status
        self.original_temp_label = self.temp_cluster_label
        # Фиксируем состояние матчинга
        self.original_matched_label = self.extra_data.get('matched_portrait_cluster_label')
        self.original_gender_faceonnx = self.extra_data.get('gender_faceonnx')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Face":
        if not isinstance(data, dict):
            raise ValueError("Запись лица должна быть JSON-объектом.")
        bbox = data.get("bbox")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                for value in bbox
            )
        ):
            raise ValueError(f"Некорректный bbox лица: {bbox!r}")

        student_id = data.get("student_id")
        if student_id is not None and not isinstance(student_id, str):
            raise ValueError("student_id должен быть строкой или null.")
        quality_status = data.get("quality_status", "ok")
        if not isinstance(quality_status, str):
            raise ValueError("quality_status должен быть строкой.")
        temp_child_name = data.get("temp_child_name")
        if temp_child_name is not None and not isinstance(temp_child_name, str):
            raise ValueError("temp_child_name должен быть строкой или null.")

        known_fields = {'bbox', 'cluster_label', 'student_id', 'face_index',
                        'quality_status', 'temp_cluster_label', 'temp_child_name'}
        
        kwargs = {k: data[k] for k in known_fields if k in data}
        kwargs["bbox"] = [float(value) for value in bbox]
        legacy_fields = {'child_name', 'matched_child_name'}
        extra_data = {
            k: v for k, v in data.items()
            if k not in known_fields and k not in legacy_fields
        }
        
        for field_name in ("cluster_label", "temp_cluster_label", "face_index"):
            kwargs[field_name] = _optional_integer(
                kwargs.get(field_name),
                field_name,
            )
        if kwargs.get("face_index") is not None and kwargs["face_index"] < 0:
            raise ValueError("face_index не может быть отрицательным.")

        instance = cls(**kwargs)
        instance.extra_data = extra_data
        
        instance.original_cluster_label = instance.cluster_label
        instance.original_student_id = instance.student_id
        instance.original_quality_status = instance.quality_status
        instance.original_temp_label = instance.temp_cluster_label
        
        # Инициализируем исходное состояние матчинга из extra_data
        instance.original_matched_label = extra_data.get('matched_portrait_cluster_label')
        instance.original_gender_faceonnx = extra_data.get('gender_faceonnx') 
        
        return instance

    def to_dict(self) -> Dict[str, Any]:
        data = self.extra_data.copy()
        data.pop('child_name', None)
        data.pop('matched_child_name', None)
        data.update({
            'bbox': self.bbox,
            'face_index': self.face_index,
            'cluster_label': self.cluster_label,
            'student_id': self.student_id,
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
        filename = _validate_filename(filename)
        if not isinstance(data, dict):
            raise ValueError(f"{filename}: запись изображения должна быть объектом.")
        raw_faces_data = data.get("faces", [])
        if not isinstance(raw_faces_data, list):
            raise ValueError(f"{filename}: поле faces должно быть массивом.")
        parsed_faces = [Face.from_dict(f) for f in raw_faces_data]
        
        raw_removed = data.get("removed_faces", [])
        if not isinstance(raw_removed, list):
            raise ValueError(f"{filename}: поле removed_faces должно быть массивом.")
        parsed_removed = [Face.from_dict(f) for f in raw_removed]

        declared_face_count = data.get("face_count", len(parsed_faces))
        if (
            isinstance(declared_face_count, bool)
            or not isinstance(declared_face_count, int)
            or declared_face_count < 0
        ):
            raise ValueError(f"{filename}: face_count должен быть неотрицательным целым.")
        if declared_face_count != len(parsed_faces):
            raise ValueError(
                f"{filename}: face_count={declared_face_count}, "
                f"но faces содержит {len(parsed_faces)} записей."
            )
        face_count = len(parsed_faces)
        raw_shape = data.get("original_shape")
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) < 2:
            raise ValueError(f"{filename}: некорректный original_shape {raw_shape!r}.")
        try:
            original_height = _optional_integer(raw_shape[0], "original_shape[0]")
            original_width = _optional_integer(raw_shape[1], "original_shape[1]")
        except ValueError as exc:
            raise ValueError(
                f"{filename}: некорректный original_shape {raw_shape!r}."
            ) from exc
        if not original_height or not original_width:
            raise ValueError(f"{filename}: некорректный original_shape {raw_shape!r}.")

        location_cluster = _optional_integer(
            data.get("location_cluster"),
            "location_cluster",
        )
        location_name = data.get("location_name")
        if location_name is not None and not isinstance(location_name, str):
            raise ValueError(f"{filename}: location_name должен быть строкой или null.")
        
        instance = cls(
            filename=filename,
            faces=parsed_faces,
            location_cluster=location_cluster,
            location_name=location_name,
            original_shape=(original_height, original_width),
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
