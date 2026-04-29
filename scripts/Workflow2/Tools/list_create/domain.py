"""
domain.py
=========
Модуль содержит основные сущности предметной области (Domain Entities)
и конфигурационные классы.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------
CHILDREN_LIST_FILENAME = "children.txt"
DEFAULT_AUTOSAVE_FORMATS = ["html", "txt"]


# ----------------------------------------------------------------------
# Сущности (Entities)
# ----------------------------------------------------------------------
@dataclass
class ExtraService:
    """Представляет дополнительную услугу для конкретного ученика."""
    name: str
    qty: int = 1
    cost: int = 0
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtraService':
        return cls(**data)


@dataclass
class Student:
    """
    Основная сущность ученика.
    """
    # Порядок и идентификация
    surname: str = ""
    name: str = ""
    patronymic: str = ""         # НОВОЕ: Отчество
    rank: str = "ученик"         # НОВОЕ: Ранг (ученик, учитель, директор и т.д.)
    shoot_order: Optional[int] = None
    alpha_order: int = 0

    # Визуальное оформление
    color1: Optional[str] = None    
    color1_fg: Optional[str] = None 
    color2: Optional[str] = None    
    color2_fg: Optional[str] = None 

    # Бизнес-логика (Услуги)
    service_type: str = ""
    service_cost: int = 0
    extra_services: List[ExtraService] = field(default_factory=list)
    
    # Дополнительная информация (Цитаты, Хобби и т.д.)
    info: Dict[str, str] = field(default_factory=dict)

    @property
    def total_cost(self) -> int:
        extras_sum = sum(ex.cost * ex.qty for ex in self.extra_services)
        return self.service_cost + extras_sum

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['extra_services'] = [ex.to_dict() for ex in self.extra_services]
        # info сериализуется автоматически, так как это dict
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Student':
        d = data.copy()
        extras_raw = d.pop('extra_services', [])
        extras = [ExtraService.from_dict(e) for e in extras_raw]
        
        if d.get('shoot_order') == "":
            d['shoot_order'] = None
            
        # Защита от лишних полей
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in d.items() if k in valid_keys}
        
        return cls(extra_services=extras, **filtered_data)


# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
@dataclass
class AppConfig:
    """
    Конфигурация приложения.
    """
    wf_dest_dir: Optional[str] = None
    wf_output_txt_file: Optional[str] = None
    wf_autosave_formats: List[str] = field(default_factory=lambda: DEFAULT_AUTOSAVE_FORMATS)   
    wf_default_info_fields: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args_namespace: Any) -> 'AppConfig':
        """Создает конфиг из аргументов argparse (Namespace)."""
        return cls(
            wf_dest_dir=getattr(args_namespace, 'wf_dest_dir', None),
            wf_output_txt_file=getattr(args_namespace, 'wf_output_txt_file', None),
            wf_autosave_formats=getattr(args_namespace, 'wf_autosave_formats', DEFAULT_AUTOSAVE_FORMATS),
            wf_default_info_fields=getattr(args_namespace, 'wf_default_info_fields', [])
        )