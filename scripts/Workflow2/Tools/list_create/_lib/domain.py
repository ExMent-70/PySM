"""
domain.py
=========
Содержит основные сущности предметной области и конфигурационные классы.
"""

from dataclasses import dataclass, field, asdict
import re
import secrets
from typing import List, Optional, Any, Dict

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------
DEFAULT_AUTOSAVE_FORMATS = ["html", "csv"]
LIST_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
LIST_ID_PATTERN = re.compile(r"^[A-HJ-NP-Z2-9]{4}$")
STUDENT_ID_PATTERN = re.compile(r"^(?P<list_id>[A-HJ-NP-Z2-9]{4})-S(?P<number>\d{3})$")
MAX_STUDENT_NUMBER = 999


def generate_list_id() -> str:
    """Создаёт короткий идентификатор нового файла списка."""

    return "".join(secrets.choice(LIST_ID_ALPHABET) for _ in range(4))


def validate_list_id(list_id: str) -> str:
    """Проверяет и возвращает нормализованный идентификатор списка."""

    normalized = str(list_id).strip().upper()
    if not LIST_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "list_id должен состоять из четырёх заглавных латинских букв/цифр "
            "без I, O, 0 и 1."
        )
    return normalized


def parse_student_id(student_id: str, expected_list_id: str | None = None) -> int:
    """Проверяет полный ID и возвращает числовую часть записи."""

    normalized = str(student_id).strip().upper()
    match = STUDENT_ID_PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError("student_id должен иметь формат A7K3-S001.")

    list_id = match.group("list_id")
    if expected_list_id is not None and list_id != validate_list_id(expected_list_id):
        raise ValueError(
            f"student_id {normalized} относится к списку {list_id}, "
            f"а открыт список {expected_list_id}."
        )

    number = int(match.group("number"))
    if not 1 <= number <= MAX_STUDENT_NUMBER:
        raise ValueError("Номер записи student_id должен быть в диапазоне S001–S999.")
    return number


@dataclass
class StudentIdAllocator:
    """Хранит идентичность списка и выдаёт неповторяющиеся ID записей."""

    list_id: str = field(default_factory=generate_list_id)
    next_student_number: int = 1

    def __post_init__(self) -> None:
        self.list_id = validate_list_id(self.list_id)
        if not 1 <= self.next_student_number <= MAX_STUDENT_NUMBER + 1:
            raise ValueError("next_student_number должен быть в диапазоне 1–1000.")

    def allocate(self) -> str:
        """Возвращает следующий ID и навсегда сдвигает счётчик вперёд."""

        if self.next_student_number > MAX_STUDENT_NUMBER:
            raise ValueError("Исчерпан диапазон student_id: достигнут номер S999.")
        student_id = f"{self.list_id}-S{self.next_student_number:03d}"
        self.next_student_number += 1
        return student_id

    def assign_missing(self, students: List['Student']) -> None:
        """Назначает ID новым объектам, созданным парсерами или интерфейсом."""

        for student in students:
            if not student.student_id:
                student.student_id = self.allocate()

    def validate_students(self, students: List['Student']) -> None:
        """Проверяет формат, принадлежность и уникальность ID списка."""

        seen: Dict[str, Student] = {}
        highest_number = 0
        for student in students:
            label = f"{student.surname} {student.name}".strip() or "без ФИО"
            if not student.student_id:
                raise ValueError(f"У записи «{label}» отсутствует student_id.")

            normalized = student.student_id.strip().upper()
            try:
                number = parse_student_id(normalized, self.list_id)
            except ValueError as exc:
                raise ValueError(f"Некорректный ID записи «{label}»: {exc}") from exc

            if normalized in seen:
                other = seen[normalized]
                other_label = f"{other.surname} {other.name}".strip() or "без ФИО"
                raise ValueError(
                    f"student_id {normalized} повторяется у записей "
                    f"«{other_label}» и «{label}»."
                )
            student.student_id = normalized
            seen[normalized] = student
            highest_number = max(highest_number, number)

        if self.next_student_number <= highest_number:
            raise ValueError(
                "next_student_number должен быть больше всех ранее выданных "
                "номеров student_id."
            )


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
    student_id: str = ""
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
        student = cls(extra_services=extras, **filtered_data)
        if not student.student_id:
            raise ValueError("Запись ученика не содержит обязательный student_id.")
        parse_student_id(student.student_id)
        student.student_id = student.student_id.strip().upper()
        return student


# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
@dataclass
class AppConfig:
    """
    Конфигурация приложения.
    """
    wf_dest_dir: Optional[str] = None
    wf_autosave_formats: List[str] = field(default_factory=lambda: DEFAULT_AUTOSAVE_FORMATS)   
    wf_default_info_fields: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args_namespace: Any) -> 'AppConfig':
        """Создает конфиг из аргументов argparse (Namespace)."""
        return cls(
            wf_dest_dir=getattr(args_namespace, 'wf_dest_dir', None),
            wf_autosave_formats=getattr(args_namespace, 'wf_autosave_formats', DEFAULT_AUTOSAVE_FORMATS),
            wf_default_info_fields=getattr(args_namespace, 'wf_default_info_fields', [])
        )
