# pysm_lib/models.py

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    ValidationInfo,
    ValidationError,
)
from typing import List, Dict, Optional, Any, Union, Literal
import uuid
import logging

from .locale_manager import LocaleManager

locale_manager = LocaleManager()
models_validator_logger = logging.getLogger("PyScriptManager.ModelsValidator")


# --- ИЗМЕНЕНИЕ: Добавлен тип "list" ---
ContextVariableType = Literal[
    "string",
    "string_multiline",
    "int",
    "float",
    "bool",
    "file_path",
    "dir_path",
    "choice",
    "date",
    "datetime",
    "password",
    "json",
    "list",
]


class ContextVariableModel(BaseModel):
    """Модель для одной переменной в контексте коллекции."""

    type: ContextVariableType = "string"
    # --- ИЗМЕНЕНИЕ: value теперь поддерживает List[str] ---
    value: Optional[Union[str, int, float, bool, dict, List[str]]] = None
    description: Optional[str] = Field(default=None)
    read_only: bool = Field(default=False)
    choices: Optional[List[str]] = Field(
        default=None, description="Список вариантов для типа 'choice'"
    )
    model_config = ConfigDict(validate_assignment=True)


class ScriptSetEntryValueEnabled(BaseModel):
    # --- ИЗМЕНЕНИЕ: value теперь поддерживает List[str] ---
    value: Optional[Union[str, int, float, bool, dict, List[str]]] = None
    enabled: bool = False
    model_config = ConfigDict(extra="forbid")


class ScriptArgMetaDetailModel(BaseModel):
    type: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    min_val: Optional[Union[int, float]] = Field(default=None, alias="min")
    max_val: Optional[Union[int, float]] = Field(default=None, alias="max")
    choices: Optional[List[Any]] = None
    filter: Optional[str] = None
    decimals: Optional[int] = None
    model_config = ConfigDict(populate_by_name=True)


class ScriptInfoModel(BaseModel):
    type: Literal["script"] = "script"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    folder_abs_path: str
    run_filename: Optional[str] = None
    run_file_abs_path: Optional[str] = None
    description: Optional[str] = None
    category: str = Field(
        default_factory=lambda: locale_manager.get(
            "models.script_info.default_category"
        )
    )
    author: str = Field(
        default_factory=lambda: locale_manager.get("models.script_info.default_author")
    )
    version: str = "1.0.0"
    command_line_args: Optional[Dict[str, Any]] = None
    command_line_args_meta: Optional[Dict[str, ScriptArgMetaDetailModel]] = None
    passport_valid: bool = False
    passport_error: Optional[str] = None
    is_raw: bool = Field(
        default=False,
        description="True, если скрипт обнаружен, но для него нет паспорта.",
    )
    specific_python_interpreter: Optional[str] = None
    script_specific_env_paths: Optional[List[str]] = None
    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "folder_abs_path", "run_file_abs_path", "specific_python_interpreter"
    )
    @classmethod
    def check_path_not_empty_if_not_none(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v

    @field_validator("script_specific_env_paths")
    @classmethod
    def check_paths_in_list_not_empty(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned_paths = [p for p in v if p and p.strip()]
        return cleaned_paths if cleaned_paths else None

    @field_validator("command_line_args_meta", mode="before")
    @classmethod
    def convert_meta_to_model(
        cls, v: Optional[Dict[str, Any]], info: ValidationInfo
    ) -> Optional[Dict[str, ScriptArgMetaDetailModel]]:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError(
                locale_manager.get(
                    "models.errors.must_be_dict", field="command_line_args_meta"
                )
            )
        if all(
            isinstance(meta_val, ScriptArgMetaDetailModel) for meta_val in v.values()
        ):
            return v
        processed_meta: Dict[str, ScriptArgMetaDetailModel] = {}
        errors_in_meta: Dict[str, Any] = {}
        for key, meta_data_or_model in v.items():
            if isinstance(meta_data_or_model, ScriptArgMetaDetailModel):
                processed_meta[key] = meta_data_or_model
            elif isinstance(meta_data_or_model, dict):
                try:
                    processed_meta[key] = ScriptArgMetaDetailModel(**meta_data_or_model)
                except ValidationError as e_meta:
                    errors_in_meta[key] = e_meta.errors(include_url=False)
            else:
                errors_in_meta[key] = locale_manager.get(
                    "models.errors.invalid_type_for_meta", type=type(meta_data_or_model)
                )
        if errors_in_meta:
            log_msg = locale_manager.get(
                "models.log_errors.validation_failed_meta", errors=errors_in_meta
            )
            models_validator_logger.error(log_msg)
            raise ValueError(log_msg)
        return processed_meta if processed_meta else None

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # 1. БЛОК: Новый валидатор модели для установки значения по умолчанию
    @model_validator(mode="after")
    def set_default_for_required_choices(self) -> "ScriptInfoModel":
        """
        Проверяет аргументы и устанавливает значение по умолчанию для
        обязательных полей типа 'choice', у которых оно не задано.
        Это гарантирует, что UI получит корректное начальное значение.
        """
        if self.command_line_args_meta:
            for arg_name, meta in self.command_line_args_meta.items():
                # Условие: аргумент обязательный, тип - choice,
                # нет значения по умолчанию, но есть список вариантов.
                if (
                    meta.required
                    and meta.type == "choice"
                    and meta.default is None
                    and meta.choices
                ):
                    # Назначаем первый вариант из списка как значение по умолчанию.
                    meta.default = meta.choices[0]
                    models_validator_logger.debug(
                        f"Для обязательного аргумента '{arg_name}' типа 'choice' "
                        f"установлено значение по умолчанию: '{meta.default}'"
                    )
        return self

    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


class CategoryNodeModel(BaseModel):
    type: Literal["category"] = "category"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    folder_abs_path: str
    description: Optional[str] = None
    icon_standard_pixmap_name: Optional[str] = None
    icon_file_abs_path: Optional[str] = None
    sort_order: Optional[int] = None
    children: List["ScanTreeNodeType"] = Field(default_factory=list)
    model_config = ConfigDict(validate_assignment=True)


ScanTreeNodeType = Union[CategoryNodeModel, ScriptInfoModel]
CategoryNodeModel.model_rebuild()


class ScriptRootModel(BaseModel):
    id: str = Field(default_factory=lambda: f"root_{uuid.uuid4().hex[:8]}")
    path: str


class ScriptSetEntryModel(BaseModel):
    id: str = Field(..., min_length=1)
    instance_id: str = Field(
        default_factory=lambda: f"instance_{uuid.uuid4().hex[:12]}"
    )
    name: Optional[str] = Field(
        default=None,
        description="Пользовательское имя для экземпляра скрипта в наборе.",
    )
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    description: Optional[str] = Field(
        default=None,
        description="Пользовательское описание для этого конкретного экземпляра скрипта.",
    )
    silent_mode: bool = Field(
        default=False,
        description="Если True, при запуске не выводится служебная информация (имя, параметры и т.д.).",
    )
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    command_line_args: Dict[str, ScriptSetEntryValueEnabled] = Field(
        default_factory=dict
    )
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @field_validator("command_line_args", mode="before")
    @classmethod
    def ensure_command_line_args_values_are_models(
        cls, v: Any, info: ValidationInfo
    ) -> Dict[str, ScriptSetEntryValueEnabled]:
        if not isinstance(v, dict):
            raise ValueError(
                locale_manager.get(
                    "models.errors.must_be_dict", field="command_line_args"
                )
            )
        if all(isinstance(val, ScriptSetEntryValueEnabled) for val in v.values()):
            return v
        validated_args: Dict[str, ScriptSetEntryValueEnabled] = {}
        errors_found: Dict[str, Any] = {}
        for key, value_data in v.items():
            if isinstance(value_data, ScriptSetEntryValueEnabled):
                validated_args[key] = value_data
            elif isinstance(value_data, dict):
                try:
                    validated_args[key] = ScriptSetEntryValueEnabled(**value_data)
                except ValidationError as e_entry_val:
                    errors_found[key] = e_entry_val.errors(include_url=False)
            else:
                errors_found[key] = locale_manager.get(
                    "models.errors.invalid_type_for_arg_value",
                    key=key,
                    type=type(value_data),
                )
        if errors_found:
            log_msg = locale_manager.get(
                "models.log_errors.validation_failed_entry",
                id=info.data.get("id", "N/A"),
                errors=errors_found,
            )
            models_validator_logger.error(log_msg)
        return v

    def create_copy(self) -> "ScriptSetEntryModel":
        """Создает глубокую копию экземпляра с новым уникальным ID."""
        new_model = self.model_copy(deep=True)
        new_model.instance_id = f"instance_{uuid.uuid4().hex[:12]}"
        return new_model


class ScriptSetNodeModel(BaseModel):
    type: Literal["script_set"] = "script_set"
    id: str = Field(default_factory=lambda: f"setnode_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    script_entries: List[ScriptSetEntryModel] = Field(default_factory=list)
    model_config = ConfigDict(validate_assignment=True)


class SetFolderNodeModel(BaseModel):
    type: Literal["set_folder"] = "set_folder"
    id: str = Field(default_factory=lambda: f"foldernode_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    children: List["SetHierarchyNodeType"] = Field(default_factory=list)
    model_config = ConfigDict(validate_assignment=True)


SetHierarchyNodeType = Union[SetFolderNodeModel, ScriptSetNodeModel]


class ScriptSetsCollectionModel(BaseModel):
    collection_name: str = Field(
        default_factory=lambda: locale_manager.get("models.collection.default_name")
    )
    description: Optional[str] = Field(
        default=None,
        description=locale_manager.get("models.collection.description"),
    )
    data_format_version: str = "1.1"

    # --- ИЗМЕНЕНИЕ: Добавлено новое поле ---
    execution_mode: str = Field(
        default="sequential_full",
        description="Режим запуска по умолчанию для коллекции: sequential_full, sequential_step, single_from_set",
    )

    script_roots: List[ScriptRootModel] = Field(
        default_factory=list,
        description=locale_manager.get("models.collection.script_roots_description"),
    )

    root_nodes: List[SetHierarchyNodeType] = Field(default_factory=list)

    context_data: Dict[str, ContextVariableModel] = Field(
        default_factory=dict, exclude=True
    )

    model_config = ConfigDict(validate_assignment=True)


SetFolderNodeModel.model_rebuild()
ScriptSetsCollectionModel.model_rebuild()
