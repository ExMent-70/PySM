# _rmbgtools/core/config_manager.py

import argparse
from pydantic import BaseModel, model_validator
from typing import Type, get_args, get_origin, Literal, Optional, Union
import inspect

from ..config import load_config, AppConfig
from .. import logger

class ConfigManager:
    """
    Управляет слиянием конфигураций из TOML-файла и аргументов CLI,
    а также автоматически генерирует аргументы для argparse.
    """
    def __init__(self, command: str, config_model: Type[BaseModel]):
        """
        Args:
            command (str): Имя команды ('global', 'remove', 'segment'), соответствующее секции в AppConfig.
            config_model (Type[BaseModel]): Pydantic-модель для этой команды.
        """
        self.command = command
        self.config_model = config_model

    def _add_args_for_model(self, parser_group, model: Type[BaseModel], prefix: str = ""):
        """Рекурсивно добавляет аргументы для Pydantic-модели и ее вложенных моделей."""
        for field_name, field in model.model_fields.items():
            field_type = field.annotation
            
            if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                new_prefix = f"{prefix}{field_name}_" if prefix else f"{field_name}_"
                self._add_args_for_model(parser_group, field_type, new_prefix)
                continue

            arg_name = f"--{prefix}{field_name}".replace('_', '-')
            
            kwargs = { "help": field.description or f"Override '{field_name}' from config." }
            
            origin_type = get_origin(field_type)
            if origin_type is Union: field_type = get_args(field_type)[0]
            origin_type = get_origin(field_type)

            # --- ИЗМЕНЕНИЕ: Добавляем новую логику для извлечения choices ---
            if origin_type is Literal:
                kwargs["choices"] = get_args(field_type)
                kwargs["type"] = type(kwargs["choices"][0])
            # Проверяем, есть ли наши кастомные `choices` в метаданных поля
            elif field.json_schema_extra and 'choices' in field.json_schema_extra:
                kwargs["choices"] = field.json_schema_extra['choices']
                kwargs["type"] = str
            elif field_type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            else:
                kwargs["type"] = field_type

            parser_group.add_argument(arg_name, **kwargs)

    # --- ИСПРАВЛЕННЫЙ МЕТОД ---
    def add_cli_arguments(self, parser: argparse.ArgumentParser, group_title: str):
        """
        Добавляет все необходимые аргументы CLI в парсер, создавая для них группу.

        Args:
            parser (argparse.ArgumentParser): Основной парсер.
            group_title (str): Заголовок для группы аргументов в справке (--help).
        """
        group = parser.add_argument_group(group_title)
        self._add_args_for_model(group, self.config_model)

    def merge(self, cli_args: argparse.Namespace, global_config: Optional[BaseModel] = None) -> BaseModel:
        """
        Загружает TOML, применяет переопределения из CLI и возвращает финальный конфиг.
        """
        app_config = load_config(cli_args.config)
        
        if self.command == 'global':
            base_config = app_config.global_settings
        else:
            base_config = getattr(app_config, self.command)
        
        cli_dict = vars(cli_args)
        self._apply_overrides(base_config, cli_dict)
        
        # Если это не глобальный конфиг, можно унаследовать значения из него
        if global_config:
            for field_name, field in global_config.model_fields.items():
                # Если в cli_args нет переопределения для глобального параметра,
                # и он есть в нашем конфиге, устанавливаем его.
                # Это полезно для таких полей как `device`, `model_dir`
                if cli_dict.get(field_name) is None and hasattr(base_config, field_name):
                     setattr(base_config, field_name, getattr(global_config, field_name))

        return base_config

    def _apply_overrides(self, config_obj: BaseModel, cli_dict: dict, prefix: str = ""):
        """Рекурсивно применяет значения из CLI к объекту Pydantic-конфигурации."""
        for field_name, field in config_obj.model_fields.items():
            field_type = field.annotation
            
            if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                new_prefix = f"{prefix}{field_name}_" if prefix else f"{field_name}_"
                self._apply_overrides(getattr(config_obj, field_name), cli_dict, new_prefix)
            else:
                cli_key = f"{prefix}{field_name}"
                if cli_key in cli_dict and cli_dict[cli_key] is not None:
                    setattr(config_obj, field_name, cli_dict[cli_key])