# pysm_lib/set_manager.py

import pathlib
import json
import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Tuple, Union


from .models import (
    ScriptSetsCollectionModel,
    SetFolderNodeModel,
    ScriptSetNodeModel,
    SetHierarchyNodeType,
    ScriptSetEntryModel,
    ScriptRootModel,
    ContextVariableModel,
    FavoriteScriptModel,
)

from pydantic import ValidationError

from .app_constants import APPLICATION_ROOT_DIR
from .path_utils import to_relative_if_possible, resolve_path
from .locale_manager import LocaleManager
from .pysm_context import pysm_context


locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")

SCRIPT_SETS_DIR_NAME = "script_collections"
SCRIPTS_DIR_NAME = "scripts"
SCRIPT_SETS_ROOT_DIR_DEFAULT = APPLICATION_ROOT_DIR / SCRIPT_SETS_DIR_NAME


class SetManager:
    def __init__(
        self, default_sets_root_dir: pathlib.Path = SCRIPT_SETS_ROOT_DIR_DEFAULT
    ):
        self.default_sets_root_dir: pathlib.Path = default_sets_root_dir
        logger.info(
            locale_manager.get(
                "set_manager.log_info.init", path=self.default_sets_root_dir
            )
        )
        try:
            self.default_sets_root_dir.mkdir(parents=True, exist_ok=True)
            default_scripts_dir = APPLICATION_ROOT_DIR / SCRIPTS_DIR_NAME
            default_scripts_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.critical(
                locale_manager.get(
                    "set_manager.log_critical.create_default_dirs_failed", error=e
                ),
                exc_info=True,
            )

        self._is_dirty: bool = False
        self._nodes_by_id_cache: Dict[str, SetHierarchyNodeType] = {}

        self.current_collection_model: ScriptSetsCollectionModel = (
            self.create_new_empty_collection()
        )
        self.current_collection_file_path: Optional[pathlib.Path] = None

    def _get_context_file_path(
        self, collection_file_path: pathlib.Path
    ) -> pathlib.Path:
        """Формирует путь к файлу контекста на основе пути к файлу коллекции."""
        return collection_file_path.with_suffix(".context.json")

    def _load_context_data_from_path(
        self, context_file_path: pathlib.Path
    ) -> Dict[str, ContextVariableModel]:
        """
        Читает и парсит файл контекста, возвращая словарь с моделями.
        Пропускает системные переменные ('pysm_*').
        """
        if not context_file_path.is_file():
            return {}

        try:
            logger.info(
                locale_manager.get(
                    "set_manager.log_info.loading_context_file",
                    path=context_file_path,
                )
            )
            raw_context_data = None
            for attempt in range(3):
                try:
                    with open(context_file_path, "r", encoding="utf-8") as f:
                        raw_context_data = json.load(f)
                    break
                except json.JSONDecodeError:
                    if attempt == 2:
                        raise
                    time.sleep(0.05)

            if isinstance(raw_context_data, dict):
                # Валидируем только пользовательские переменные
                return {
                    k: ContextVariableModel(**v)
                    for k, v in raw_context_data.items()
                    if not k.startswith("pysm_")
                }
            else:
                logger.error(
                    locale_manager.get(
                        "set_manager.log_error.context_file_invalid_format",
                        path=context_file_path,
                    )
                )
                return {}
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.context_reload_failed",
                    path=context_file_path,
                    error=e,
                ),
                exc_info=True,
            )
            return {}

    def _rebuild_nodes_cache(self):
        self._nodes_by_id_cache.clear()

        def _recursive_walk(nodes: List[SetHierarchyNodeType]):
            for node in nodes:
                self._nodes_by_id_cache[node.id] = node
                if isinstance(node, SetFolderNodeModel) and node.children:
                    _recursive_walk(node.children)

        _recursive_walk(self.current_collection_model.root_nodes)
        logger.debug(
            locale_manager.get(
                "set_manager.log_debug.cache_rebuilt",
                count=len(self._nodes_by_id_cache),
            )
        )

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    def _set_dirty(self, state: bool):
        if self._is_dirty != state:
            self._is_dirty = state
            logger.debug(
                locale_manager.get("set_manager.log_debug.dirty_flag_set", state=state)
            )

    def create_new_empty_collection(self) -> ScriptSetsCollectionModel:
        logger.info(locale_manager.get("set_manager.log_info.creating_new_collection"))
        self.current_collection_model = ScriptSetsCollectionModel()
        main_root_folder = SetFolderNodeModel(
            name=locale_manager.get("set_manager.default_folder_name")
        )
        self.current_collection_model.root_nodes = [main_root_folder]

        default_scripts_path = APPLICATION_ROOT_DIR / SCRIPTS_DIR_NAME
        default_scripts_path_str = str(default_scripts_path.resolve())
        if not any(
            r.path == default_scripts_path_str
            for r in self.current_collection_model.script_roots
        ):
            self.current_collection_model.script_roots.append(
                ScriptRootModel(path=default_scripts_path_str)
            )

        self.current_collection_file_path = None
        self._rebuild_nodes_cache()
        self._set_dirty(True)
        return self.current_collection_model
        
    # --- НАЧАЛО ИЗМЕНЕНИЙ (Добавлен новый метод) ---
    def create_collection_from_current(self) -> ScriptSetsCollectionModel:
        logger.info("SetManager: Создание рабочего процесса из шаблона.")
        
        # Делаем глубокую копию текущей модели (все настройки, переменные и структура сохранятся)
        new_model = self.current_collection_model.model_copy(deep=True)
        
        # Сбрасываем имя коллекции на дефолтное "Новый рабочий процесс"
        new_model.collection_name = locale_manager.get("set_manager.new_collection_name")        
        
        # Устанавливаем новую модель, сбрасываем путь к файлу и помечаем как "измененную"
        self.current_collection_model = new_model
        self.current_collection_file_path = None
        self._rebuild_nodes_cache()
        self._set_dirty(True)
        
        return self.current_collection_model
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---        

    def load_collection_from_file(self, file_path: pathlib.Path) -> bool:
        logger.info(
            locale_manager.get("set_manager.log_info.loading_from_file", path=file_path)
        )
        if not file_path.is_file():
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.file_not_found", path=file_path
                )
            )
            self.create_new_empty_collection()
            self.current_collection_model.collection_name = file_path.stem
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_collection = ScriptSetsCollectionModel(**data)

            context_file_path = self._get_context_file_path(file_path)
            loaded_collection.context_data = self._load_context_data_from_path(
                context_file_path
            )



            collection_base_dir = file_path.parent
            for root in loaded_collection.script_roots:
                root.path = resolve_path(root.path, base_dir=collection_base_dir)

            was_adapted = False
            if not loaded_collection.root_nodes or not isinstance(
                loaded_collection.root_nodes[0], SetFolderNodeModel
            ):
                logger.warning(
                    locale_manager.get(
                        "set_manager.log_warning.adapting_structure", path=file_path
                    )
                )
                new_main_root = SetFolderNodeModel(
                    name=locale_manager.get("set_manager.default_folder_name")
                )
                new_main_root.children = loaded_collection.root_nodes
                loaded_collection.root_nodes = [new_main_root]
                was_adapted = True

            self.current_collection_model = loaded_collection
            self.current_collection_file_path = file_path.resolve()
            self._rebuild_nodes_cache()
            self._set_dirty(was_adapted)
            logger.info(
                locale_manager.get(
                    "set_manager.log_info.collection_loaded",
                    name=loaded_collection.collection_name,
                    dirty=self.is_dirty,
                )
            )
            return True
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.load_failed", path=file_path, error=e
                ),
                exc_info=True,
            )
            self.create_new_empty_collection()
            self.current_collection_model.collection_name = file_path.stem
            return False

    def _atomic_write_json(self, target_path: pathlib.Path, data_to_dump: dict):
        """
        Атомарно записывает JSON-совместимый словарь в файл.
        Сначала пишет во временный файл, затем переименовывает.
        """
        # Создаем временный файл в той же директории, чтобы 'os.replace' был атомарным
        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        try:
            # Записываем данные во временный файл
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data_to_dump, f, indent=2, ensure_ascii=False)
            # Если запись прошла успешно, атомарно заменяем основной файл
            os.replace(temp_path, target_path)
        except Exception as e:
            # В случае ошибки удаляем временный файл, если он был создан
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    # Игнорируем ошибки при удалении временного файла,
                    # так как исходное исключение важнее
                    pass
            # Пробрасываем исходное исключение наверх для обработки
            raise e


    def save_collection_to_file(self, file_path: Optional[pathlib.Path] = None) -> bool:
        target_path = file_path or self.current_collection_file_path
        if not target_path:
            # Невозможно сохранить, если путь не указан и не был определен ранее
            return False

        # Создаем глубокую копию модели для безопасного сохранения
        collection_copy_for_save = self.current_collection_model.model_copy(deep=True)

        # Обновляем имя коллекции, если оно дефолтное или файл сохраняется под новым именем
        if collection_copy_for_save.collection_name == locale_manager.get(
            "set_manager.new_collection_name"
        ) or (
            self.current_collection_file_path
            and target_path.stem != self.current_collection_file_path.stem
        ):
            collection_copy_for_save.collection_name = target_path.stem
            self.current_collection_model.collection_name = target_path.stem

        # Преобразуем пути к корням скриптов в относительные
        collection_base_dir = target_path.parent
        for root in collection_copy_for_save.script_roots:
            root.path = to_relative_if_possible(root.path, base_dir=collection_base_dir)

        try:
            # Убеждаемся, что директория для сохранения существует
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            # 1. Атомарно сохраняем основной файл коллекции
            self._atomic_write_json(
                target_path, collection_copy_for_save.model_dump(mode="json")
            )

            # 2. Готовим данные контекста и атомарно сохраняем/удаляем файл контекста
            context_file_path = self._get_context_file_path(target_path)
            context_data_to_save = {
                k: v.model_dump(mode="json")
                for k, v in self.current_collection_model.context_data.items()
            }

            if context_data_to_save:
                # Если в контексте есть данные, атомарно записываем их
                logger.info(
                    locale_manager.get(
                        "set_manager.log_info.saving_context_file",
                        path=context_file_path,
                    )
                )
                self._atomic_write_json(context_file_path, context_data_to_save)
            elif context_file_path.exists():
                # Если данных нет, а файл существует - удаляем его
                logger.info(
                    locale_manager.get(
                        "set_manager.log_info.deleting_empty_context_file",
                        path=context_file_path,
                    )
                )
                context_file_path.unlink()
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

            # Если все прошло успешно, обновляем внутренние состояния
            self.current_collection_file_path = target_path.resolve()
            self._set_dirty(False)
            return True

        except Exception as e:
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.save_failed", path=target_path, error=e
                ),
                exc_info=True,
            )
            return False

    def update_collection_context(self, new_context: Dict[str, ContextVariableModel]):
        if self.current_collection_model.context_data != new_context:
            self.current_collection_model.context_data = new_context
            self._set_dirty(True)

    def replace_context_from_raw_snapshot(self, raw_context_data: dict) -> bool:
        if not isinstance(raw_context_data, dict):
            return False
        try:
            context_data = {
                k: ContextVariableModel(**v)
                for k, v in raw_context_data.items()
                if not k.startswith("pysm_")
            }
        except (TypeError, ValidationError) as e:
            logger.error(f"Failed to apply runtime context snapshot: {e}", exc_info=True)
            return False
        self.current_collection_model.context_data = context_data
        self._set_dirty(True)
        return True

    def save_current_context_to_file(
        self, context_file_path: Optional[pathlib.Path] = None
    ) -> bool:
        target_path = context_file_path
        if target_path is None:
            if not self.current_collection_file_path:
                return False
            target_path = self._get_context_file_path(self.current_collection_file_path)

        context_data_to_save = {
            k: v.model_dump(mode="json")
            for k, v in self.current_collection_model.context_data.items()
        }
        try:
            if context_data_to_save:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write_json(target_path, context_data_to_save)
            elif target_path.exists():
                target_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to save context checkpoint '{target_path}': {e}", exc_info=True)
            return False

    def update_collection_properties(
        self, name: Optional[str] = None, description: Optional[str] = None
    ):
        if name is not None and self.current_collection_model.collection_name != name:
            self.current_collection_model.collection_name = name
            self._set_dirty(True)
        if (
            description is not None
            and self.current_collection_model.description != description
        ):
            self.current_collection_model.description = description
            self._set_dirty(True)

    def add_script_root(self, path: str) -> Optional[ScriptRootModel]:
        if any(
            root.path == path for root in self.current_collection_model.script_roots
        ):
            return None
        new_root = ScriptRootModel(path=path)
        self.current_collection_model.script_roots.append(new_root)
        self._set_dirty(True)
        return new_root

    def remove_script_root(self, root_id: str) -> bool:
        initial_len = len(self.current_collection_model.script_roots)
        self.current_collection_model.script_roots = [
            r for r in self.current_collection_model.script_roots if r.id != root_id
        ]
        if len(self.current_collection_model.script_roots) < initial_len:
            self._set_dirty(True)
            return True
        return False

    def update_script_root(self, root_id: str, new_path: str) -> bool:
        for root in self.current_collection_model.script_roots:
            if root.id == root_id:
                if root.path != new_path:
                    root.path = new_path
                    self._set_dirty(True)
                return True
        return False

    def get_main_root_folder(self) -> Optional[SetFolderNodeModel]:
        if self.current_collection_model.root_nodes and isinstance(
            self.current_collection_model.root_nodes[0], SetFolderNodeModel
        ):
            return self.current_collection_model.root_nodes[0]
        return None

    def _find_node_recursive(
        self, node_id: str, nodes_list: List[SetHierarchyNodeType]
    ) -> Optional[Tuple[SetHierarchyNodeType, List[SetHierarchyNodeType], int]]:
        for i, node in enumerate(nodes_list):
            if node.id == node_id:
                return node, nodes_list, i
            if isinstance(node, SetFolderNodeModel) and node.children:
                found = self._find_node_recursive(node_id, node.children)
                if found:
                    return found
        return None

    def get_node_by_id(self, node_id: str) -> Optional[SetHierarchyNodeType]:
        return self._nodes_by_id_cache.get(node_id)

    def get_set_node_by_id(self, set_node_id: str) -> Optional[ScriptSetNodeModel]:
        node = self.get_node_by_id(set_node_id)
        return node if isinstance(node, ScriptSetNodeModel) else None

    def get_folder_node_by_id(
        self, folder_node_id: str
    ) -> Optional[SetFolderNodeModel]:
        node = self.get_node_by_id(folder_node_id)
        return node if isinstance(node, SetFolderNodeModel) else None

    def get_all_nodes_for_display(self) -> List[SetHierarchyNodeType]:
        return self.current_collection_model.root_nodes

    def add_folder_node(
        self, name: str, parent_folder_id: Optional[str] = None, **kwargs
    ) -> Optional[SetFolderNodeModel]:
        parent_folder = (
            self.get_node_by_id(parent_folder_id)
            if parent_folder_id
            else self.get_main_root_folder()
        )
        if not isinstance(parent_folder, SetFolderNodeModel):
            return None
        new_folder = SetFolderNodeModel(name=name, **kwargs)
        parent_folder.children.append(new_folder)
        self._rebuild_nodes_cache()
        self._set_dirty(True)
        return new_folder

    def add_set_node(
        self, name: str, parent_folder_id: Optional[str] = None, **kwargs
    ) -> Optional[ScriptSetNodeModel]:
        parent_folder = (
            self.get_node_by_id(parent_folder_id)
            if parent_folder_id
            else self.get_main_root_folder()
        )
        if not isinstance(parent_folder, SetFolderNodeModel):
            return None
        new_set = ScriptSetNodeModel(name=name, **kwargs)
        parent_folder.children.append(new_set)
        self._rebuild_nodes_cache()
        self._set_dirty(True)
        return new_set

    def delete_node(self, node_id: str) -> bool:
        main_root = self.get_main_root_folder()
        if not main_root or node_id == main_root.id:
            return False
        find_result = self._find_node_recursive(node_id, main_root.children)
        if not find_result:
            return False
        _, parent_list, index = find_result
        parent_list.pop(index)
        self._rebuild_nodes_cache()
        self._cleanup_orphaned_favorites()
        self._set_dirty(True)
        return True

    def update_node_properties(
        self,
        node_id: str,
        new_name: Optional[str] = None,
        new_description: Optional[str] = None,
    ) -> bool:
        node_to_update = self.get_node_by_id(node_id)
        if not node_to_update:
            return False
        updated = False
        if (
            new_name is not None
            and node_to_update.name != new_name
            and new_name.strip()
        ):
            node_to_update.name = new_name.strip()
            updated = True
        if (
            new_description is not None
            and node_to_update.description != new_description
        ):
            node_to_update.description = new_description
            updated = True
        if updated:
            self._set_dirty(True)
        return updated

    def move_node(
        self, node_id_to_move: str, new_parent_id: Optional[str], target_index: int = -1
    ) -> bool:
        main_root = self.get_main_root_folder()
        if not main_root or node_id_to_move == main_root.id:
            return False
        find_result = self._find_node_recursive(node_id_to_move, main_root.children)
        if not find_result:
            return False
        node_to_move, original_parent_list, original_index = find_result
        original_parent_list.pop(original_index)
        new_parent_folder = (
            self.get_node_by_id(new_parent_id) if new_parent_id else main_root
        )
        if not isinstance(new_parent_folder, SetFolderNodeModel):
            original_parent_list.insert(original_index, node_to_move)
            return False
        if isinstance(node_to_move, SetFolderNodeModel):
            temp_parent = new_parent_folder
            while temp_parent:
                if temp_parent.id == node_to_move.id:
                    original_parent_list.insert(original_index, node_to_move)
                    return False
                parent_of_temp_found = False
                for p_id, p_node in self._nodes_by_id_cache.items():
                    if isinstance(p_node, SetFolderNodeModel):
                        child_ids = {child.id for child in p_node.children}
                        if temp_parent.id in child_ids:
                            temp_parent = self.get_folder_node_by_id(p_id)
                            parent_of_temp_found = True
                            break
                if not parent_of_temp_found:
                    temp_parent = None
        target_list = new_parent_folder.children
        if target_index != -1 and 0 <= target_index <= len(target_list):
            target_list.insert(target_index, node_to_move)
        else:
            target_list.append(node_to_move)
        self._set_dirty(True)
        self._rebuild_nodes_cache()
        return True

    def add_script_entry_model_to_set(
        self, set_node_id: str, script_entry_model: ScriptSetEntryModel
    ) -> Optional[ScriptSetEntryModel]:
        target_set = self.get_set_node_by_id(set_node_id)
        if not target_set:
            return None
        target_set.script_entries.append(script_entry_model)
        self._set_dirty(True)
        return script_entry_model

    def remove_script_entry_from_set(
        self, set_node_id: str, instance_id_to_remove: str
    ) -> bool:
        target_set = self.get_set_node_by_id(set_node_id)
        if not target_set:
            return False
        initial_len = len(target_set.script_entries)
        target_set.script_entries = [
            e
            for e in target_set.script_entries
            if e.instance_id != instance_id_to_remove
        ]
        if len(target_set.script_entries) < initial_len:
            self._cleanup_orphaned_favorites()
            self._set_dirty(True)
            return True
        return False

    def reorder_script_entries_in_set(
        self, set_node_id: str, new_ordered_instance_ids: List[str]
    ) -> bool:
        target_set = self.get_set_node_by_id(set_node_id)
        if not target_set:
            return False
        entries_map = {e.instance_id: e for e in target_set.script_entries}
        if set(new_ordered_instance_ids) != set(entries_map.keys()):
            return False
        try:
            target_set.script_entries = [
                entries_map[inst_id] for inst_id in new_ordered_instance_ids
            ]
            self._set_dirty(True)
            return True
        except KeyError:
            return False

    def get_script_entry_by_instance_id(
        self, set_node_id: str, instance_id: str
    ) -> Optional[ScriptSetEntryModel]:
        target_set = self.get_set_node_by_id(set_node_id)
        if not target_set:
            return None
        return next(
            (e for e in target_set.script_entries if e.instance_id == instance_id), None
        )

    def update_script_entry(
        self, set_node_id: str, updated_entry: ScriptSetEntryModel
    ) -> bool:
        target_set = self.get_set_node_by_id(set_node_id)
        if not target_set:
            return False

        for i, entry in enumerate(target_set.script_entries):
            if entry.instance_id == updated_entry.instance_id:
                target_set.script_entries[i] = updated_entry
                self._set_dirty(True)
                return True
        return False

    def reload_context_from_file(self) -> bool:
        if not self.current_collection_file_path:
            logger.warning(
                locale_manager.get("set_manager.log_warning.reload_context_no_path")
            )
            return False

        context_file_path = self._get_context_file_path(
            self.current_collection_file_path
        )
        self.current_collection_model.context_data = self._load_context_data_from_path(
            context_file_path
        )

        return True
        
    # 1. БЛОК: Новый метод export_set_node
    # ==============================================================================
    def export_set_node(self, node_id: str) -> Optional[str]:
        """
        Экспортирует ScriptSetNodeModel в JSON-строку.

        :param node_id: ID ScriptSetNodeModel для экспорта.
        :return: JSON-строка, представляющая набор, или None в случае ошибки.
        """
        set_node = self.get_set_node_by_id(node_id)
        if not set_node:
            logger.warning(
                locale_manager.get(
                    "set_manager.log_warning.export_set_not_found", node_id=node_id
                )
            )
            return None
        
        try:
            # Используем Pydantic метод model_dump_json для сериализации
            json_data = set_node.model_dump_json(indent=2)
            logger.info(
                locale_manager.get(
                    "set_manager.log_info.set_exported_success", name=set_node.name
                )
            )
            return json_data
        except Exception as e:
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.export_set_failed",
                    name=set_node.name,
                    error=e,
                ),
                exc_info=True,
            )
            return None

    # 2. БЛОК: Новый метод import_set_node
    # ==============================================================================
    def import_set_node(self, json_data: str, parent_folder_id: str) -> Optional[ScriptSetNodeModel]:
        """
        Импортирует ScriptSetNodeModel из JSON-строки, генерирует новые ID
        и добавляет его в указанную родительскую папку.
        Также обновляет ссылки на instance_id внутри параметров скриптов.

        :param json_data: JSON-строка, представляющая набор.
        :param parent_folder_id: ID SetFolderNodeModel, куда будет импортирован набор.
        :return: Импортированный ScriptSetNodeModel или None в случае ошибки.
        """
        parent_folder = self.get_folder_node_by_id(parent_folder_id)
        if not parent_folder:
            logger.warning(
                locale_manager.get(
                    "set_manager.log_warning.import_parent_not_found", folder_id=parent_folder_id
                )
            )
            return None

        try:
            # 1. Десериализация и создание карты сопоставления ID
            original_set_node = ScriptSetNodeModel.model_validate_json(json_data)
            
            # Новый ID для самого набора
            new_set_id = f"setnode_{uuid.uuid4().hex[:12]}"
            
            id_map: Dict[str, str] = {}
            new_script_entries: List[ScriptSetEntryModel] = []

            for entry in original_set_node.script_entries:
                old_instance_id = entry.instance_id
                new_instance_id = f"instance_{uuid.uuid4().hex[:12]}"
                id_map[old_instance_id] = new_instance_id
                
                # Создаем копию элемента с новым ID
                new_entry = entry.model_copy(deep=True)
                new_entry.instance_id = new_instance_id
                new_script_entries.append(new_entry)
            
            # 2. Обновление ссылок в параметрах
            for new_entry in new_script_entries:
                if not new_entry.command_line_args:
                    continue
                
                # Проходим по всем аргументам командной строки
                for arg_name, arg_value_enabled_model in new_entry.command_line_args.items():
                    value = arg_value_enabled_model.value
                    
                    if isinstance(value, str):
                        # Если значение - строка и является старым instance_id
                        if value in id_map:
                            arg_value_enabled_model.value = id_map[value]
                    elif isinstance(value, list):
                        # Если значение - список, проверяем каждый элемент списка
                        updated_list: List[Union[str, int, float, bool, dict]] = []
                        for item in value:
                            if isinstance(item, str) and item in id_map:
                                updated_list.append(id_map[item])
                            else:
                                updated_list.append(item)
                        arg_value_enabled_model.value = updated_list
            
            # Создаем новый объект ScriptSetNodeModel с новым ID и обновленными элементами
            imported_set_node = ScriptSetNodeModel(
                id=new_set_id,
                name=original_set_node.name,
                description=original_set_node.description,
                script_entries=new_script_entries
            )

            # 3. Добавление в коллекцию
            parent_folder.children.append(imported_set_node)
            self._rebuild_nodes_cache()
            self._set_dirty(True)
            logger.info(
                locale_manager.get(
                    "set_manager.log_info.set_imported_success", name=imported_set_node.name
                )
            )
            return imported_set_node
        except ValidationError as ve:
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.import_set_validation_failed",
                    error=ve.errors(include_url=False),
                ),
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                locale_manager.get(
                    "set_manager.log_error.import_set_generic_failed", error=e
                ),
                exc_info=True,
            )
            return None        

    # ==============================================================================
    # БЛОК: Управление избранными скриптами (Quick Access)
    # ==============================================================================

    def toggle_favorite(self, instance_id: str, icon_name: str = "STAR", icon_color: Optional[str] = None) -> bool:
        favorites = self.current_collection_model.favorite_scripts
        for i, fav in enumerate(favorites):
            if fav.instance_id == instance_id:
                favorites.pop(i)
                self._set_dirty(True)
                return False 
        
        favorites.append(FavoriteScriptModel(instance_id=instance_id, icon_name=icon_name, icon_color=icon_color))
        self._set_dirty(True)
        return True

    def update_favorite_icon(self, instance_id: str, new_icon_name: str, new_icon_color: Optional[str] = None) -> bool:
        for fav in self.current_collection_model.favorite_scripts:
            if fav.instance_id == instance_id:
                if fav.icon_name != new_icon_name or fav.icon_color != new_icon_color:
                    fav.icon_name = new_icon_name
                    fav.icon_color = new_icon_color
                    self._set_dirty(True)
                    return True
                return False
        return False

    def is_favorite(self, instance_id: str) -> bool:
        """Проверяет, находится ли скрипт в избранном."""
        return any(fav.instance_id == instance_id for fav in self.current_collection_model.favorite_scripts)

    def get_favorite_icon(self, instance_id: str) -> str:
        """Возвращает имя иконки для избранного скрипта или STAR по умолчанию."""
        for fav in self.current_collection_model.favorite_scripts:
            if fav.instance_id == instance_id:
                return fav.icon_name
        return "STAR"

    def get_all_favorites(self) -> List[FavoriteScriptModel]:
        """Возвращает список всех избранных скриптов."""
        return self.current_collection_model.favorite_scripts

    def find_script_entry_by_instance_id(self, instance_id: str) -> Optional[ScriptSetEntryModel]:
        """Ищет ScriptSetEntryModel во всей коллекции по instance_id."""
        def _search_in_nodes(nodes: List[SetHierarchyNodeType]) -> Optional[ScriptSetEntryModel]:
            for node in nodes:
                if isinstance(node, ScriptSetNodeModel):
                    for entry in node.script_entries:
                        if entry.instance_id == instance_id:
                            return entry
                elif isinstance(node, SetFolderNodeModel) and node.children:
                    found = _search_in_nodes(node.children)
                    if found:
                        return found
            return None
        return _search_in_nodes(self.current_collection_model.root_nodes)

    def find_parent_set_id_for_instance(self, instance_id: str) -> Optional[str]:
        """Ищет ID родительского набора (ScriptSetNodeModel) по instance_id."""
        def _search_in_nodes(nodes: List[SetHierarchyNodeType]) -> Optional[str]:
            for node in nodes:
                if isinstance(node, ScriptSetNodeModel):
                    for entry in node.script_entries:
                        if entry.instance_id == instance_id:
                            return node.id
                elif isinstance(node, SetFolderNodeModel) and node.children:
                    found = _search_in_nodes(node.children)
                    if found:
                        return found
            return None
        return _search_in_nodes(self.current_collection_model.root_nodes)

    def _cleanup_orphaned_favorites(self):
        """Удаляет из избранного ссылки на несуществующие экземпляры скриптов."""
        existing_instance_ids = set()
        def _collect_ids(nodes: List[SetHierarchyNodeType]):
            for node in nodes:
                if isinstance(node, ScriptSetNodeModel):
                    for entry in node.script_entries:
                        existing_instance_ids.add(entry.instance_id)
                elif isinstance(node, SetFolderNodeModel) and node.children:
                    _collect_ids(node.children)
                    
        _collect_ids(self.current_collection_model.root_nodes)
        
        initial_count = len(self.current_collection_model.favorite_scripts)
        self.current_collection_model.favorite_scripts =[
            fav for fav in self.current_collection_model.favorite_scripts
            if fav.instance_id in existing_instance_ids
        ]
        
        if len(self.current_collection_model.favorite_scripts) != initial_count:
            self._set_dirty(True)            

    def find_entry_and_parent_set(self, instance_id: str) -> Optional[Tuple[ScriptSetEntryModel, ScriptSetNodeModel]]:
        """
        Выполняет глобальный поиск по всей коллекции.
        Возвращает кортеж (экземпляр скрипта, родительский набор), если скрипт найден.
        """
        def _search_in_nodes(nodes: List[SetHierarchyNodeType]) -> Optional[Tuple[ScriptSetEntryModel, ScriptSetNodeModel]]:
            for node in nodes:
                if isinstance(node, ScriptSetNodeModel):
                    for entry in node.script_entries:
                        if entry.instance_id == instance_id:
                            return entry, node
                elif isinstance(node, SetFolderNodeModel) and node.children:
                    found = _search_in_nodes(node.children)
                    if found:
                        return found
            return None
            
        return _search_in_nodes(self.current_collection_model.root_nodes)
