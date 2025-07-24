# pysm_lib/set_manager.py

import pathlib
import json
import logging
import os
from typing import List, Optional, Dict, Tuple


from .models import (
    ScriptSetsCollectionModel,
    SetFolderNodeModel,
    ScriptSetNodeModel,
    SetHierarchyNodeType,
    ScriptSetEntryModel,
    ScriptRootModel,
    ContextVariableModel,
)

from pydantic import ValidationError

from .app_constants import APPLICATION_ROOT_DIR
from .path_utils import to_relative_if_possible, resolve_path
from .locale_manager import LocaleManager


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
            with open(context_file_path, "r", encoding="utf-8") as f:
                raw_context_data = json.load(f)

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
