# pysm_lib/script_scanner.py

import pathlib
import json
from typing import List, Dict, Optional, Tuple, Any
import logging
from pydantic import ValidationError

from PySide6.QtCore import QObject, Signal, Slot

from .models import (
    ScriptInfoModel,
    CategoryNodeModel,
    ScanTreeNodeType,
    ScriptRootModel,
)

# --- НОВЫЙ ИМПОРТ ---
from .locale_manager import LocaleManager

# --- ИЗМЕНЕНИЕ: Создаем локальный экземпляр для доступа к строкам ---
locale_manager = LocaleManager()
logger = logging.getLogger(f"PyScriptManager.{__name__}")


def read_category_passport(
    category_folder_path: pathlib.Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    passport_file_name = "_category_meta.json"
    passport_file_path = category_folder_path / passport_file_name
    # --- СТРОКА ИЗМЕНЕНА ---
    logger.debug(
        locale_manager.get(
            "script_scanner.log_debug.reading_category_passport",
            path=passport_file_path,
        )
    )
    if not passport_file_path.is_file():
        return None, None
    try:
        with open(passport_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            # --- СТРОКА ИЗМЕНЕНА ---
            return (
                None,
                locale_manager.get(
                    "script_scanner.error.category_passport_not_dict",
                    filename=passport_file_name,
                ),
            )
        processed_data = data.copy()
        if "icon_file_path_relative" in processed_data and isinstance(
            processed_data["icon_file_path_relative"], str
        ):
            relative_icon_path_str = processed_data.pop("icon_file_path_relative")
            if relative_icon_path_str.strip():
                abs_icon_path = (
                    category_folder_path / relative_icon_path_str
                ).resolve()
                if abs_icon_path.is_file():
                    processed_data["icon_file_abs_path"] = str(abs_icon_path)
        return processed_data, None
    except Exception as e:
        # --- СТРОКА ИЗМЕНЕНА ---
        return (
            None,
            locale_manager.get(
                "script_scanner.error.read_category_passport_failed",
                filename=passport_file_name,
                error=e,
            ),
        )


# pysm_lib/script_scanner.py

# ... (импорты и другие функции без изменений) ...


# 1. БЛОК: Функция read_script_passport (ИЗМЕНЕНА)
def read_script_passport(
    script_folder_path: pathlib.Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    passport_file_name = "script_passport.json"
    passport_file_path = script_folder_path / passport_file_name

    if not passport_file_path.is_file():
        return None, None

    try:
        with open(passport_file_path, "r", encoding="utf-8") as f:
            passport_data_from_file = json.load(f)
    except Exception as e:
        # --- СТРОКА ИЗМЕНЕНА ---
        return None, locale_manager.get(
            "script_scanner.error.read_script_passport_failed",
            filename=passport_file_name,
            error=e,
        )
    if not isinstance(passport_data_from_file, dict):
        # --- СТРОКА ИЗМЕНЕНА ---
        return (
            None,
            locale_manager.get(
                "script_scanner.error.script_passport_not_dict",
                filename=passport_file_name,
            ),
        )

    processed_passport_data = passport_data_from_file.copy()
    pre_scan_errors_list: List[str] = []

    if (
        "description" not in processed_passport_data
        or not isinstance(processed_passport_data.get("description"), str)
        or not processed_passport_data.get("description", "").strip()
    ):
        # --- СТРОКА ИЗМЕНЕНА ---
        pre_scan_errors_list.append(
            locale_manager.get("script_scanner.error.missing_description")
        )

    # --- ИЗМЕНЕНИЕ: Логика преобразования путей УДАЛЕНА ---
    # Теперь мы просто проверяем, что 'script_specific_env_paths' - это список строк,
    # но НЕ преобразуем их в абсолютные пути. Они остаются "как есть".
    if "script_specific_env_paths" in processed_passport_data:
        specific_paths_val = processed_passport_data["script_specific_env_paths"]
        if isinstance(specific_paths_val, list):
            for i, p_val in enumerate(specific_paths_val):
                if not isinstance(p_val, str) or not p_val.strip():
                    # --- СТРОКА ИЗМЕНЕНА ---
                    pre_scan_errors_list.append(
                        locale_manager.get(
                            "script_scanner.error.invalid_path_in_list", index=i + 1
                        )
                    )
                    # Если нашли ошибку, удаляем все поле, чтобы не передавать некорректные данные
                    del processed_passport_data["script_specific_env_paths"]
                    break
        elif specific_paths_val is not None:
            # --- СТРОКА ИЗМЕНЕНА ---
            pre_scan_errors_list.append(
                locale_manager.get("script_scanner.error.env_paths_not_list")
            )
            if "script_specific_env_paths" in processed_passport_data:
                del processed_passport_data["script_specific_env_paths"]

    final_pre_scan_error_summary = (
        "; ".join(pre_scan_errors_list) if pre_scan_errors_list else None
    )
    return processed_passport_data, final_pre_scan_error_summary


# ... (остальные функции без изменений) ...


def find_run_file(script_folder_path: pathlib.Path) -> Optional[pathlib.Path]:
    run_file_name = f"run_{script_folder_path.name}.py"
    run_file_path = script_folder_path / run_file_name
    return run_file_path if run_file_path.is_file() else None


def _scan_directory_recursive(
    current_dir_path: pathlib.Path,
    root_model: ScriptRootModel,
) -> List[ScanTreeNodeType]:
    # --- СТРОКА ИЗМЕНЕНА ---
    logger.debug(
        locale_manager.get(
            "script_scanner.log_debug.recursive_scan_start",
            path=current_dir_path,
            root_id=root_model.id,
        )
    )
    children_nodes: List[ScanTreeNodeType] = []
    base_scan_path = pathlib.Path(root_model.path)

    try:
        dir_items = sorted(
            list(current_dir_path.iterdir()),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except OSError as e:
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.error(
            locale_manager.get(
                "script_scanner.log_error.access_denied", path=current_dir_path, error=e
            )
        )
        return []

    for item_path in dir_items:
        item_name = item_path.name
        if item_name.startswith(".") or (
            item_name.startswith("_") and item_name != "_category_meta.json"
        ):
            continue
        if not item_path.is_dir():
            continue

        script_folder_abs_path = item_path.resolve()
        try:
            relative_path_for_id = script_folder_abs_path.relative_to(
                base_scan_path
            ).as_posix()
            final_id = f"{root_model.id}/{relative_path_for_id}"
        except ValueError:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.error(
                locale_manager.get(
                    "script_scanner.log_error.id_generation_failed",
                    path=script_folder_abs_path,
                    base_path=base_scan_path,
                )
            )
            continue

        run_file_path_obj = find_run_file(script_folder_abs_path)

        if run_file_path_obj:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.debug(
                locale_manager.get(
                    "script_scanner.log_debug.folder_is_script", folder_name=item_name
                )
            )

            raw_script_passport_data, script_passport_read_error = read_script_passport(
                script_folder_abs_path
            )

            constructor_data_script: Dict[str, Any] = {
                "id": final_id,
                "name": item_name,
                "folder_abs_path": str(script_folder_abs_path),
                "run_filename": run_file_path_obj.name,
                "run_file_abs_path": str(run_file_path_obj),
            }

            if raw_script_passport_data:
                constructor_data_script.update(raw_script_passport_data)
                constructor_data_script["is_raw"] = False
                constructor_data_script["passport_valid"] = not bool(
                    script_passport_read_error
                )
                constructor_data_script["passport_error"] = script_passport_read_error
            else:
                constructor_data_script["is_raw"] = True
                constructor_data_script["passport_valid"] = False
                # --- СТРОКА ИЗМЕНЕНА ---
                constructor_data_script["passport_error"] = locale_manager.get(
                    "script_scanner.error.passport_missing"
                )

            try:
                script_model = ScriptInfoModel(**constructor_data_script)
                children_nodes.append(script_model)
            except ValidationError as ve_script:
                error_model = ScriptInfoModel(
                    id=final_id,
                    name=item_name,
                    folder_abs_path=str(script_folder_abs_path),
                    is_raw=True,
                    passport_valid=False,
                    # --- СТРОКА ИЗМЕНЕНА ---
                    passport_error=locale_manager.get(
                        "script_scanner.error.critical_model_error",
                        error=ve_script.errors(include_url=False),
                    ),
                )
                children_nodes.append(error_model)
                # --- СТРОКА ИЗМЕНЕНА ---
                logger.error(
                    locale_manager.get(
                        "script_scanner.log_error.pydantic_script_error",
                        name=item_name,
                        error=ve_script.errors(include_url=False),
                    )
                )

        else:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.debug(
                locale_manager.get(
                    "script_scanner.log_debug.folder_is_category", folder_name=item_name
                )
            )
            category_meta_dict, _ = read_category_passport(script_folder_abs_path)
            sub_children = _scan_directory_recursive(script_folder_abs_path, root_model)

            if sub_children or category_meta_dict:
                try:
                    constructor_data_cat = {
                        "id": final_id,
                        "name": (
                            category_meta_dict.get("display_name", item_name)
                            if category_meta_dict
                            else item_name
                        ),
                        "folder_abs_path": str(script_folder_abs_path),
                        "description": (
                            category_meta_dict.get("description")
                            if category_meta_dict
                            else None
                        ),
                        "children": sub_children,
                    }
                    category_node = CategoryNodeModel(**constructor_data_cat)
                    children_nodes.append(category_node)
                except ValidationError as ve_cat:
                    # --- СТРОКА ИЗМЕНЕНА ---
                    logger.error(
                        locale_manager.get(
                            "script_scanner.log_error.pydantic_category_error",
                            name=item_name,
                            error=ve_cat.errors(include_url=False),
                        )
                    )

    return children_nodes


def scan_scripts_directory(
    script_roots: List[ScriptRootModel],
) -> List[ScanTreeNodeType]:
    # --- СТРОКА ИЗМЕНЕНА ---
    logger.info(
        locale_manager.get(
            "script_scanner.log_info.scan_start", count=len(script_roots)
        )
    )
    if not script_roots:
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.warning(locale_manager.get("script_scanner.log_warning.empty_root_list"))
        return []

    all_scanned_nodes: List[ScanTreeNodeType] = []

    for root_model in script_roots:
        root_path_str = root_model.path
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.info(
            locale_manager.get(
                "script_scanner.log_info.scanning_root",
                path=root_path_str,
                id=root_model.id,
            )
        )

        try:
            base_path = pathlib.Path(root_path_str)
            if not base_path.is_dir():
                # --- СТРОКА ИЗМЕНЕНА ---
                logger.error(
                    locale_manager.get(
                        "script_scanner.log_error.path_not_found_or_not_dir",
                        path=root_path_str,
                    )
                )
                # --- СТРОКИ ИЗМЕНЕНЫ ---
                error_node = CategoryNodeModel(
                    id=root_model.id,
                    name=locale_manager.get(
                        "script_scanner.error_node.name", name=base_path.name
                    ),
                    folder_abs_path=str(base_path),
                    description=locale_manager.get(
                        "script_scanner.error_node.description", path=root_path_str
                    ),
                )
                all_scanned_nodes.append(error_node)
                continue
        except Exception as e:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.error(
                locale_manager.get(
                    "script_scanner.log_error.path_processing_error",
                    path=root_path_str,
                    error=e,
                ),
                exc_info=True,
            )
            continue

        scanned_children = _scan_directory_recursive(base_path.resolve(), root_model)
        # --- СТРОКА ИЗМЕНЕНА ---
        root_node_as_category = CategoryNodeModel(
            id=root_model.id,
            name=base_path.name,
            folder_abs_path=str(base_path.resolve()),
            description=locale_manager.get(
                "script_scanner.category_node.root_description",
                path=str(base_path.resolve()),
            ),
            children=scanned_children,
        )

        all_scanned_nodes.append(root_node_as_category)
        # --- СТРОКА ИЗМЕНЕНА ---
        logger.info(
            locale_manager.get(
                "script_scanner.log_info.scan_root_finished",
                path=root_path_str,
                count=len(scanned_children),
            )
        )
    # --- СТРОКА ИЗМЕНЕНА ---
    logger.info(
        locale_manager.get(
            "script_scanner.log_info.scan_all_finished", count=len(all_scanned_nodes)
        )
    )
    return all_scanned_nodes


class ScriptScannerWorker(QObject):
    """
    Worker для выполнения сканирования в отдельном потоке.
    """

    finished = Signal(list)
    error = Signal(str)

    @Slot(list)
    def run(self, script_roots: List[ScriptRootModel]):
        """
        Запускает сканирование и испускает сигнал с результатом или ошибкой.
        """
        try:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.info(locale_manager.get("script_scanner.log_info.worker_started"))
            result = scan_scripts_directory(script_roots)
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.info(locale_manager.get("script_scanner.log_info.worker_finished"))
            self.finished.emit(result)
        except Exception as e:
            # --- СТРОКА ИЗМЕНЕНА ---
            logger.error(
                locale_manager.get("script_scanner.log_error.worker_error", error=e),
                exc_info=True,
            )
            self.error.emit(
                locale_manager.get("script_scanner.error.worker_runtime_error", error=e)
            )
