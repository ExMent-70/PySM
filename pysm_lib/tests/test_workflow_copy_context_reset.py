"""Проверки сброса контекста при копировании рабочего процесса."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import toml

from pysm_lib.app_controller import AppController, _CollectionCopyRestorePoint
from pysm_lib.config_manager import AppConfigModel, ConfigManager
from pysm_lib.models import ContextVariableModel, ScriptSetsCollectionModel
from pysm_lib.set_manager import SetManager


class WorkflowCopyContextResetTests(unittest.TestCase):
    class SignalStub:
        def __init__(self) -> None:
            self.calls = []

        def emit(self, *args) -> None:
            self.calls.append(args)

    def test_config_normalizes_reset_variable_names(self) -> None:
        config = AppConfigModel.model_validate(
            {
                "workflow_copy": {
                    "reset_context_variables": [
                        " project.path ",
                        "project.path",
                        "",
                        "student_ids",
                    ]
                }
            }
        )

        self.assertEqual(
            config.workflow_copy.reset_context_variables,
            ["project.path", "student_ids"],
        )

    def test_copy_resets_configured_values_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SetManager(Path(temp_dir))
            source = ScriptSetsCollectionModel(
                collection_name="Source",
                context_data={
                    "title": ContextVariableModel(
                        type="string",
                        value="Album",
                        description="Album title",
                        read_only=True,
                    ),
                    "counter": ContextVariableModel(type="int", value=12),
                    "ratio": ContextVariableModel(type="float", value=1.5),
                    "enabled": ContextVariableModel(type="bool", value=True),
                    "student_ids": ContextVariableModel(
                        type="list",
                        value=["S001", "S002"],
                    ),
                    "settings": ContextVariableModel(
                        type="json",
                        value={"brightness": 1.2},
                    ),
                    "project": ContextVariableModel(
                        type="json",
                        value={
                            "path": r"D:\Photos",
                            "name": "School 42",
                            "items": [
                                {"path": r"D:\Photos\001.jpg", "selected": True}
                            ],
                        },
                    ),
                    "untouched": ContextVariableModel(
                        type="string",
                        value="keep",
                    ),
                },
            )
            manager.current_collection_model = source

            copied = manager.create_collection_from_current(
                [
                    "title",
                    "counter",
                    "ratio",
                    "enabled",
                    "student_ids",
                    "settings",
                    "project.path",
                    "project.items.0.selected",
                    "missing.value",
                ]
            )

            self.assertEqual(
                manager.last_reset_context_variables,
                [
                    "title",
                    "counter",
                    "ratio",
                    "enabled",
                    "student_ids",
                    "settings",
                    "project.path",
                    "project.items.0.selected",
                ],
            )

        self.assertEqual(copied.context_data["title"].value, "")
        self.assertEqual(copied.context_data["title"].description, "Album title")
        self.assertIs(copied.context_data["title"].read_only, True)
        self.assertEqual(copied.context_data["counter"].value, 0)
        self.assertEqual(copied.context_data["ratio"].value, 0.0)
        self.assertIs(copied.context_data["enabled"].value, False)
        self.assertEqual(copied.context_data["student_ids"].value, [])
        self.assertEqual(copied.context_data["settings"].value, {})
        self.assertEqual(copied.context_data["project"].value["path"], "")
        self.assertEqual(
            copied.context_data["project"].value["name"],
            "School 42",
        )
        self.assertIs(
            copied.context_data["project"].value["items"][0]["selected"],
            False,
        )
        self.assertEqual(copied.context_data["untouched"].value, "keep")

        self.assertEqual(source.context_data["title"].value, "Album")
        self.assertEqual(source.context_data["project"].value["path"], r"D:\Photos")

    def test_save_preserves_workflow_copy_list_and_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            initial_config = AppConfigModel.model_validate(
                {
                    "environment_variables": {
                        "NO_ALBUMENTATIONS_UPDATE": "1",
                    }
                }
            )
            config_path.write_text(
                toml.dumps(initial_config.model_dump(mode="python")),
                encoding="utf-8",
            )
            manager = ConfigManager(config_path)

            disk_config = toml.load(config_path)
            disk_config["workflow_copy"]["reset_context_variables"] = [
                "project.path",
                "student_ids",
            ]
            config_path.write_text(toml.dumps(disk_config), encoding="utf-8")

            self.assertTrue(manager.save_config())

            saved_config = toml.load(config_path)
            self.assertEqual(
                saved_config["workflow_copy"]["reset_context_variables"],
                ["project.path", "student_ids"],
            )
            self.assertEqual(
                manager.config.workflow_copy.reset_context_variables,
                ["project.path", "student_ids"],
            )
            self.assertEqual(
                saved_config["environment_variables"],
                {"NO_ALBUMENTATIONS_UPDATE": "1"},
            )

    def test_copy_command_passes_configured_reset_list_to_set_manager(self) -> None:
        reset_names = ["project.path", "student_ids"]
        received = []
        confirm_calls = []
        restore_calls = []
        source_file = Path(r"D:\Projects\Source\Source.pysmc")
        source_model = ScriptSetsCollectionModel(
            collection_name="Source",
            context_data={
                "wf_psd_path": ContextVariableModel(
                    type="dir_path",
                    value=r"D:\Photos\School 42",
                )
            },
        )
        set_manager = SimpleNamespace(
            create_collection_from_current=lambda names: received.append(names),
            last_reset_context_variables=["project.path", "student_ids"],
            current_collection_model=source_model,
            current_collection_file_path=source_file,
            is_dirty=True,
        )
        config_manager = SimpleNamespace(
            config=SimpleNamespace(
                workflow_copy=SimpleNamespace(
                    reset_context_variables=reset_names,
                )
            ),
            last_used_sets_collection_file="source.pysmc",
            reload_workflow_copy_config=lambda: True,
            save_config=lambda: True,
        )

        def get_locale_text(key: str, **kwargs) -> str:
            if key == "user_actions.context_variables_reset":
                return f"{key}:\n{kwargs['variables']}"
            if key == "user_actions.collection_copy_save_warning":
                return (
                    "<b>ОБЯЗАТЕЛЬНО СОХРАНИТЕ НОВЫЙ РАБОЧИЙ ПРОЦЕСС</b>\n"
                    "Например, в отдельную папку внутри текущей рабочей папки "
                    "<b>{wf_psd_path}</b>"
                )
            return key

        def restore_source(restore_point) -> None:
            restore_calls.append(restore_point)
            set_manager.current_collection_model = restore_point.collection_model
            controller.current_collection_file_path = (
                restore_point.controller_file_path
            )

        controller = SimpleNamespace(
            _update_suggested_save_dir=lambda: None,
            set_manager=set_manager,
            current_collection_file_path=source_file,
            selected_set_node_id="setnode_source",
            clear_console_request=self.SignalStub(),
            _log_welcome_message=lambda: None,
            log_message_to_console=self.SignalStub(),
            locale_manager=SimpleNamespace(get=get_locale_text),
            config_manager=config_manager,
            set_active_script_set_node=lambda _node_id: None,
            _request_collection_view_update=lambda: None,
            collection_dirty_state_changed=self.SignalStub(),
            status_message_updated=self.SignalStub(),
            _confirm_new_collection_save=lambda warning: (
                confirm_calls.append(warning) or False
            ),
            _restore_collection_after_copy_cancel=restore_source,
        )

        AppController.new_collection_from_template_requested_by_gui(controller)

        self.assertEqual(received, [reset_names])
        self.assertEqual(controller.current_collection_file_path, source_file)
        self.assertEqual(len(restore_calls), 1)
        self.assertEqual(
            restore_calls[0].collection_model.collection_name,
            "Source",
        )
        self.assertEqual(
            restore_calls[0].selected_set_node_id,
            "setnode_source",
        )
        self.assertEqual(
            confirm_calls,
            [
                "<b>ОБЯЗАТЕЛЬНО СОХРАНИТЕ НОВЫЙ РАБОЧИЙ ПРОЦЕСС</b>\n"
                "Например, в отдельную папку внутри текущей рабочей папки "
                r"<b>D:\Photos\School 42</b>"
            ],
        )
        self.assertEqual(
            controller.log_message_to_console.calls[:4],
            [
                (
                    "runner_info",
                    "user_actions.collection_new_from_template",
                ),
                (
                    "runner_info",
                    "user_actions.context_variables_reset:\n"
                    "• project.path\n"
                    "• student_ids",
                ),
                ("EMPTY_LINE", ""),
                (
                    "runner_info",
                    "<b>ОБЯЗАТЕЛЬНО СОХРАНИТЕ НОВЫЙ РАБОЧИЙ ПРОЦЕСС</b>\n"
                    "Например, в отдельную папку внутри текущей рабочей папки "
                    r"<b>D:\Photos\School 42</b>",
                ),
            ],
        )

    def test_copy_project_paths_restrict_selection_to_working_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_root = Path(temp_dir) / "PSD"
            outside_root = Path(temp_dir) / "Outside"
            working_root.mkdir()
            outside_root.mkdir()

            project_name, project_dir, project_file = (
                AppController._copy_project_paths(
                    working_root,
                    working_root / "New Project.pysmc",
                )
            )

            self.assertEqual(project_name, "New Project")
            self.assertEqual(project_dir, working_root / "New Project")
            self.assertEqual(
                project_file,
                working_root / "New Project" / "New Project.pysmc",
            )
            with self.assertRaisesRegex(ValueError, "outside_working_root"):
                AppController._copy_project_paths(
                    working_root,
                    outside_root / "New Project.pysmc",
                )

    def test_cancel_restores_complete_source_collection_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_file = temp_path / "Source" / "Source.pysmc"
            manager = SetManager(temp_path / "collections")
            source_model = ScriptSetsCollectionModel(
                collection_name="Source",
                context_data={
                    "wf_session_name": ContextVariableModel(
                        type="string",
                        value="Source",
                    )
                },
            )
            restore_point = _CollectionCopyRestorePoint(
                collection_model=source_model.model_copy(deep=True),
                controller_file_path=source_file,
                manager_file_path=source_file,
                selected_set_node_id="setnode_source",
                is_dirty=False,
            )
            manager.current_collection_model = ScriptSetsCollectionModel(
                collection_name="Copy",
                context_data={
                    "wf_session_name": ContextVariableModel(
                        type="string",
                        value="",
                    )
                },
            )
            manager.current_collection_file_path = None
            manager._set_dirty(True)

            selected_nodes = []
            config_saves = []
            refresh_calls = []
            view_updates = []
            config_manager = SimpleNamespace(
                last_used_sets_collection_file="",
                save_config=lambda: config_saves.append(True) or True,
            )

            def get_locale_text(key: str, **kwargs) -> str:
                if key == "user_actions.collection_opened":
                    return f"Opened {kwargs['name']}"
                if key == "app_controller.status_collection_loaded":
                    return f"Loaded {kwargs['name']}"
                return key

            controller = SimpleNamespace(
                set_manager=manager,
                current_collection_file_path=None,
                selected_set_node_id=None,
                selected_set_node_model=None,
                set_active_script_set_node=(
                    lambda node_id: selected_nodes.append(node_id)
                ),
                config_manager=config_manager,
                clear_console_request=self.SignalStub(),
                _log_welcome_message=lambda: None,
                log_message_to_console=self.SignalStub(),
                locale_manager=SimpleNamespace(get=get_locale_text),
                _log_collection_properties=lambda: None,
                refresh_available_scripts_list=lambda: refresh_calls.append(True),
                _request_collection_view_update=(
                    lambda node_id=None: view_updates.append(node_id)
                ),
                collection_dirty_state_changed=self.SignalStub(),
                status_message_updated=self.SignalStub(),
                _node_id_to_select_after_scan=None,
            )

            AppController._restore_collection_after_copy_cancel(
                controller,
                restore_point,
            )

            self.assertEqual(
                manager.current_collection_model.collection_name,
                "Source",
            )
            self.assertEqual(
                manager.current_collection_model.context_data[
                    "wf_session_name"
                ].value,
                "Source",
            )
            self.assertEqual(manager.current_collection_file_path, source_file)
            self.assertFalse(manager.is_dirty)
            self.assertEqual(controller.current_collection_file_path, source_file)
            self.assertEqual(selected_nodes, ["setnode_source"])
            self.assertEqual(
                config_manager.last_used_sets_collection_file,
                str(source_file),
            )
            self.assertEqual(config_saves, [True])
            self.assertEqual(refresh_calls, [True])
            self.assertEqual(view_updates, ["setnode_source"])

    def test_save_copy_creates_project_folder_and_updates_session_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            working_root = temp_path / "PSD"
            working_root.mkdir()
            manager = SetManager(temp_path / "collections")
            manager.current_collection_model = ScriptSetsCollectionModel(
                collection_name="Новый рабочий процесс",
                context_data={
                    "wf_psd_path": ContextVariableModel(
                        type="dir_path",
                        value=str(working_root),
                    ),
                    "wf_session_name": ContextVariableModel(
                        type="string",
                        value="Old Project",
                    ),
                },
            )
            project_dir = working_root / "New Project"
            project_file = project_dir / "New Project.pysmc"
            saved_targets = []
            errors = []

            def save_collection(target: Path) -> bool:
                saved_targets.append(target)
                return manager.save_collection_to_file(target)

            def get_locale_text(key: str, **kwargs) -> str:
                if key == "user_actions.collection_copy_saved":
                    return (
                        "Новый проект сохранён.\n\n"
                        f"Рабочая папка проекта - {kwargs['project_dir']}\n\n"
                        f"Файл проекта - {kwargs['project_file']}"
                    )
                return key

            controller = SimpleNamespace(
                set_manager=manager,
                _show_collection_copy_save_error=(
                    lambda key, **kwargs: errors.append((key, kwargs))
                ),
                _select_new_collection_target=lambda root: (
                    "New Project",
                    project_dir,
                    project_file,
                ),
                update_collection_context=manager.update_collection_context,
                save_current_collection_requested_by_gui=save_collection,
                log_message_to_console=self.SignalStub(),
                locale_manager=SimpleNamespace(get=get_locale_text),
            )

            saved = AppController._save_new_collection_in_context_root(controller)

            self.assertTrue(saved)
            self.assertEqual(errors, [])
            self.assertEqual(saved_targets, [project_file])
            self.assertTrue(project_dir.is_dir())
            self.assertTrue(project_file.is_file())
            context_file = project_file.with_suffix(".context.json")
            self.assertTrue(context_file.is_file())
            context_data = json.loads(context_file.read_text(encoding="utf-8"))
            self.assertEqual(
                context_data["wf_session_name"]["value"],
                "New Project",
            )
            self.assertEqual(
                manager.current_collection_model.context_data[
                    "wf_session_name"
                ].value,
                "New Project",
            )
            self.assertEqual(
                controller.log_message_to_console.calls,
                [
                    (
                        "runner_info",
                        "Новый проект сохранён.\n\n"
                        f"Рабочая папка проекта - {project_dir}\n\n"
                        f"Файл проекта - {project_file}",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
