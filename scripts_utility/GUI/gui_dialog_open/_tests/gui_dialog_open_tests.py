"""Тесты контрактов выбора пути для gui_dialog_open."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from scripts_utility.GUI.gui_dialog_open import run_gui_dialog_open as dialog_open


class PathBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.root = self.base_dir / "allowed"
        self.child = self.root / "nested"
        self.grandchild = self.child / "deep"
        self.sibling = self.base_dir / "allowed-other"
        self.grandchild.mkdir(parents=True)
        self.sibling.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_only_in_initial_dir_accepts_direct_children_only(self) -> None:
        self.assertFalse(
            dialog_open._is_selection_allowed(
                str(self.root),
                str(self.root),
                dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR,
            )
        )
        self.assertTrue(
            dialog_open._is_selection_allowed(
                str(self.child),
                str(self.root),
                dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR,
            )
        )
        self.assertFalse(
            dialog_open._is_selection_allowed(
                str(self.grandchild),
                str(self.root),
                dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR,
            )
        )

    def test_only_in_initial_dir_and_subfolders_rejects_root(self) -> None:
        self.assertFalse(
            dialog_open._is_selection_allowed(
                str(self.root),
                str(self.root),
                dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS,
            )
        )
        self.assertTrue(
            dialog_open._is_selection_allowed(
                str(self.grandchild),
                str(self.root),
                dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS,
            )
        )

    def test_initial_dir_and_subfolders_accepts_root_and_descendants(self) -> None:
        for candidate in (self.root, self.child, self.grandchild):
            with self.subTest(candidate=candidate):
                self.assertTrue(
                    dialog_open._is_selection_allowed(
                        str(candidate),
                        str(self.root),
                        dialog_open.LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS,
                    )
                )

    def test_restricted_modes_reject_similarly_named_sibling(self) -> None:
        for limit_mode in (
            dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR,
            dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS,
            dialog_open.LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS,
        ):
            with self.subTest(limit_mode=limit_mode):
                self.assertFalse(
                    dialog_open._is_selection_allowed(
                        str(self.sibling),
                        str(self.root),
                        limit_mode,
                    )
                )

    def test_all_accepts_path_outside_initial_dir(self) -> None:
        self.assertTrue(
            dialog_open._is_selection_allowed(
                str(self.sibling),
                str(self.root),
                dialog_open.LIMIT_MODE_ALL,
            )
        )

    def test_link_to_outside_is_rejected_when_supported(self) -> None:
        link = self.root / "outside-link"
        try:
            link.symlink_to(self.sibling, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Создание ссылки недоступно: {error}")

        self.assertFalse(
            dialog_open._is_selection_allowed(
                str(link),
                str(self.root),
                dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS,
            )
        )


class LimitModeNormalizationTests(unittest.TestCase):
    def test_path_resolved_choice_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved_choice = str(
                Path(temp_dir) / dialog_open.LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS
            )

            result = dialog_open._normalize_limit_mode(resolved_choice)

        self.assertEqual(
            result,
            dialog_open.LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS,
        )

    def test_legacy_boolean_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Неизвестный режим"):
            dialog_open._normalize_limit_mode(True)

    def test_cli_requires_one_of_the_new_choice_values(self) -> None:
        with (
            mock.patch.object(dialog_open, "IS_MANAGED_RUN", False),
            mock.patch.object(
                sys,
                "argv",
                [
                    "run_gui_dialog_open.py",
                    "--dlg_open_limit_to_initial_dir",
                    dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR,
                ],
            ),
        ):
            config = dialog_open.get_config()

        self.assertEqual(
            config.dlg_open_limit_to_initial_dir,
            dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR,
        )

    def test_cli_default_is_all(self) -> None:
        with (
            mock.patch.object(dialog_open, "IS_MANAGED_RUN", False),
            mock.patch.object(sys, "argv", ["run_gui_dialog_open.py"]),
        ):
            config = dialog_open.get_config()

        self.assertEqual(
            config.dlg_open_limit_to_initial_dir,
            dialog_open.LIMIT_MODE_ALL,
        )


class InitialDirectoryTests(unittest.TestCase):
    def test_configured_existing_directory_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(
                dlg_open_initial_dir=temp_dir,
                dlg_open_var="selected_path",
            )

            initial_dir, configured = dialog_open._determine_initial_directory(config)

        self.assertEqual(initial_dir, dialog_open._normalized_absolute_path(temp_dir))
        self.assertTrue(configured)

    def test_previous_selected_file_remains_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_file = Path(temp_dir) / "selected.txt"
            selected_file.touch()
            config = SimpleNamespace(
                dlg_open_initial_dir="",
                dlg_open_var="selected_path",
                dlg_open_result_mode="full_path",
            )
            previous_value = SimpleNamespace(exists=True, value=str(selected_file))

            with mock.patch.object(
                dialog_open,
                "read_context_value",
                return_value=previous_value,
            ):
                initial_dir, configured = dialog_open._determine_initial_directory(config)

        self.assertEqual(initial_dir, dialog_open._normalized_absolute_path(temp_dir))
        self.assertFalse(configured)

    def test_template_in_previous_directory_is_resolved_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir)
            selected_dir = (
                app_root
                / "script_collections"
                / "DEMO_WORKFLOW"
                / "2027_RAW"
            )
            selected_dir.mkdir(parents=True)
            stored_value = (
                r"{pysm_sys_info.app_root_dir}"
                r"\script_collections\DEMO_WORKFLOW\2027_RAW"
            )
            config = SimpleNamespace(
                dlg_open_initial_dir="",
                dlg_open_var="selected_path",
                dlg_open_result_mode="full_path",
            )
            previous_value = SimpleNamespace(exists=True, value=stored_value)
            context = mock.Mock()
            context.resolve_template.return_value = str(selected_dir)
            context.resolve_path.return_value = selected_dir

            with (
                mock.patch.object(dialog_open, "pysm_context", context),
                mock.patch.object(
                    dialog_open,
                    "read_context_value",
                    return_value=previous_value,
                ),
            ):
                initial_dir, configured = dialog_open._determine_initial_directory(config)

        self.assertEqual(
            initial_dir,
            dialog_open._normalized_absolute_path(str(selected_dir)),
        )
        self.assertFalse(configured)
        context.resolve_template.assert_called_once_with(stored_value)
        context.resolve_path.assert_called_once_with(str(selected_dir))

    def test_name_mode_uses_collection_dir_instead_of_result_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            misleading_name = base_dir / "selected-name"
            collection_dir = base_dir / "collection"
            misleading_name.mkdir()
            collection_dir.mkdir()
            config = SimpleNamespace(
                dlg_open_initial_dir="",
                dlg_open_var="selected_path",
                dlg_open_result_mode="name",
            )

            def read_value(_context: object, key: str) -> SimpleNamespace:
                if key == "pysm_info.collection_dir":
                    return SimpleNamespace(exists=True, value=str(collection_dir))
                return SimpleNamespace(exists=True, value=str(misleading_name))

            with mock.patch.object(
                dialog_open,
                "read_context_value",
                side_effect=read_value,
            ):
                initial_dir, configured = dialog_open._determine_initial_directory(config)

        self.assertEqual(
            initial_dir,
            dialog_open._normalized_absolute_path(str(collection_dir)),
        )
        self.assertFalse(configured)

    def test_configured_missing_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_dir = Path(temp_dir) / "missing"
            config = SimpleNamespace(
                dlg_open_initial_dir=str(missing_dir),
                dlg_open_var="selected_path",
            )

            with self.assertRaisesRegex(ValueError, "Начальная папка не существует"):
                dialog_open._determine_initial_directory(config)


class DialogSelectionTests(unittest.TestCase):
    def test_only_in_initial_dir_rejects_deeper_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            initial_dir = Path(temp_dir) / "allowed"
            direct_child = initial_dir / "direct"
            nested_child = direct_child / "nested"
            nested_child.mkdir(parents=True)
            config = SimpleNamespace(
                dlg_open_type="directory",
                dlg_open_title="Выберите папку",
                dlg_open_filter="",
                dlg_open_limit_to_initial_dir=(
                    dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR
                ),
            )

            with (
                mock.patch.object(
                    dialog_open.QFileDialog,
                    "getExistingDirectory",
                    side_effect=[str(nested_child), str(direct_child)],
                ) as directory_dialog,
                mock.patch.object(
                    dialog_open,
                    "_show_outside_initial_dir_warning",
                ) as warning,
            ):
                result = dialog_open._open_selection_dialog(config, str(initial_dir))

        self.assertEqual(
            result,
            dialog_open._normalized_absolute_path(str(direct_child)),
        )
        self.assertEqual(directory_dialog.call_count, 2)
        warning.assert_called_once()

    def test_initial_dir_and_subfolders_accepts_initial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            initial_dir = Path(temp_dir) / "allowed"
            initial_dir.mkdir()
            config = SimpleNamespace(
                dlg_open_type="directory",
                dlg_open_title="Выберите папку",
                dlg_open_filter="",
                dlg_open_limit_to_initial_dir=(
                    dialog_open.LIMIT_MODE_INITIAL_DIR_AND_SUBFOLDERS
                ),
            )

            with (
                mock.patch.object(
                    dialog_open.QFileDialog,
                    "getExistingDirectory",
                    return_value=str(initial_dir),
                ),
                mock.patch.object(
                    dialog_open,
                    "_show_outside_initial_dir_warning",
                ) as warning,
            ):
                result = dialog_open._open_selection_dialog(config, str(initial_dir))

        self.assertEqual(
            result,
            dialog_open._normalized_absolute_path(str(initial_dir)),
        )
        warning.assert_not_called()

    def test_initial_directory_is_rejected_and_nested_directory_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            initial_dir = Path(temp_dir) / "allowed"
            nested_dir = initial_dir / "nested"
            nested_dir.mkdir(parents=True)
            config = SimpleNamespace(
                dlg_open_type="directory",
                dlg_open_title="Выберите папку",
                dlg_open_filter="",
                dlg_open_limit_to_initial_dir=(
                    dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS
                ),
            )

            with (
                mock.patch.object(
                    dialog_open.QFileDialog,
                    "getExistingDirectory",
                    side_effect=[str(initial_dir), str(nested_dir)],
                ) as directory_dialog,
                mock.patch.object(
                    dialog_open,
                    "_show_outside_initial_dir_warning",
                ) as warning,
            ):
                result = dialog_open._open_selection_dialog(config, str(initial_dir))

        self.assertEqual(result, dialog_open._normalized_absolute_path(str(nested_dir)))
        self.assertEqual(directory_dialog.call_count, 2)
        warning.assert_called_once()

    def test_outside_file_is_rejected_and_dialog_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            initial_dir = base_dir / "allowed"
            outside_dir = base_dir / "outside"
            initial_dir.mkdir()
            outside_dir.mkdir()
            outside_file = outside_dir / "outside.txt"
            inside_file = initial_dir / "inside.txt"
            outside_file.touch()
            inside_file.touch()
            config = SimpleNamespace(
                dlg_open_type="file",
                dlg_open_title="Выберите файл",
                dlg_open_filter="Все файлы (*.*)",
                dlg_open_limit_to_initial_dir=(
                    dialog_open.LIMIT_MODE_ONLY_IN_INITIAL_DIR_AND_SUBFOLDERS
                ),
            )

            with (
                mock.patch.object(
                    dialog_open.QFileDialog,
                    "getOpenFileName",
                    side_effect=[
                        (str(outside_file), ""),
                        (str(inside_file), ""),
                    ],
                ) as file_dialog,
                mock.patch.object(
                    dialog_open,
                    "_show_outside_initial_dir_warning",
                ) as warning,
            ):
                result = dialog_open._open_selection_dialog(config, str(initial_dir))

        self.assertEqual(result, dialog_open._normalized_absolute_path(str(inside_file)))
        self.assertEqual(file_dialog.call_count, 2)
        warning.assert_called_once()

    def test_cancel_returns_empty_result(self) -> None:
        config = SimpleNamespace(
            dlg_open_type="directory",
            dlg_open_title="Выберите папку",
            dlg_open_filter="",
            dlg_open_limit_to_initial_dir=dialog_open.LIMIT_MODE_ALL,
        )
        with mock.patch.object(
            dialog_open.QFileDialog,
            "getExistingDirectory",
            return_value="",
        ):
            result = dialog_open._open_selection_dialog(config, ".")

        self.assertEqual(result, "")


class ContextResultTests(unittest.TestCase):
    def test_file_name_mode_returns_string(self) -> None:
        value, var_type = dialog_open._prepare_context_result(
            r"D:\Photos\IMG_0001.jpg",
            "name",
            "file",
        )

        self.assertEqual(value, "IMG_0001.jpg")
        self.assertEqual(var_type, "string")

    def test_directory_name_mode_returns_string(self) -> None:
        value, var_type = dialog_open._prepare_context_result(
            r"D:\Photos\Group 1",
            "name",
            "directory",
        )

        self.assertEqual(value, "Group 1")
        self.assertEqual(var_type, "string")

    def test_full_directory_path_preserves_path_type(self) -> None:
        selected_path = r"D:\Photos\Group 1"
        value, var_type = dialog_open._prepare_context_result(
            selected_path,
            "full_path",
            "directory",
        )

        self.assertEqual(value, selected_path)
        self.assertEqual(var_type, "dir_path")


if __name__ == "__main__":
    unittest.main()
