"""CLI contracts for three independent photo-selection stages."""

from __future__ import annotations

import argparse

try:
    from pysm_lib.pysm_context import ConfigResolver
except ImportError:
    ConfigResolver = None


TITLES = {
    "selection_import": "Импорт выбора фотографий",
    "selection_copy": "Копирование выбранных фотографий",
    "selection_assignments": "План вёрстки",
}


def make_parser(stage: str) -> argparse.ArgumentParser:
    """Expose only inputs used by the chosen stage."""
    parser = argparse.ArgumentParser(description=TITLES[stage])
    if stage != "selection_copy":
        parser.add_argument("--student_list_file", required=True, help="Общий список учеников *.list.")
    analysis_help = (
        "Папка с photo_selection.json и info_faces.json."
        if stage == "selection_copy"
        else "Папка с photo_selection.json и результатами анализа."
    )
    parser.add_argument("--analysis_dir", required=True, help=analysis_help)
    if stage != "selection_copy":
        parser.add_argument("--session_name", required=True, help="Имя сессии класса/группы.")
        parser.add_argument("--photo_session", required=True, help="Текущая фотосессия.")
    if stage != "selection_import":
        parser.add_argument("--source_dir", required=True, help="Корень исходных файлов.")
        parser.add_argument("--dest_dir", required=True, help="Корень целевых папок локаций.")
        parser.add_argument("--exclude_dirs", default="Masks", help="Исключаемые имена папок, через запятую.")
    if stage == "selection_copy":
        parser.add_argument(
            "--mode", choices=("copy", "move"), default="copy",
            help="copy — копировать с сохранением оригиналов; move — переместить файлы.",
        )
        parser.add_argument(
            "--on_conflict", choices=("skip", "overwrite", "rename"), default="skip",
            help="Если файл существует: skip — пропустить, overwrite — перезаписать, rename — добавить числовой суффикс.",
        )
    else:
        parser.add_argument("--title", default=TITLES[stage], help="Заголовок окна.")
        parser.add_argument("--message", default="", help="HTML-инструкция в информационной панели.")
    return parser


def resolve_config(stage: str) -> argparse.Namespace:
    """Resolve PySM templates before creating any window or accessing inputs."""
    parser = make_parser(stage)
    return ConfigResolver(parser).resolve_all() if ConfigResolver else parser.parse_args()
