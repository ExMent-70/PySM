# analize/cluster_faces/run_log_report.py

# --- Блок 1: Импорты и настройка путей ---
import sys
from pathlib import Path
from typing import Optional

# Попытка импортировать библиотеки из окружения PySM
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib.pysm_context import pysm_context
    from pysm_lib.pysm_theme_api import theme_api

    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    theme_api = None

# --- Блок 2: Константы и иконки ---
ICON_DIR = Path(__file__).resolve().parent / "image"

ICONS = {
    "folder": ICON_DIR / "folder_open.svg",
    "html": ICON_DIR / "html5.svg",
    "database": ICON_DIR / "database.svg",
    "file": ICON_DIR / "file.svg",
    "not_found": ICON_DIR / "error.svg",
}


# --- Блок 3: Вспомогательные функции вывода ---
def log_resource_info(
    description: str,
    resource_path: Path,
    icon_type: str,
    highlight_text: Optional[str] = None,
):
    """
    Выводит информацию о ресурсе.
    Если description пуст, выводит пустую строку-разделитель.
    """
    # --- НОВЫЙ БЛОК: Обработка пустой строки для создания разделителя ---
    if not description:
        # Выводим <br> для создания вертикального отступа (пустой строки).
        print(f"PYSM_HTML_BLOCK:\n", file=sys.stderr, flush=True)
        return

    if highlight_text:
        description = description.replace(
            highlight_text, f"<b>{highlight_text}</b>"
        )

    img_style = "width:20px; height:20px; vertical-align:middle; margin-right:8px;"
    block_style = "padding: 5px 6px; border-radius: 4px; margin-bottom: 4px;"

    icon_path = ICONS.get(icon_type, ICONS["not_found"])
    if not icon_path.exists():
        icon_path = ICONS["not_found"]
    img_tag = f'<img src="{icon_path.as_uri()}" alt="{icon_type}" style="{img_style}">'

    if resource_path and resource_path.exists():
        style = theme_api.get_dynamic_style(
            "api_link", "background-color: #27ae60; color: white;"
        )
        link_text = (
            f'<div style="{style} {block_style}">'
            f'{img_tag}{description}\n</div>'
        )
        pysm_context.log_link(url_or_path=str(resource_path), text=link_text)
    else:
        style = theme_api.get_dynamic_style(
            "api_link_error", "background-color: #c0392b; color: white;"
        )
        error_icon_path = ICONS["not_found"]
        error_img_tag = f'<img src="{error_icon_path.as_uri()}" alt="Not Found" style="{img_style}">'
        error_html = (
            f'<div style="{style} {block_style}">'
            f'{error_img_tag}{description} (не найден)\n</div>'
        )
        print(f"PYSM_HTML_BLOCK:{error_html}", file=sys.stderr, flush=True)


def log_header(text: str):
    """Выводит стилизованный заголовок блока, используя theme_api."""
    header_style = theme_api.get_dynamic_style("script_description", "")
    #html = f"<div style='{header_style}'>{text}</div>"
    html = f"<table style='text-align: center; width: 1200px; background-color: red; border: 1px solid #dddddd;'><tr><td>Название</td><td>Оригинальное название</td><td>Год</td></tr><tr><td>Человек-паук: Возвращение домой b dczlklk fk l;k lk ;lk ;lk  klk lklk </td><td>Spider-Man: Homecoming</td><td>2017</td></tr></table>"
    print(f"PYSM_HTML_BLOCK:{html}", file=sys.stderr, flush=True)
    print("", file=sys.stderr)


# --- Блок 4: Основная функция ---
def main():
    """
    Формирует и выводит "информационную доску" о состоянии ресурсов проекта,
    используя API и темы среды PySM.
    """
    if not IS_MANAGED_RUN or not pysm_context:
        print("Ошибка: Скрипт запущен вне окружения PySM или контекст не найден.", file=sys.stderr)
        return

    session_path_str = pysm_context.get("wf_session_path")
    psd_path_str = pysm_context.get("wf_psd_path")
    idsgn_catalog_path_str = pysm_context.get("var_idsgn_catalog")

    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")
    children_file_name = pysm_context.get("wf_children_file_name")

    # --- Секция 1: Глобальные ресурсы ---
    log_resource_info("", Path(), "file", None), 
    log_header("Блок 1: Глобальные ресурсы")
    log_resource_info("", Path(), "file", None), 

    if session_path_str:
        log_resource_info("Базовая папка для исходных RAW-файлов", Path(session_path_str), "folder")
    if psd_path_str:
        log_resource_info("Базовая папка для альбомов (PSD, INDD)", Path(psd_path_str), "folder")
    if idsgn_catalog_path_str:
        log_resource_info("Базовая папка с шаблонами InDesign", Path(idsgn_catalog_path_str).parent, "folder")
    print("", file=sys.stderr)

    # --- Секция 2: Ресурсы сессии Capture One ---
    log_header("Блок 2: Ресурсы сессии Capture One")
    if session_path_str and session_name:
        base_path = Path(session_path_str) / session_name
        analysis_dir = base_path / "Output" / f"Analysis_{photo_session}"
        rows = [
            ("", Path(), "file", None), 
            ("Рабочая папка активной сессии Capture One", base_path, "folder", session_name),
            ("Папка 'Output'", base_path / "Output", "folder", None),
            ("Папка 'Selects'", base_path / "Selects", "folder", None),
            ("Папка с результатами анализа", analysis_dir, "folder", None),
            ("HTML-отчёт кластеризации", analysis_dir / "face_clustering_report.html", "html", None),
            (f"Файл '{photo_session}_{children_file_name}'", base_path / f"{photo_session}_{children_file_name}", "file", f"{photo_session}_{children_file_name}"),
            (f"Файл сессии '{session_name}.cosessiondb'", base_path / f"{session_name}.cosessiondb", "database", f"{session_name}.cosessiondb"),
        ]
        for desc, path, icon, h_text in rows:
            log_resource_info(desc, path, icon, h_text)
    print("", file=sys.stderr)

    # --- Секция 3: Рабочие файлы Photoshop и InDesign ---
    log_header("Блок 3: Рабочие файлы Photoshop и InDesign")
    if psd_path_str and session_name:
        psd_work_path = Path(psd_path_str) / session_name
        rows = [
            ("", Path(), "file", None), 
            (f"Рабочая папка альбомов '{session_name}'", psd_work_path, "folder", session_name),
            ("Папка с шаблонами InDesign", psd_work_path / "Альбом" / "_ШАБЛОНЫ_", "folder", None),
            ("Папка с файлами PSD", psd_work_path / "Альбом" / "Фото", "folder", None),
            # Пример использования пустой строки для создания разделителя
            ("", Path(), "file", None), 
            ("Приложение к договору (HTML)", psd_work_path / f"{session_name}.html", "html", None),
        ]
        for desc, path, icon, h_text in rows:
            log_resource_info(desc, path, icon, h_text)
    print("", file=sys.stderr)


if __name__ == "__main__":
    main()