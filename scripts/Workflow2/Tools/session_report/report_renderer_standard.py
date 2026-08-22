# report_renderer_standard.py

from pathlib import Path
from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder

# Импортируем функции сканирования из общего модуля
from report_common import (
    scan_directory_for_extensions, 
    scan_subfolders, 
    scan_analysis_structure
)

def generate_standard_html(
    config, 
    path_session_base, 
    path_psd_base, 
    path_c1_session, 
    session_name, 
    photo_session, 
    wf_idsgn_catalog_str, 
    wf_portrait_session
) -> str:
    """Логика формирования стандартного древовидного отчета."""
    
    # Инициализация билдера из API
    builder = StandardTreeBuilder(icon_size=config.icon_size_tree)

    # --- Блок 1: Глобальные ресурсы ---
    global_nodes = []
    if path_session_base:
        global_nodes.append(ResourceNode("RAW Base", path_session_base, "folder", "Корневая папка RAW"))
    if path_psd_base:
        global_nodes.append(ResourceNode("Albums Base", path_psd_base, "folder", "Корневая папка Альбомов"))
    if wf_idsgn_catalog_str:
        cat_path = Path(wf_idsgn_catalog_str)
        cat_node = ResourceNode("Каталог шаблонов", cat_path.parent, "folder", f"Файл: {cat_path.name}")
        if cat_path.parent.exists():
             cat_node.children = scan_directory_for_extensions(cat_path.parent, ['.idml', '.indd'], 'indd')
        global_nodes.append(cat_node)
    
    # Добавляем секцию в отчет
    builder.add_section("Блок 1: Глобальные ресурсы", global_nodes)

    # --- Блок 2: Capture One ---
    if path_c1_session:
        c1_nodes = [
            ResourceNode(session_name, path_c1_session, "folder", "Папка сессии"),
            ResourceNode(f"{session_name}.cosessiondb", path_c1_session / f"{session_name}.cosessiondb", "c1", "Файл сессии"),
        ]

        if wf_portrait_session and path_session_base:
            c1_nodes.append(ResourceNode("Портретная сессия", path_session_base / wf_portrait_session, "folder", f"ID: {wf_portrait_session}"))

        # Capture
        capture_path = path_c1_session / "Capture"
        capture_node = ResourceNode("Capture", capture_path, "folder")
        if capture_path.exists():
            capture_node.children = scan_subfolders(capture_path)
        c1_nodes.append(capture_node)

        # Output (с фильтрацией)
        output_path = path_c1_session / "Output"
        output_node = ResourceNode("Output", output_path, "folder")
        target_session = photo_session if config.report_scope == "current" else ""
        if output_path.exists():
            output_node.children = scan_analysis_structure(output_path, config.report_scope, target_session)
        c1_nodes.append(output_node)
        
        # Selects
        selects_path = path_c1_session / "Selects"
        selects_node = ResourceNode("Selects", selects_path, "folder")
        if selects_path.exists():
            selects_node.children = scan_subfolders(selects_path)
        c1_nodes.append(selects_node)

        builder.add_section("Блок 2: Исходные RAW-файлы. AI-анализ", c1_nodes)

    # --- Блок 3: Альбомы ---
    if path_psd_base and session_name:
        work_path = path_psd_base / session_name
        
        psd_nodes = [
            ResourceNode(session_name, work_path, "folder", "Рабочая папка"),
        ]
        
        # Основные файлы
        list_file = work_path / f"{session_name}.list"
        psd_nodes.append(ResourceNode(f"{session_name}.list", list_file, "code", "Файл списка"))
        psd_nodes.append(ResourceNode("Приложение (HTML)", work_path / f"{session_name}.html", "html", is_critical=False))
        psd_nodes.append(ResourceNode("Выпускникам", work_path / "Выпускникам", "folder", is_critical=False))

        # Фото
        photos_path = work_path / "Альбом" / "Фото"
        photos_node = ResourceNode("Фото (PSD)", photos_path, "folder")
        if photos_path.exists():
            photos_node.children = scan_subfolders(photos_path)
        psd_nodes.append(photos_node)

        # Готовые
        pages_path = work_path / "Альбом" / "Готовые страницы"
        psd_nodes.append(ResourceNode("Готовые страницы", pages_path, "folder", is_critical=False))

        # Шаблоны
        templates_path = work_path / "Альбом" / "_ШАБЛОНЫ_"
        tpl_node = ResourceNode("_ШАБЛОНЫ_", templates_path, "folder", "Папка шаблонов")
        if templates_path.exists():
            tpl_node.children = scan_directory_for_extensions(templates_path, ['.indd', '.idml'], 'indd')
        psd_nodes.append(tpl_node)
        
        builder.add_section("Блок 3: Работа с альбомами", psd_nodes)

    return builder.get_html()
