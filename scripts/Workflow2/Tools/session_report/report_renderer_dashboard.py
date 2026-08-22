# report_renderer_dashboard.py

from pysm_lib.pysm_report_api import ResourceNode, DashboardBuilder, icons
from report_common import scan_directory_for_extensions, scan_subfolders, scan_analysis_structure, check_xmp_presence
from pathlib import Path

def generate_dashboard_html(config, path_session_base, path_psd_base, path_c1_session, session_name, photo_session, wf_portrait_session, wf_idsgn_catalog_str) -> str:
    
    # 1. Создаем билдер
    builder = DashboardBuilder(icon_size=config.icon_size_dashboard)

    # 2. Глобальные ресурсы
    builder.add_header_simple("Блок 1. Общие ресурсы")
    raw = ResourceNode("RAW Base", path_session_base, "folder") if path_session_base else None
    psd = ResourceNode("PSD Base", path_psd_base, "folder") if path_psd_base else None
    cat = ResourceNode("Каталог<br>шаблонов", Path(wf_idsgn_catalog_str).parent, "folder") if wf_idsgn_catalog_str else None
    builder.add_table_simple([raw, psd, cat])

    if not path_c1_session: return builder.get_html()

    # 3. Проект
    root_node = ResourceNode(session_name, path_c1_session, "folder")
    builder.add_header_simple(f"Блок 2.  Исходные RAW-файлы. AI-анализ фотографий ({session_name})", root_node)
    
    proj_nodes = [
        ResourceNode("Capture", path_c1_session / "Capture", "folder"),
        ResourceNode("Output", path_c1_session / "Output", "folder"),
        ResourceNode("Selects", path_c1_session / "Selects", "folder"),
        ResourceNode("Сессия C1", path_c1_session / f"{session_name}.cosessiondb", "c1")
    ]
    builder.add_table_simple(proj_nodes)

    # 4. Сессии (Сложная таблица)
    output_path = path_c1_session / "Output"
    target = photo_session if config.report_scope == "current" else ""
    analysis_nodes = scan_analysis_structure(output_path, config.report_scope, target)

    for anode in analysis_nodes:
        suffix = anode.name.replace("Analysis_", "")
        has_xmp = check_xmp_presence(path_c1_session / "Capture", suffix)
        
        # Заголовок с XMP
        xmp_html = f'<span style="color: #27AE60; font-weight: bold; margin-left: 10px;">+ XMP</span>' if has_xmp else ""
        session_node = ResourceNode(f"Фотосессия {suffix}: файлы RAW", path_c1_session / "Capture" / suffix, "folder")
        builder.add_header_boxed(session_node.name, session_node, extra_html=xmp_html)

        # Собираем матрицу
        # Ряд 1: Папка Analysis, JPG, Маски, Отчет
        # Ряд 2: JSON файлы
        
        # Находим детей по именам (как раньше)
        n_jpg = anode.find_child_by_name("JPG")
        n_masks = anode.find_child_by_name("Masks")
        n_html = anode.find_child_by_name("html")
        if n_html: n_html.name = "HTML<br>отчет"
        
        n_json_face = anode.find_child_by_name("info_face")
        if n_json_face: n_json_face.name = "Информация<br>о лицах(JSON)"
             
        n_matches = anode.find_child_by_name("matches")
        if n_matches: n_matches.name = "Портрет-Группа<br>(JSON)"
        
        n_error = anode.find_child_by_name("error")
        if n_error: n_error.name = "Ошибки<br>идентификации<br>(JSON)"

        anode.name = "AI-анализ"

        rows = [
            [anode, n_jpg, n_masks, n_html],
            [n_json_face, n_matches, n_error] # None для пустой ячейки
        ]
        builder.add_table_matrix(rows)

    # 5. Альбомы
    if path_psd_base:
        work_path = path_psd_base / session_name
        builder.add_header_simple("Блок 3: Работа с альбомами (JPG/PSD/INDD)", ResourceNode("Work", work_path, "folder"))
        
        # Основные
        main_nodes = [
            ResourceNode("Список", work_path / f"{session_name}.list", "code"),
            ResourceNode("Договор", work_path / f"{session_name}.html", "html"),
            ResourceNode("Выпускникам", work_path / "Выпускникам", "folder"),
            ResourceNode("В печать", work_path / "Альбом" / "Готовые страницы", "folder")
        ]
        builder.add_table_simple(main_nodes)

        # PSD (Grid)
        psd_path = work_path / "Альбом" / "Фото"
        builder.add_header_boxed("Файлы PSD (сгруппированы по фотосессиям и сюжетам)", ResourceNode("PSD", psd_path, "folder"))
        if psd_path.exists():
            builder.add_grid(scan_subfolders(psd_path), columns=5)
        
        # Шаблоны (Zebra List)
        tpl_path = work_path / "Альбом" / "_ШАБЛОНЫ_"
        builder.add_header_boxed("Развороты альбомов (файлы InDesign)", ResourceNode("TPL", tpl_path, "folder"))
        if tpl_path.exists():
            files = scan_directory_for_extensions(tpl_path, ['.indd', '.idml'], 'indd')
            builder.add_list_zebra(files)

    return builder.get_html()
