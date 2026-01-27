# report_renderer_dashboard.py

import sys
import math
from typing import List, Optional
from pathlib import Path

# --- Импорты из ядра PySM ---
try:
    from pysm_lib.pysm_theme_api import theme_api
    from pysm_lib.pysm_icons import icons
except ImportError:
    pass

# Импортируем общие модели и функции сканирования
from report_common import (
    ResourceNode, 
    scan_directory_for_extensions, 
    scan_subfolders, 
    scan_analysis_structure,
    check_xmp_presence
)

# --- ФУНКЦИЯ-ФАБРИКА (ВЕРНУЛ НА МЕСТО) ---
def generate_dashboard_html(
    config, 
    path_session_base, 
    path_psd_base, 
    path_c1_session, 
    session_name, 
    photo_session, 
    children_file_name, 
    wf_portrait_session, 
    wf_idsgn_catalog_str
) -> str:
    """Собирает данные и генерирует HTML для Dashboard отчета."""
    
    renderer = DashboardRenderer(icon_size=config.icon_size_dashboard)

    # 1. Глобальные папки
    raw_node = ResourceNode("RAW Base", path_session_base, "folder") if path_session_base else None
    psd_node = ResourceNode("PSD Base", path_psd_base, "folder") if path_psd_base else None
    cat_node = None
    if wf_idsgn_catalog_str:
        cat_path = Path(wf_idsgn_catalog_str)
        cat_node = ResourceNode("Каталог<br>шаблонов", cat_path.parent, "folder", f"{cat_path.name}")

    renderer.render_global_block(raw_node, psd_node, cat_node)

    if not path_c1_session:
        return renderer.get_html()

    # 2. Проект (Capture One Session)
    root_session_node = ResourceNode(session_name, path_c1_session, "folder", "Корень")
    project_nodes = [
        ResourceNode("Capture", path_c1_session / "Capture", "folder"),
        ResourceNode("Output", path_c1_session / "Output", "folder"),
        ResourceNode("Selects", path_c1_session / "Selects", "folder"),
        ResourceNode("Сессия<br>Capture One", path_c1_session / f"{session_name}.cosessiondb", "c1") 
    ]
    renderer.render_project_block(root_session_node, project_nodes)

    # 3. Сессии (Analysis blocks)
    output_path = path_c1_session / "Output"
    target_session = photo_session if config.report_scope == "current" else ""
    
    analysis_nodes = scan_analysis_structure(output_path, config.report_scope, target_session)
    
    for analysis_node in analysis_nodes:
        current_session_suffix = analysis_node.name.replace("Analysis_", "")
        
        capture_path_base = path_c1_session / "Capture"
        has_xmp = check_xmp_presence(capture_path_base, current_session_suffix)
        
        session_folder_node = ResourceNode(
            name=f"Фотосессия {current_session_suffix}: файлы RAW",
            path=capture_path_base / current_session_suffix,
            type="folder",
            meta={"has_xmp": has_xmp}
        )

        # Переименование для отображения
        analysis_node.name = "AI-анализ"
        if analysis_node.children:
            for child in analysis_node.children:
                if child.name == "info_group_faces.json": 
                    child.name = "Групповые<br>фото (JSON)"
                elif child.name == "info_portrait_faces.json": 
                    child.name = "Портретные<br>фото (JSON)"
                elif child.name == "matches_portrait_to_group.json": 
                    child.name = "Портрет-<br>Группа (JSON)"  # <--- НОВОЕ ИМЯ
                elif child.name == "error_matches.json":
                    child.name = "Ошибки<br>(JSON)"
                elif child.name == "face_clustering_report.html": 
                    child.name = "HTML<br>отчет"

        child_fname = f"{current_session_suffix}_{children_file_name}"
        child_file_node = ResourceNode(name=child_fname, path=path_c1_session / child_fname, type="txt")

        renderer.render_session_block(session_folder_node, analysis_node, child_file_node)

    # 4. Работа с альбомами
    if path_psd_base and session_name:
        work_path = path_psd_base / session_name
        
        # Основные узлы
        list_file = work_path / f"{session_name}.list"
        main_nodes = [
            ResourceNode("Список класса<br>(JSON)", list_file, "code"),
            ResourceNode("Договор<br>(HTML)", work_path / f"{session_name}.html", "html", is_critical=False),
            ResourceNode("Выпускникам<br>(файлы JPG)", work_path / "Выпускникам", "folder", is_critical=False),
            ResourceNode("В печать<br>(файлы JPG)", work_path / "Альбом" / "Готовые страницы", "folder", is_critical=False)
        ]

        # PSD
        photos_path = work_path / "Альбом" / "Фото"
        photos_node = ResourceNode("Фото (PSD)", photos_path, "folder")
        psd_subfolders = []
        if photos_path.exists():
            psd_subfolders = scan_subfolders(photos_path)

        # Шаблоны
        templates_path = work_path / "Альбом" / "_ШАБЛОНЫ_"
        tpl_node = ResourceNode("Шаблоны", templates_path, "folder")
        tpl_files = []
        if templates_path.exists():
            tpl_files = scan_directory_for_extensions(templates_path, ['.indd', '.idml'], 'indd')

        renderer.render_albums_block(main_nodes, photos_node, psd_subfolders, tpl_node, tpl_files)

    return renderer.get_html()


# --- КЛАСС РЕНДЕРЕРА ---
class DashboardRenderer:
    """
    Рендерер для блочного (Dashboard) отчета.
    Формирует таблицы с рамками и плиточным расположением элементов.
    """
    def __init__(self, icon_size: int = 32):
        self.html_parts = []
        self.icon_size = icon_size
        
        # --- НАСТРОЙКИ ОТСТУПОВ (КОНСТАНТЫ) ---
        self.spacing_block_top = "12px"     
        self.spacing_table_bottom = "5px"   
        self.cell_padding = "4px"           
        self.icon_text_gap = "2px"          
        self.header_padding = "5px 8px"     
        # --------------------------------------

        # Цвета из темы
        self.bg_color = self._get_theme_color("table_background_base", "color", "#ffffff")
        # Цвет для зебры
        self.bg_color_alt = self._get_theme_color("table_background_alternate", "color", "#f9f9f9")
        
        self.text_main = self._get_theme_color("script_stdout", "color", "#2c3e50")
        self.text_sub = self._get_theme_color("runner_info", "color", "#7f8c8d")
        self.border_color = self._get_theme_color("script_info", "color", "#95a5a6")
        self.color_accent = self._get_theme_color("api_link", "color", "#3498DB")
        self.color_header_bg = self._get_theme_color("collection_info", "background-color", "#ecf0f1")
        if self.color_header_bg == "transparent": self.color_header_bg = "#ecf0f1"

    def _get_theme_color(self, style_name: str, css_prop: str, default: str) -> str:
        try:
            style_dict = theme_api.get_parsed_style(style_name)
            val = style_dict.get(css_prop)
            if val and (val.startswith("#") or val.startswith("rgb")): return val
            val_fb = style_dict.get("color")
            if val_fb and (val_fb.startswith("#") or val_fb.startswith("rgb")): return val_fb
        except NameError:
            pass
        return default

    def _style_table(self, no_top_margin: bool = False) -> str:
        margin_top = "0" if no_top_margin else "5px"
        return (
            f"width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px; "
            f"background-color: {self.bg_color}; margin-bottom: {self.spacing_table_bottom}; margin-top: {margin_top};"
        )

    def _style_td(self, align="center") -> str:
        return (
            f"border: 0px solid {self.border_color}; "
            f"padding: {self.cell_padding}; "
            f"vertical-align: top; text-align: {align}; color: {self.text_main};"
        )

    def _render_header(self, text: str, link: Optional[str] = None, mode: str = "simple", extra_html: str = ""):
        icon_html = ""
        if link and link != "#":
            try:
                icon_svg = icons.OPEN(size=16)
                icon_html = f'<span style="margin-right: 8px; vertical-align: middle;">{icon_svg}</span>&nbsp;'
            except:
                pass 

        content = f"{icon_html}<span style='vertical-align: middle;'>{text}</span>"
        
        if link:
            content = f'<a href="{link}" style="text-decoration: none; color: {self.text_main};">{content}</a>'

        if mode == "boxed":
            style = (
                f"border: 1px solid {self.border_color}; "
                f"border-bottom: none; "
                f"background-color: {self.color_header_bg}; "
                f"padding: {self.header_padding}; "
                f"font-family: sans-serif; font-size: 14px; font-weight: bold; "
                f"color: {self.text_main}; "
                f"margin-bottom: 0; "
                f"margin-top: {self.spacing_block_top}; "
                f"display: flex; align-items: center;"
            )
            self.html_parts.append(f'<div style="{style}">{content} {extra_html}</div>')
        else:
            style = (
                f"font-family: sans-serif; font-size: 16px; font-weight: bold; "
                f"color: {self.text_main}; margin-bottom: 2px; "
                f"margin-top: {self.spacing_block_top}; "
                f"border-bottom: 2px solid {self.color_accent}; padding-bottom: 2px;"
            )
            self.html_parts.append(f'<div style="{style}">{content} {extra_html}</div>')

    def render_global_block(self, raw_node: Optional[ResourceNode], psd_node: Optional[ResourceNode], cat_node: Optional[ResourceNode]):
        self._render_header("Глобальные ресурсы", mode="simple")
        table_html = f'<table style="{self._style_table()}">'
        table_html += f'<tr>'
        table_html += self._render_large_cell(raw_node)
        table_html += self._render_large_cell(psd_node)
        if cat_node:
            table_html += self._render_large_cell(cat_node)
        table_html += f'</tr></table>'
        self.html_parts.append(table_html)

    def render_project_block(self, root_node: ResourceNode, nodes: List[ResourceNode]):
        session_name = root_node.name
        try:
            link = root_node.path.resolve().as_uri()
        except:
            link = "#"
        
        self._render_header(f"Блок 2: Исходные RAW-файлы. AI-анализ фотографий", link, mode="simple")

        if not nodes: return

        table_html = f'<table style="{self._style_table()}">'
        table_html += '<tr>'
        for node in nodes:
            table_html += self._render_large_cell(node)
        table_html += '</tr></table>'
        self.html_parts.append(table_html)

    def render_session_block(self, session_node: ResourceNode, analysis_node: ResourceNode, children_file_node: Optional[ResourceNode]):
        xmp_info = ""
        if session_node and session_node.meta.get("has_xmp"):
            try:
                xmp_text = f'<span style="color: #27AE60; font-weight: bold; margin-left: 10px;">+ XMP</span>'
            except:
                xmp_text = " (XMP)"
            xmp_info = xmp_text

        session_name = session_node.name if session_node else "Unknown Session"
        session_link = session_node.path.resolve().as_uri() if session_node else "#"
        
        self._render_header(session_name, session_link, mode="boxed", extra_html=xmp_info)

        jpg_node = analysis_node.find_child_by_name("JPG")
        masks_node = analysis_node.find_child_by_name("Masks")
        json_group = analysis_node.find_child_by_name("Групповые")
        json_portrait = analysis_node.find_child_by_name("Портрет")
        json_matches = analysis_node.find_child_by_name("Портрет-")
        json_errors = analysis_node.find_child_by_name("Ошибки")
        html_report = analysis_node.find_child_by_name("HTML")

        table_html = f'<table style="{self._style_table(no_top_margin=True)}">' 
        table_html += '<tr>'
        table_html += self._render_large_cell(analysis_node)
        table_html += self._render_large_cell(jpg_node)
        table_html += self._render_large_cell(masks_node)
        table_html += self._render_large_cell(html_report)
        table_html += self._render_large_cell(children_file_node)        
        table_html += '</tr>'
        table_html += '<tr>'
        table_html += self._render_large_cell(json_group)
        table_html += self._render_large_cell(json_portrait)
        table_html += self._render_large_cell(json_matches)   
        table_html += self._render_large_cell(json_errors)        
        table_html += '</tr></table>'
        self.html_parts.append(table_html)

    def render_albums_block(self, main_nodes: List[ResourceNode], photos_node: ResourceNode, psd_subfolders: List[ResourceNode], tpl_folder_node: ResourceNode, tpl_files: List[ResourceNode]):
        try:
            main_link = main_nodes[0].path.parent.as_uri()
        except:
            main_link = "#"
        
        self._render_header("Блок 3: Работа с альбомами (JPG/PSD/InDesign)", main_link, mode="simple")
        
        table_html = f'<table style="{self._style_table()}">'
        table_html += '<tr>'
        for node in main_nodes:
            table_html += self._render_large_cell(node)
        table_html += '</tr></table>'
        self.html_parts.append(table_html)

        try:
            psd_link = photos_node.path.as_uri()
        except:
            psd_link = "#"
        
        self._render_header("Файлы PSD (сгруппированы по фотосессиям и сюжетам)", psd_link, mode="boxed")
        
        if psd_subfolders:
            self._render_grid_section(psd_subfolders, columns=5, no_top_margin=True)
        else:
            self.html_parts.append(f'<div style="border: 1px solid {self.border_color}; border-top: none; padding: {self.cell_padding}; color:{self.text_sub}; font-style: italic;">Папка пуста</div>')

        try:
            tpl_link = tpl_folder_node.path.as_uri()
        except:
            tpl_link = "#"
            
        self._render_header("Развороты альбомов (файлы InDesign)", tpl_link, mode="boxed")
        
        if tpl_files:
            # Вывод списка файлов с зеброй
            self._render_list_section(tpl_files, no_top_margin=True)
        else:
            self.html_parts.append(f'<div style="border: 1px solid {self.border_color}; border-top: none; padding: {self.cell_padding}; color:{self.text_sub}; font-style: italic;">Шаблоны не найдены</div>')


    def _render_grid_section(self, nodes: List[ResourceNode], columns: int = 5, no_top_margin: bool = False):
        table_html = f'<table style="{self._style_table(no_top_margin)}">'
        count = len(nodes)
        rows = math.ceil(count / columns)
        for r in range(rows):
            table_html += '<tr>'
            for c in range(columns):
                idx = r * columns + c
                if idx < count:
                    table_html += self._render_large_cell(nodes[idx])
                else:
                    table_html += f'<td style="{self._style_td()} border: none;"></td>'
            table_html += '</tr>'
        table_html += '</table>'
        self.html_parts.append(table_html)


    def _render_list_section(self, nodes: List[ResourceNode], no_top_margin: bool = False):
            """Рендерит список узлов построчно (зебра)."""
            table_html = f'<table width="100%" style="{self._style_table(no_top_margin)}">'
            
            # Вычисляем уменьшенный размер иконки (80% от базового)
            list_icon_size = int(self.icon_size * 0.8)
            
            # Ширина колонки подстраивается под уменьшенную иконку
            icon_col_width = list_icon_size + 12

            for i, node in enumerate(nodes):
                row_bg = self.bg_color if i % 2 == 0 else self.bg_color_alt
                
                try:
                    href = node.path.resolve().as_uri()
                except:
                    href = "#"

                # Иконка уменьшенного размера
                icon = node.get_icon_html(size=list_icon_size)
                name = node.name
                
                link_style = f"text-decoration: none; color: {self.text_main};"
                if not node.exists:
                     link_style = "text-decoration: line-through; color: #999;"

                row_style = f"background-color: {row_bg}; border-bottom: 1px solid {self.border_color};"
                row_html = f'<tr style="{row_style}">'
                
                # 1. Иконка (Фиксированная ширина)
                row_html += f'<td width="{icon_col_width}" align="center" style="padding: 4px; vertical-align: middle;">'
                row_html += f'<a href="{href}" style="text-decoration: none; display: block;">{icon}</a>'
                row_html += f'</td>'
                
                # 2. Текст (Растягивается)
                row_html += f'<td width="100%" style="padding: 4px 10px; vertical-align: middle; text-align: left;">'
                row_html += f'<a href="{href}" style="{link_style}">{name}</a>'
                row_html += f'</td>'
                
                row_html += '</tr>'
                table_html += row_html
                
            table_html += '</table>'
            self.html_parts.append(table_html)


    def _render_large_cell(self, node: Optional[ResourceNode]) -> str:
        if not node:
            return f'<td style="{self._style_td()} background-color: #f2f2f2;">-</td>'
        
        try:
            href = node.path.resolve().as_uri()
        except:
            href = "#"

        icon = node.get_icon_html(size=self.icon_size)
        name_html = f'<div style="word-wrap: break-word; font-size: 12px; margin-top: {self.icon_text_gap};">{node.name}</div>'
        
        link_style = f"text-decoration: none; color: {self.text_main};"
        if not node.exists:
             link_style = "text-decoration: line-through; color: #999;"

        cell_content = (
            f'<a href="{href}" style="{link_style} display: block;">'
            f'{icon}'
            f'{name_html}'
            f'</a>'
        )
        return f'<td style="{self._style_td()}">{cell_content}</td>'

    def get_html(self) -> str:
        return "".join(self.html_parts)