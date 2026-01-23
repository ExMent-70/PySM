# report_renderer_standard.py

import sys
from pathlib import Path
from typing import List

# --- Импорты из ядра PySM ---
try:
    from pysm_lib.pysm_theme_api import theme_api
    from pysm_lib.pysm_icons import icons
except ImportError:
    pass

# Импортируем функции сканирования
from report_common import (
    ResourceNode, 
    scan_directory_for_extensions, 
    scan_subfolders, 
    scan_analysis_structure
)

# --- ФУНКЦИЯ-ФАБРИКА ---
def generate_standard_html(
    config, 
    path_session_base, 
    path_psd_base, 
    path_c1_session, 
    session_name, 
    photo_session, 
    children_file_name, 
    wf_idsgn_catalog_str, 
    wf_portrait_session
) -> str:
    """Логика формирования стандартного древовидного отчета."""
    renderer = StandardTreeRenderer(icon_size=config.icon_size_tree)

    # Блок 1
    renderer.start_section("Блок 1: Глобальные ресурсы")
    global_nodes = []
    if path_session_base:
        global_nodes.append(ResourceNode("RAW Base", path_session_base, "folder", "Корневая папка для хранения исходных RAW-файлов со всех съёмок"))
    if path_psd_base:
        global_nodes.append(ResourceNode("Albums Base", path_psd_base, "folder", "Корневая папка для хранения рабочих материалов всех выпускных альбомов"))
    
    if wf_idsgn_catalog_str:
        cat_path = Path(wf_idsgn_catalog_str)
        cat_node = ResourceNode("Каталог шаблонов", cat_path.parent, "folder", f"Файл: {cat_path.name}")
        if cat_path.parent.exists():
             cat_node.children = scan_directory_for_extensions(cat_path.parent, ['.idml', '.indd'], 'indd')
        global_nodes.append(cat_node)
    
    renderer.render_tree(global_nodes)
    renderer.end_section()

    # Блок 2
    if path_c1_session:
        renderer.start_section("Блок 2: Исходные RAW-файлы. AI-анализ фотографий")
        
        c1_nodes = [
            ResourceNode(session_name, path_c1_session, "folder", "Базовая папка для исходных файлов текущего класса"),
            ResourceNode(f"{session_name}.cosessiondb", path_c1_session / f"{session_name}.cosessiondb", "c1", "Файл-сессия программы Capture One"),
            ResourceNode(f"{photo_session}_{children_file_name}", path_c1_session / f"{photo_session}_{children_file_name}", "txt", "Список фотографируемых")
        ]

        if wf_portrait_session and path_session_base:
            c1_nodes.append(ResourceNode("Портретная сессия", path_session_base / wf_portrait_session, "folder", f"Идентификация: {wf_portrait_session}"))

        capture_path = path_c1_session / "Capture"
        capture_node = ResourceNode("Capture", capture_path, "folder")
        if capture_path.exists():
            capture_node.children = scan_subfolders(capture_path)
        c1_nodes.append(capture_node)

        output_path = path_c1_session / "Output"
        output_node = ResourceNode("Output", output_path, "folder")
        
        target_session = photo_session if config.report_scope == "current" else ""
        
        if output_path.exists():
            output_node.children = scan_analysis_structure(output_path, config.report_scope, target_session)
        
        c1_nodes.append(output_node)
        
        selects_path = path_c1_session / "Selects"
        selects_node = ResourceNode("Selects", selects_path, "folder")
        if selects_path.exists():
            selects_node.children = scan_subfolders(selects_path)
        c1_nodes.append(selects_node)

        renderer.render_tree(c1_nodes)
        renderer.end_section()

    # Блок 3
    if path_psd_base and session_name:
        renderer.start_section("Блок 3: Работа с альбомами (PSD/InDesign)")
        work_path = path_psd_base / session_name
        
        psd_nodes = [
            ResourceNode(session_name, work_path, "folder", "Рабочая папка"),
        ]

        list_file = work_path / f"{session_name}.list"
        psd_nodes.append(ResourceNode(f"{session_name}.list", list_file, "code", "Файл списка"))

        psd_nodes.append(ResourceNode("Приложение (HTML)", work_path / f"{session_name}.html", "html", is_critical=False))

        grads_path = work_path / "Выпускникам"
        psd_nodes.append(ResourceNode("Выпускникам", grads_path, "folder", is_critical=False))

        photos_path = work_path / "Альбом" / "Фото"
        photos_node = ResourceNode("Фото (PSD)", photos_path, "folder")
        if photos_path.exists():
            photos_node.children = scan_subfolders(photos_path)
        psd_nodes.append(photos_node)

        pages_path = work_path / "Альбом" / "Готовые страницы"
        psd_nodes.append(ResourceNode("Готовые страницы", pages_path, "folder", is_critical=False))

        templates_path = work_path / "Альбом" / "_ШАБЛОНЫ_"
        tpl_node = ResourceNode("_ШАБЛОНЫ_", templates_path, "folder", "Папка шаблонов")
        if templates_path.exists():
            tpl_node.children = scan_directory_for_extensions(templates_path, ['.indd', '.idml'], 'indd')
        psd_nodes.append(tpl_node)
        
        renderer.render_tree(psd_nodes)
        renderer.end_section()

    return renderer.get_html()


class StandardTreeRenderer:
    """
    Рендерер для стандартного древовидного отчета (список).
    Поддерживает чередование цветов строк (Zebra striping).
    """
    def __init__(self, icon_size: int = 20):
        self.html_parts = []
        self.row_counter = 0
        self.icon_size = icon_size
        
        # --- Загрузка цветов из темы ---
        self.bg_color_base = self._get_theme_color("table_background_base", "color", "#ffffff")
        self.bg_color_alt = self._get_theme_color("table_background_alternate", "color", "#f9f9f9")
        self.header_bg = self._get_theme_color("collection_info", "background-color", "#ecf0f1")
        if self.header_bg == "transparent":
            self.header_bg = "#ecf0f1"

        self.text_main = self._get_theme_color("script_stdout", "color", "#2c3e50")
        self.text_sub = self._get_theme_color("runner_info", "color", "#7f8c8d")
        self.color_ok = self._get_theme_color("status_success", "color", "#27AE60")
        self.color_error = self._get_theme_color("status_error", "color", "#E74C3C")
        self.color_accent = self._get_theme_color("api_link", "color", "#3498DB")

    def _get_theme_color(self, style_name: str, css_prop: str, default: str) -> str:
        try:
            style_dict = theme_api.get_parsed_style(style_name)
            val = style_dict.get(css_prop)
            if val and (val.startswith("#") or val.startswith("rgb") or val.startswith("rgba")):
                return val
            val_fallback = style_dict.get("color")
            if val_fallback and (val_fallback.startswith("#") or val_fallback.startswith("rgb")):
                return val_fallback
        except NameError:
            pass # Если theme_api недоступен
        return default

    def start_section(self, title: str):
        self.row_counter = 0
        header_style = (
            f"font-family: sans-serif; font-size: 16px; font-weight: bold; "
            f"color: {self.text_main}; margin-top: 20px; margin-bottom: 5px; "
            f"padding-bottom: 5px; border-bottom: 1px solid {self.color_accent};"
        )
        self.html_parts.append(f'<div style="{header_style}">{title}</div>')
        
        th_style = (
            f"padding: 8px; border: none; font-weight: bold; "
            f"color: {self.text_main}; opacity: 0.8;"
        )

        self.html_parts.append(
            f'<table style="width: 100%; border-spacing: 0; font-family: sans-serif; font-size: 13px;">'
            f'<tr style="background-color: {self.header_bg}; text-align: left;">'
            f'<th style="{th_style} width: 40px; border-top-left-radius: 4px;">Тип</th>'
            f'<th style="{th_style}">Ресурс / Структура</th>'
            f'<th style="{th_style} border-top-right-radius: 4px;">Состояние / Путь</th>'
            '</tr>'
        )

    def render_tree(self, nodes: List[ResourceNode], level: int = 0):
        for node in nodes:
            self._render_row(node, level)
            if node.children:
                self.render_tree(node.children, level + 1)

    def _render_row(self, node: ResourceNode, level: int):
        padding_left = 8 + (level * 25)
        # Получаем стрелку через API иконок
        try:
            prefix_icon = icons.ARROW_SUB(size=16) if level > 0 else ""
        except NameError:
            prefix_icon = ""
        
        # Zebra striping
        is_even = (self.row_counter % 2 == 0)
        row_bg = self.bg_color_base if is_even else self.bg_color_alt
        self.row_counter += 1
        
        status_color = self.text_sub
        status_text = node.description
        path_str = str(node.path)

        if node.exists:
            name_style = f"color: {self.text_main}; font-weight: bold;" if level == 0 else f"color: {self.text_main}; opacity: 0.9;"
            if not status_text:
                status_text = "OK"
                status_color = self.color_ok
        else:
            name_style = f"color: {self.color_error};"
            status_text = "НЕ НАЙДЕНО"
            status_color = self.color_error
            if not node.is_critical:
                 name_style = f"color: {self.text_sub}; opacity: 0.7;"
                 status_text = "Отсутствует"
        
        try:
            link_href = node.path.resolve().as_uri()
        except Exception:
            link_href = "#"

        icon_tag = node.get_icon_html(size=self.icon_size)
        td_style = "padding: 6px; vertical-align: middle; border: none;"

        row_html = (
            f'<tr style="background-color: {row_bg};">'
            f'<td style="{td_style} text-align: center;">'
            f'<a href="{link_href}" style="text-decoration: none;">{icon_tag}</a>'
            f'</td>'
            f'<td style="{td_style} padding-left: {padding_left}px;">'
            f'{prefix_icon}'
            f'<a href="{link_href}" style="{name_style} text-decoration: none;">{node.name}</a>'
            f'</td>'
            f'<td style="{td_style} color: {status_color};">'
            f'<b>{status_text}</b><br>'
            f'<span style="font-size: 10px; color: {self.text_sub}; opacity: 0.8;">{path_str}</span>'
            f'</td>'
            f'</tr>'
        )
        self.html_parts.append(row_html)

    def end_section(self):
        self.html_parts.append('</table><br>')

    def get_html(self) -> str:
        return "".join(self.html_parts)