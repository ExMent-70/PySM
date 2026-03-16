# pysm_lib/report_api.py

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path

try:
    from .pysm_theme_api import theme_api
    from .pysm_icons import icons
except ImportError:
    # Заглушки для автономного тестирования
    theme_api = None
    icons = None

# --- 1. МОДЕЛЬ ДАННЫХ ---

@dataclass
class ResourceNode:
    """
    Базовый элемент для построения отчетов PySM.
    """
    name: str
    path: Optional[Path]
    type: str  # 'folder', 'file', 'code', 'html', 'c1', 'img', 'db', 'archive'
    description: str = ""
    is_critical: bool = True
    children: List['ResourceNode'] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        if self.path is None: return False
        return self.path.exists()

    def get_icon_html(self, size: int = 20) -> str:
        """Возвращает HTML иконки."""
        if not icons: return ""
        
        # Если файла нет и он критичен - ошибка
        if self.path and not self.exists and self.is_critical:
            return icons.ERROR(size=size)
        
        t = self.type.lower()
        mapping = {
            'folder': icons.FOLDER, 'file': icons.FILE, 'c1': getattr(icons, 'FILE_C1', icons.FILE),
            'html': icons.FILE_HTML, 'indd': icons.FILE_INDD, 'db': icons.FILE_DB,
            'img': icons.FILE_IMAGE, 'archive': icons.FILE_ARCHIVE, 'code': icons.FILE_CODE,
            'txt': getattr(icons, 'FILE_TXT', icons.FILE)
        }
        
        method = mapping.get(t, icons.FILE)
        return method(size=size)

    def find_child_by_name(self, name_part: str) -> Optional['ResourceNode']:
        """Помощник для поиска дочернего узла по части имени."""
        if not self.children:
            return None
        for child in self.children:
            if name_part.lower() in child.name.lower():
                return child
        return None

# --- 2. УПРАВЛЕНИЕ ТЕМОЙ ---

class ReportTheme:
    """Помощник для получения цветов из theme.toml"""
    def __init__(self):
        self.bg_base = self._get("table_background_base", "color", "#ffffff")
        self.bg_alt = self._get("table_background_alternate", "color", "#f9f9f9")
        self.header_bg = self._get("collection_info", "background-color", "#ecf0f1")
        if self.header_bg == "transparent": self.header_bg = "#ecf0f1"
        
        self.text_main = self._get("script_stdout", "color", "#2c3e50")
        self.text_sub = self._get("runner_info", "color", "#7f8c8d")
        
        self.border = self._get("script_info", "color", "#95a5a6")
        self.accent = self._get("api_link", "color", "#3498DB")
        
        self.ok = self._get("status_success", "color", "#27AE60")
        self.error = self._get("status_error", "color", "#E74C3C")

    def _get(self, style_name, prop, default):
        if not theme_api: return default
        try:
            d = theme_api.get_parsed_style(style_name)
            val = d.get(prop) or d.get("color")
            if val and (val.startswith("#") or val.startswith("rgb")): return val
        except: pass
        return default

# --- 3. БАЗОВЫЙ РЕНДЕРЕР ---

class BaseReportBuilder:
    def __init__(self):
        self.parts = []
        self.theme = ReportTheme()

    def get_html(self) -> str:
        return "".join(self.parts)

    def _render_link(self, node: ResourceNode, content: str) -> str:
        href = node.path.resolve().as_uri() if (node.path) else "#"
        style = f"text-decoration: none; color: {self.theme.text_main};"
        if node.path and not node.exists:
            style = "text-decoration: line-through; color: #999;"
        return f'<a href="{href}" style="{style} display: block;">{content}</a>'

# --- 4. РЕНДЕРЕР ДЕРЕВА (Standard) ---

class StandardTreeBuilder(BaseReportBuilder):
    def __init__(self, icon_size: int = 20):
        super().__init__()
        self.icon_size = icon_size
        self._row_counter = 0

    def add_section(self, title: str, root_nodes: List[ResourceNode]):
        """Добавляет секцию с заголовком и деревом файлов."""
        self._row_counter = 0
        self._add_header(title)
        self._start_table()
        self._render_nodes_recursive(root_nodes, 0)
        self._end_table()

    def _add_header(self, text):
        style = (f"font-family: sans-serif; font-size: 16px; font-weight: bold; "
                 f"color: {self.theme.text_main}; margin-top: 20px; margin-bottom: 5px; "
                 f"padding-bottom: 5px; border-bottom: 1px solid {self.theme.accent};")
        self.parts.append(f'<div style="{style}">{text}</div>')

    def _start_table(self):
        th_style = f"padding: 8px; border: none; font-weight: bold; color: {self.theme.text_main}; opacity: 0.8;"
        self.parts.append(
            f'<table style="width: 100%; border-spacing: 0; font-family: sans-serif; font-size: 13px;">'
            f'<tr style="background-color: {self.theme.header_bg}; text-align: left;">'
            f'<th style="{th_style} width: 40px; border-top-left-radius: 4px;">Тип</th>'
            f'<th style="{th_style}">Ресурс</th>'
            f'<th style="{th_style} border-top-right-radius: 4px;">Путь / Статус</th></tr>'
        )

    def _end_table(self):
        self.parts.append('</table><br>')

    def _render_nodes_recursive(self, nodes: List[ResourceNode], level: int):
        for node in nodes:
            self._render_row(node, level)
            if node.children:
                self._render_nodes_recursive(node.children, level + 1)

    def _render_row(self, node: ResourceNode, level: int):
        pad_left = 8 + (level * 25)
        arrow = icons.ARROW_SUB(size=16) if (level > 0 and icons) else ""
        bg = self.theme.bg_base if (self._row_counter % 2 == 0) else self.theme.bg_alt
        self._row_counter += 1

        status_text = node.description
        status_color = self.theme.text_sub
        
        path_str = str(node.path) if node.path else ""
        
        name_style = "" 
        if node.exists:
            name_style = f"font-weight: bold;" if level == 0 else ""
            if not status_text:
                status_text = "OK"; status_color = self.theme.ok
        else:
            name_style = f"color: {self.theme.error};"
            status_text = "НЕ НАЙДЕНО"; status_color = self.theme.error
            if not node.is_critical:
                status_text = "Отсутствует"; status_color = self.theme.text_sub
                name_style = f"color: {self.theme.text_sub};"

        icon = node.get_icon_html(self.icon_size)
        href = node.path.resolve().as_uri() if node.path else "#"
        
        td_style = "padding: 6px; vertical-align: middle; border: none;"
        
        row = (
            f'<tr style="background-color: {bg};">'
            f'<td style="{td_style} text-align: center;"><a href="{href}">{icon}</a></td>'
            f'<td style="{td_style} padding-left: {pad_left}px;">{arrow} '
            f'<a href="{href}" style="text-decoration:none; color:{self.theme.text_main}; {name_style}">{node.name}</a></td>'
            f'<td style="{td_style} color: {status_color};"><b>{status_text}</b><br>'
            f'<span style="font-size: 10px; color: {self.theme.text_sub};">{path_str}</span></td></tr>'
        )
        self.parts.append(row)

# --- 5. РЕНДЕРЕР DASHBOARD (Grid) ---

class DashboardBuilder(BaseReportBuilder):
    def __init__(self, icon_size: int = 32):
        super().__init__()
        self.icon_size = icon_size
        self.table_style = f"border-collapse: collapse; font-family: sans-serif; font-size: 13px; background-color: {self.theme.bg_base}; margin-bottom: 5px;"
        
        # ИЗМЕНЕНИЕ: vertical-align: top (для выравнивания по верху)
        self.td_style = f"border: 0px solid {self.theme.border}; padding: 4px; vertical-align: top; color: {self.theme.text_main};"

    def add_header_simple(self, text: str, link_node: Optional[ResourceNode] = None):
        """Простой заголовок (подчеркнутый)."""
        self._render_header_html(text, link_node, mode="simple")

    def add_header_boxed(self, text: str, link_node: Optional[ResourceNode] = None, extra_html: str = ""):
        """Заголовок в рамке (как у сессии или подраздела)."""
        self._render_header_html(text, link_node, mode="boxed", extra_html=extra_html)

    def add_table_simple(self, nodes: List[ResourceNode]):
        """Таблица в одну строку. Ширина по контенту."""
        self._start_table_tag(width=None)
        self.parts.append('<tr>')
        for node in nodes:
            self.parts.append(self._render_cell(node))
        self.parts.append('</tr></table>')

    def add_table_matrix(self, rows_of_nodes: List[List[Optional[ResourceNode]]]):
        """Матрица. Ширина по контенту."""
        self._start_table_tag(width=None, no_top_margin=True)
        for row in rows_of_nodes:
            self.parts.append('<tr>')
            for node in row:
                self.parts.append(self._render_cell(node))
            self.parts.append('</tr>')
        self.parts.append('</table>')

    def add_grid(self, nodes: List[ResourceNode], columns: int = 5):
        """Сетка. Ширина по контенту."""
        self._start_table_tag(width=None, no_top_margin=True)
        count = len(nodes)
        rows = math.ceil(count / columns)
        for r in range(rows):
            self.parts.append('<tr>')
            for c in range(columns):
                idx = r * columns + c
                if idx < count:
                    self.parts.append(self._render_cell(nodes[idx]))
                else:
                    self.parts.append(f'<td style="{self.td_style} border: none;"></td>')
            self.parts.append('</tr>')
        self.parts.append('</table>')

    def add_list_zebra(self, nodes: List[ResourceNode]):
        """Список (зебра). Ширина 100%."""
        self._start_table_tag(width="100%", no_top_margin=True)
        small_icon = int(self.icon_size * 0.8)
        
        for i, node in enumerate(nodes):
            bg = self.theme.bg_base if i % 2 == 0 else self.theme.bg_alt
            icon = node.get_icon_html(small_icon)
            href = node.path.resolve().as_uri() if node.path else "#"
            
            link_style = f"text-decoration: none; color: {self.theme.text_main};"
            if node.path and not node.exists: link_style = "text-decoration: line-through; color: #999;"

            row = (
                f'<tr style="background-color: {bg}; border-bottom: 1px solid {self.theme.border};">'
                f'<td width="{small_icon + 12}" align="center" style="padding: 4px;">'
                f'<a href="{href}" style="text-decoration: none; display:block;">{icon}</a></td>'
                f'<td width="100%" style="padding: 4px 10px; text-align: left;">'
                f'<a href="{href}" style="{link_style}">{node.name}</a></td></tr>'
            )
            self.parts.append(row)
        self.parts.append('</table>')

    # --- Внутренние методы ---

    def _start_table_tag(self, width="100%", no_top_margin=False):
        margin = "0" if no_top_margin else "5px"
        width_attr = f'width="{width}"' if width else ''
        self.parts.append(f'<table {width_attr} style="{self.table_style} margin-top: {margin};">')

    def _render_header_html(self, text, node, mode, extra_html=""):
        href = node.path.resolve().as_uri() if (node and node.path) else "#"
        icon = ""
        if node and node.path and icons:
            icon = f'<span style="margin-right: 10px;">{icons.OPEN(size=16)}</span>&nbsp;'
        
        content = f'{icon}<a href="{href}" style="text-decoration:none; color:{self.theme.text_main};">{text}</a>'
        
        if mode == "boxed":
            style = (f"border: 1px solid {self.theme.border}; border-bottom: none; "
                     f"background-color: {self.theme.header_bg}; padding: 5px 8px; "
                     f"font-weight: bold; color: {self.theme.text_main}; margin-top: 12px; display: flex; align-items: center;")
            self.parts.append(f'<div style="{style}">{content} {extra_html}</div>')
        else:
            style = (f"font-size: 16px; font-weight: bold; color: {self.theme.text_main}; "
                     f"margin-top: 12px; margin-bottom: 5px; border-bottom: 2px solid {self.theme.accent}; padding-bottom: 5px;")
            self.parts.append(f'<div style="{style}">{content} {extra_html}</div>')

    def _render_cell(self, node: Optional[ResourceNode]) -> str:
        if not node:
            return f'<td style="{self.td_style} background-color: #f2f2f2;">-</td>'
        
        icon = node.get_icon_html(self.icon_size)
        name = f'<div style="font-size: 12px; margin-top: 2px;">{node.name}</div>'
        content = self._render_link(node, f'{icon}{name}')
        # Центрирование по горизонтали, по вертикали берется из self.td_style (top)
        return f'<td style="{self.td_style} text-align: center;">{content}</td>'