# report_common.py

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union

# --- Настройка окружения и импортов ---
try:
    # Попытка импорта из текущего окружения (если скрипт рядом с pysm_lib)
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib.pysm_icons import icons
except ImportError:
    # Заглушка для локальной разработки без ядра PySM
    class MockIcons:
        def __getattr__(self, name): return lambda **kwargs: ""
    icons = MockIcons()


# --- Модели данных ---

@dataclass
class ResourceNode:
    """
    Универсальный узел для представления файла или папки в отчете.
    Используется как в древовидном, так и в табличном шаблоне.
    """
    name: str
    path: Path
    type: str  # 'folder', 'file', 'indd', 'html', 'db', 'img'
    description: str = ""
    is_critical: bool = True
    children: List['ResourceNode'] = field(default_factory=list)
    
    # Дополнительные метаданные (например, наличие XMP)
    meta: dict = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def get_icon_html(self, size: int = 20) -> str:
        """Возвращает HTML тег иконки через API pysm_icons."""
        
        # Если критичный файл отсутствует - показываем ошибку
        if not self.exists and self.is_critical:
            return icons.ERROR(size=size)
        
        # Маппинг типов на методы API иконок
        # Используем .upper() для надежности, если type передан в разном регистре
        t = self.type.lower()
        
        if t == 'folder':
            return icons.FOLDER(size=size)
        elif t == 'file':
            return icons.FILE(size=size)
        elif t == 'c1':
            return icons.FILE_C1(size=size)
        elif t == 'html':
            return icons.FILE_HTML(size=size)
        elif t == 'indd':
            return icons.FILE_INDD(size=size)
        elif t == 'db':
            return icons.FILE_DB(size=size)
        elif t == 'img':
            return icons.FILE_IMAGE(size=size)
        elif t == 'archive':
            return icons.FILE_ARCHIVE(size=size)
        elif t == 'code':
            return icons.FILE_CODE(size=size)
        elif t == 'txt':
            return icons.FILE_TXT(size=size)
        
        # Fallback
        return icons.FILE(size=size)

    def find_child_by_name(self, name_part: str) -> Optional['ResourceNode']:
        """Помощник для поиска дочернего узла (для Dashboard рендерера)."""
        for child in self.children:
            if name_part.lower() in child.name.lower():
                return child
        return None


# --- Функции сканирования и логика ---

def scan_directory_for_extensions(
    folder_path: Path, 
    extensions: List[str], 
    node_type: str
) -> List[ResourceNode]:
    """Сканирует папку на наличие файлов с определенными расширениями."""
    nodes = []
    if folder_path.exists() and folder_path.is_dir():
        for ext in extensions:
            # Case insensitive search emulation usually required on Linux, 
            # but glob is case-sensitive. On Windows it's usually fine.
            # Here we stick to simple glob.
            for file_path in folder_path.glob(f"*{ext}"):
                nodes.append(ResourceNode(
                    name=file_path.name,
                    path=file_path,
                    type=node_type,
                    description=f"Файл {ext.upper()}",
                    is_critical=False
                ))
    return sorted(nodes, key=lambda x: x.name)


def scan_subfolders(folder_path: Path) -> List[ResourceNode]:
    """Сканирует папку и возвращает список всех подпапок."""
    nodes = []
    if folder_path.exists() and folder_path.is_dir():
        for item in folder_path.iterdir():
            if item.is_dir():
                nodes.append(ResourceNode(
                    name=item.name,
                    path=item,
                    type='folder',
                    description="Рабочая папка",
                    is_critical=False
                ))
    return sorted(nodes, key=lambda x: x.name)


def check_xmp_presence(capture_path: Path, session_subfolder: str) -> bool:
    """
    Проверяет наличие XMP файлов в папке Capture/SessionName.
    """
    if not capture_path or not capture_path.exists():
        return False
    
    target_dir = capture_path / session_subfolder
    if target_dir.exists() and target_dir.is_dir():
        # Ищем хотя бы один XMP файл
        try:
            return any(target_dir.glob("*.xmp")) or any(target_dir.glob("*.XMP"))
        except Exception:
            return False
    return False


def scan_analysis_structure(
    output_path: Path, 
    scope: str = "current", 
    target_session_name: str = ""
) -> List[ResourceNode]:
    """
    Сканирует структуру папки Output на наличие папок Analysis_*.
    """
    nodes = []
    if not output_path.exists() or not output_path.is_dir():
        return nodes

    prefix = "Analysis_"
    target_folder_name = f"{prefix}{target_session_name}" if target_session_name else None

    for item in output_path.iterdir():
        if not item.is_dir():
            continue
        
        if not item.name.startswith(prefix):
            continue
            
        if scope == "current" and target_folder_name:
            if item.name != target_folder_name:
                continue

        desc = "Папка с данными AI-анализа"
        analysis_node = ResourceNode(
            name=item.name,
            path=item,
            type='folder',
            description=desc,
            is_critical=False
        )
        
        # --- Внутренняя структура ---

        # 1. JPG (Всегда добавляем узел)
        jpg_folder = item / "JPG"
        analysis_node.children.append(ResourceNode(
            name="JPG",
            path=jpg_folder,
            type='folder',
            description="Файлы JPG",
            is_critical=False
        ))

        # 2. Masks
        masks_folder = item / "JPG" / "Masks"
        analysis_node.children.append(ResourceNode(
            name="Masks",
            path=masks_folder,
            type='folder',
            description="Сгенерированные маски",
            is_critical=False
        ))

        # 3. JSON файлы
        json_group = item / "info_group_faces.json"
        analysis_node.children.append(ResourceNode(
            name="info_group_faces.json",
            path=json_group,
            type='code',
            description="Инфо о групповых",
            is_critical=False
        ))

        json_portrait = item / "info_portrait_faces.json"
        analysis_node.children.append(ResourceNode(
            name="info_portrait_faces.json",
            path=json_portrait,
            type='code',
            description="Инфо о портретах",
            is_critical=False
        ))

        # Файл связей F2G
        json_matches = item / "matches_portrait_to_group.json"
        analysis_node.children.append(ResourceNode(
            name="matches_portrait_to_group.json",
            path=json_matches,
            type='code',
            description="Связи Портрет->Группа",
            is_critical=False
        ))

        # 4. Отчет
        report_file = item / "face_clustering_report.html"
        
        # ### ИСПРАВЛЕНИЕ: Убрана проверка if report_file.exists(): ###
        # Узел создается всегда. is_critical=False гарантирует показ иконки HTML (а не ERROR),
        # даже если файла нет. Рендерер сам зачеркнет название.
        analysis_node.children.append(ResourceNode(
            name="face_clustering_report.html",
            path=report_file,
            type='html',
            description="HTML отчет",
            is_critical=False 
        ))
        
        nodes.append(analysis_node)

    return sorted(nodes, key=lambda x: x.name)