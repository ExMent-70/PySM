# report_common.py

import sys
from pathlib import Path
from typing import List

# --- ИМПОРТ ИЗ НОВОГО API ---
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Импортируем класс узла из API
    from pysm_lib.pysm_report_api import ResourceNode
except ImportError:
    pass

# --- ФУНКЦИИ СКАНИРОВАНИЯ (Остаются здесь, так как они специфичны для логики скрипта) ---

def scan_directory_for_extensions(folder_path: Path, extensions: List[str], node_type: str) -> List[ResourceNode]:
    nodes = []
    if folder_path.exists() and folder_path.is_dir():
        for ext in extensions:
            for file_path in folder_path.glob(f"*{ext}"):
                nodes.append(ResourceNode(file_path.name, file_path, node_type))
    return sorted(nodes, key=lambda x: x.name)

def scan_subfolders(folder_path: Path) -> List[ResourceNode]:
    nodes = []
    if folder_path.exists() and folder_path.is_dir():
        for item in folder_path.iterdir():
            if item.is_dir():
                nodes.append(ResourceNode(item.name, item, 'folder'))
    return sorted(nodes, key=lambda x: x.name)

def check_xmp_presence(capture_path: Path, session_subfolder: str) -> bool:
    if not capture_path or not capture_path.exists(): return False
    target = capture_path / session_subfolder
    return target.exists() and (any(target.glob("*.xmp")) or any(target.glob("*.XMP")))

def scan_analysis_structure(output_path: Path, scope: str = "current", target_session_name: str = "") -> List[ResourceNode]:
    nodes = []
    if not output_path.exists(): return nodes
    
    prefix = "Analysis_"
    target = f"{prefix}{target_session_name}" if target_session_name else None

    for item in output_path.iterdir():
        if not item.is_dir() or not item.name.startswith(prefix): continue
        if scope == "current" and target and item.name != target: continue

        anode = ResourceNode(item.name, item, 'folder', is_critical=False)
        
        # Добавляем детей (структура как раньше)
        anode.children.append(ResourceNode("JPG", item / "JPG", 'folder', is_critical=False))
        anode.children.append(ResourceNode("Masks", item / "JPG" / "Masks", 'folder', is_critical=False))
        anode.children.append(ResourceNode("info_faces.json", item / "info_faces.json", 'code', is_critical=False))
        anode.children.append(ResourceNode("matches_portrait_to_group.json", item / "matches_portrait_to_group.json", 'code', is_critical=False))
        anode.children.append(ResourceNode("error_matches.json", item / "error_matches.json", 'code', is_critical=False))
        anode.children.append(ResourceNode("face_clustering_report.html", item / "face_clustering_report.html", 'html', is_critical=False))
        
        nodes.append(anode)
    return sorted(nodes, key=lambda x: x.name)