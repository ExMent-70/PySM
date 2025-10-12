# analize/cluster_faces/run_cluster_faces.py

# --- Блок 1: Импорты и настройка путей ---
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_theme_api import theme_api # <--- 1. ДОБАВЛЕН ИМПОРТ
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False


# --- Блок 2: Настройка логирования и вспомогательные функции ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():

    """
    Формирует пути для кластеризации на основе переменных контекста PySM.
    """
    if IS_MANAGED_RUN and pysm_context:

        header_style = theme_api.get_dynamic_style("script_description", "")

        html_lines = []
        html_lines.append(f"<div style='{header_style}'>")
        html_lines.append("Этап 1. Глобальные переменные")
        html_lines.append("</div>")

        final_html = "".join(html_lines)
        print(f"PYSM_HTML_BLOCK:{final_html}", file=sys.stderr, flush=True)


        print("\n", file=sys.stderr)
        # Проверка базовой папки для RAW-файлов
        session_path_str = pysm_context.get("wf_session_path")
        path_to_check = Path(session_path_str)       
        if session_path_str != "" and path_to_check.exists():
            pysm_context.log_link(url_or_path=str(session_path_str), text="Базовая папка для хранения исходных RAW-файлов")
        else:
            print("⚠️ Папка для хранения исходных RAW-файлов не задана", file=sys.stderr)


        # Проверка базовой папки для файлов фотоальбомов
        psd_path = pysm_context.get("wf_psd_path")
        path_to_check = Path(psd_path)       
        if psd_path != "" and path_to_check.exists():
            pysm_context.log_link(url_or_path=str(psd_path), text="Базовая папка для хранения альбомов (файлы psd, indd и т.д.)")
        else:
            print("⚠️ Папка для хранения альбомов не задана", file=sys.stderr)
        print("\n", file=sys.stderr)

        photo_session = pysm_context.get("wf_photo_session")        
        children_file_name = pysm_context.get("wf_children_file_name")

        html_lines = []
        html_lines.append(f"<div style='{header_style}'>")
        html_lines.append("Этап 2. Локальные переменные текущей сессии Capture One")
        html_lines.append("</div>")

        final_html = "".join(html_lines)
        print(f"PYSM_HTML_BLOCK:{final_html}", file=sys.stderr, flush=True)

        print("\n", file=sys.stderr)
        # Проверка рабочей папки Capture One
        session_name = pysm_context.get("wf_session_name")
        base_path = Path(session_path_str) / session_name
        path_to_check = Path(base_path)       
        if base_path != "" and path_to_check.exists():
            pysm_context.log_link(url_or_path=str(base_path), text="Рабочая папка текущей сессии Capture One")
        else:
            print("⚠️ Рабочая папка текущей сессии Capture One не задана", file=sys.stderr)
     
        
        data_dir = base_path / "Output"
        pysm_context.log_link(url_or_path=str(data_dir), text="Папка Output")

        data_dir = base_path / "Output" / f"Analysis_{photo_session}"           
        pysm_context.log_link(url_or_path=str(data_dir), text="Папка с результатами кластеризации исходных фотографий (файлы JSON)<br>")
        print(" ", file=sys.stderr)

        report_file = data_dir / "face_clustering_report.html"
        pysm_context.log_link(url_or_path=str(report_file), text=f"HTML-отчёт с результатами кластеризации<br>")
        print(" ", file=sys.stderr)        

        children_file = base_path / f"{photo_session}_{children_file_name}"
        pysm_context.log_link(url_or_path=str(children_file), text=f"Открыть файл \"<b>{photo_session}_{children_file_name}</b>\" (список портретных кластеров)")
        print(" ", file=sys.stderr)

        session_file = base_path / f"{session_name}.cosessiondb"
        pysm_context.log_link(url_or_path=str(session_file), text=f"Открыть сессию \"<b>{session_name}.cosessiondb</b>\" в Capture One<br>")
        print(" ", file=sys.stderr)
        

        _psd_path = pysm_context.get("wf_psd_path")
        psd_path = Path(_psd_path) / session_name  
        template_path = psd_path / "Альбом" / "_ШАБЛОНЫ_"
        photo_path = psd_path / "Альбом" / "Фото"  
        html_file = psd_path / (session_name + ".html")

        html_lines = []
        html_lines.append(f"<div style='{header_style}'>")
        html_lines.append("Рабочие файлы Photoshop и InDesign")
        html_lines.append("</div>")

        final_html = "".join(html_lines)
        print(f"PYSM_HTML_BLOCK:{final_html}", file=sys.stderr, flush=True)        
        
        pysm_context.log_link(url_or_path=str(psd_path), text=f"Рабочая папка с материалами для альбомов \"{session_name}\"")
        pysm_context.log_link(url_or_path=str(template_path), text="Папка с шаблонами InDesign")
        pysm_context.log_link(url_or_path=str(photo_path), text="Папка с файлами PSD<br>")
        pysm_context.log_link(url_or_path=str(html_file), text="Приложение к договору (файл HTML)<br>")
        print(" ", file=sys.stderr)
    
   

if __name__ == "__main__":
    main()