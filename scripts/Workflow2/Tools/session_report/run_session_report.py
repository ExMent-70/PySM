# run_wf_report.py

import sys
import argparse
from pathlib import Path
import traceback

# --- Настройка окружения ---
try:
    current_script_path = Path(__file__).resolve()
    script_dir = current_script_path.parent
    project_root = current_script_path.parents[4]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from pysm_lib.pysm_context import pysm_context, ConfigResolver
    
    # Импорт фабрик отчетов
    from report_renderer_standard import generate_standard_html
    from report_renderer_dashboard import generate_dashboard_html

    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    traceback.print_exc()
    IS_MANAGED_RUN = False
    pysm_context = None


def main():
    if not IS_MANAGED_RUN or not pysm_context:
        print("Ошибка: Скрипт запущен вне окружения PySM.", file=sys.stderr)
        return

    # --- 1. Аргументы ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", dest="report_template", default="standard")
    parser.add_argument("--scope", dest="report_scope", default="current")
    parser.add_argument("--icon_size_tree", type=int, default=24)
    parser.add_argument("--icon_size_dashboard", type=int, default=48)

    resolver = ConfigResolver(parser)
    config = resolver.resolve_all()

    if not config.report_scope:
        config.report_scope = "current"

    # --- 2. Контекст ---
    session_path_str = pysm_context.get("wf_session_path")
    psd_path_str = pysm_context.get("wf_psd_path")
    wf_idsgn_catalog_str = pysm_context.get("wf_idsgn_catalog")
    wf_portrait_session = pysm_context.get("wf_portrait_session")
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")

    path_session_base = Path(session_path_str) if session_path_str else None
    path_psd_base = Path(psd_path_str) if psd_path_str else None
    path_c1_session = (path_session_base / session_name) if (path_session_base and session_name) else None

    # --- 3. Генерация ---
    html_content = ""
    
    if config.report_template == "dashboard":
        html_content = generate_dashboard_html(
            config, 
            path_session_base, 
            path_psd_base, 
            path_c1_session, 
            session_name, 
            photo_session, 
            wf_portrait_session,
            wf_idsgn_catalog_str
        )
    else:
        html_content = generate_standard_html(
            config, 
            path_session_base, 
            path_psd_base, 
            path_c1_session, 
            session_name, 
            photo_session, 
            wf_idsgn_catalog_str,
            wf_portrait_session
        )

    # --- 4. Вывод ---
    pysm_context.log_html(html_content)


if __name__ == "__main__":
    main()
