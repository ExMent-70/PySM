#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cluster_face.py
Единая точка входа для анализа лиц (Техническая чистка, Портреты, Матчинг).
"""
print("<b>КЛАСТЕРИЗАЦИЯ ФОТОГРАФИЙ</b>")
print(f"<i>Инициализация...</i>")



import argparse
import logging
import sys
from pathlib import Path

# --- Настройка путей для импорта ---
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    pass

# --- Импорты ---

try:
    # Импорт контекста для вывода
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
except ImportError:
    pass

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

from _lib.analysis_manager import AnalysisDataManager

# --- Логирование ---
log_level = "INFO"
if pysm_context:
    log_level = pysm_context.get("sys_log_level", "INFO")

logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO), 
    format="%(message)s", 
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Универсальный анализатор лиц.")
    
    # === ОСНОВНЫЕ ПАРАМЕТРЫ ===
    parser.add_argument("--a_target_dir", type=str, required=True, 
                        help="Путь к папке Analysis (содержит info_faces.json)")
    
    parser.add_argument("--a_mode", type=str, required=True, 
                        choices=["cleaning", "face", "matches"],
                        help="Режим работы")

    # === УНИВЕРСАЛЬНЫЙ ПОРОГ ===
    parser.add_argument("--a_sim_threshold", type=float, default=0.40, 
                        help="Порог схожести/радиус поиска. Для Cleaning/Face это EPS (0.25-0.4), для Matches это Threshold (0.4-0.5).")

    # === ОБЩИЕ ===
    parser.add_argument("--a_metric", type=str, default="cosine", 
                        help="Метрика расстояния (cosine/euclidean)")

    # === ПАРАМЕТРЫ РЕЖИМА Cleaning ===
    parser.add_argument("--a_clear_min_score", type=float, default=0.60, 
                        help="Мин. уверенность детектора")
    parser.add_argument("--a_clear_min_abs_area", type=int, default=2500, 
                        help="Мин. площадь лица (px)")
    parser.add_argument("--a_clear_min_rel_area", type=float, default=0.0015, 
                        help="Мин. относительная площадь")
    
    # === ПАРАМЕТРЫ КЛАСТЕРИЗАЦИИ ===
    parser.add_argument("--a_clear_min_claster_size", type=int, default=3, 
                        help="Мин. размер кластера (min_samples)")
    
    # === ПАРАМЕТРЫ РЕЖИМА FACE ===
    parser.add_argument("--a_algorithm", type=str, default="dbscan", 
                        choices=["dbscan", "hdbscan"], help="Алгоритм для портретов")
    
    # HDBSCAN specific
    parser.add_argument("--a_hdb_cluster_selection_epsilon", type=float, default=0.0)
    parser.add_argument("--a_hdb_min_samples", type=int, default=None)

    # === ПАРАМЕТРЫ РЕЖИМА MATCHES ===
    parser.add_argument("--a_ref_dir", type=str, default=None, 
                        help="Папка с эталонами (если отличается от target)")

    return ConfigResolver(parser).resolve_all()


def main():
    
    try:
        config = get_config()
        target_dir = Path(config.a_target_dir)
        mode = config.a_mode

              
        logger.debug(f"ℹ️ Режим работы: {mode.upper()}")
        logger.debug(f"ℹ️ Папка данных текущей фотосессии: <i>{target_dir}</i>")

        data_manager = AnalysisDataManager(target_dir)


        if not data_manager.load_data():
            logger.critical(f"{icon_error} Не удалось загрузить данные. Завершение работы")
            sys.exit(1)

        strategy = None
        
        # --- ЛЕНИВЫЙ ИМПОРТ СТРАТЕГИЙ ---
        # Импортируем тяжелые модули только тогда, когда точно знаем, что они нужны.
        
        if mode == "cleaning":
            logger.info(f"{icon_info} Загрузка модуля Tech...")
            # Импортируем напрямую из файла модуля, минуя __init__ пакета
            from _lib.strategies_analysis.tech import TechnicalStrategy
            strategy = TechnicalStrategy()
            
        elif mode == "face":
            logger.info(f"{icon_info} Загрузка модуля Portraits...")
            from _lib.strategies_analysis.portraits import PortraitsStrategy
            strategy = PortraitsStrategy()
            
        elif mode == "matches":
            logger.info(f"{icon_info} Загрузка модуля Matching (Scipy)...")
            from _lib.strategies_analysis.matching import MatchingStrategy
            strategy = MatchingStrategy()
            
        else:
            logger.critical(f"{icon_error} Неизвестный режим: {mode}")
            sys.exit(1)

        strategy.run(config, data_manager)
        logger.debug("=== ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===")
       
    except Exception as e:
        logger.critical(f"{icon_error} Критическая ошибка выполнения: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()