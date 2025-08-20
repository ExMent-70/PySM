# img2mask/mask_rmbg/run_mask_rmbg.py

# ======================================================================================
# Блок 1: Импорты и настройка путей
# ======================================================================================
import argparse
import os
import sys
import logging
from typing import List
from argparse import Namespace

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from _rmbgtools import logger, load_config, remove_background, ALL_REMOVER_MODELS
    from _rmbgtools.utils.cli_utils import expand_paths
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs): return x
    try:
        from _rmbgtools import logger, load_config, remove_background, ALL_REMOVER_MODELS
        from _rmbgtools.utils.cli_utils import expand_paths
    except ImportError as e:
        print(f"Критическая ошибка импорта в автономном режиме: {e}", file=sys.stderr)
        sys.exit(1)

# ======================================================================================
# Блок 2: Определение параметров и получение конфигурации
# ======================================================================================
def get_config() -> Namespace:
    parser = argparse.ArgumentParser(description="Инструмент для удаления фона.")
    parser.add_argument("--rmbg_inputs", nargs='+', help="Путь к исходным файлам или папкам.")
    parser.add_argument("--rmbg_output", help="Путь к папке для сохранения.")
    parser.add_argument("--rmbg_config", help="Путь к файлу config.toml.")
    parser.add_argument("--rmbg_model", choices=ALL_REMOVER_MODELS, help="Переопределить модель.")
    parser.add_argument("--rmbg_threads", type=int, help="Переопределить кол-во потоков.")
    parser.add_argument("--rmbg_device", choices=["auto", "cpu", "cuda"], help="Переопределить устройство.")
    parser.add_argument("--rmbg_sensitivity", type=float, help="Чувствительность маски (0.0-1.0).")
    parser.add_argument("--rmbg_process_res", type=int, help="Разрешение для обработки.")
    parser.add_argument("--rmbg_no_refine", action="store_true", help="Отключить уточнение переднего плана.")
    parser.add_argument("--rmbg_background", choices=["Alpha", "Color"], help="Тип фона.")
    parser.add_argument("--rmbg_bg_color", type=str, help="Цвет фона в HEX.")
    parser.add_argument("--rmbg_blur", type=int, help="Радиус размытия маски.")
    parser.add_argument("--rmbg_offset", type=int, help="Смещение краев маски.")
    parser.add_argument("--rmbg_smooth", type=float, help="Сила сглаживания краев.")
    parser.add_argument("--rmbg_fill_holes", action="store_true", help="Включить заполнение 'дыр'.")
    parser.add_argument("--rmbg_invert", action="store_true", help="Инвертировать маску.")
    parser.add_argument("--rmbg_verbose", action="store_true", help="Включить детальное логирование.")

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    else:
        return parser.parse_args()

# ======================================================================================
# Блок 3: Основная функция main
# ======================================================================================
def main():
    config_cli = get_config()
    if config_cli.rmbg_verbose: logger.setLevel(logging.DEBUG)
    config_path = config_cli.rmbg_config or "config.toml"
    app_config = load_config(config_path)

    if not config_cli.rmbg_inputs or not config_cli.rmbg_output:
        logger.critical("Ошибка: Аргументы --rmbg_inputs и --rmbg_output обязательны.")
        sys.exit(1)

    image_paths = expand_paths(config_cli.rmbg_inputs)
    if not image_paths:
        logger.error("Не найдено подходящих файлов изображений. Завершение.")
        sys.exit(1)

    kwargs = {
        "model_dir": app_config.global_settings.model_dir,
        "device": config_cli.rmbg_device or app_config.global_settings.device,
        "num_threads": config_cli.rmbg_threads or app_config.global_settings.num_threads,
        "model_name": config_cli.rmbg_model or app_config.remove_settings.model_name,
        "sensitivity": config_cli.rmbg_sensitivity if config_cli.rmbg_sensitivity is not None else app_config.remove_settings.rmbg_specific.sensitivity,
        "process_res": config_cli.rmbg_process_res or app_config.remove_settings.rmbg_specific.process_res,
        "refine_foreground": not config_cli.rmbg_no_refine,
        "background": config_cli.rmbg_background or app_config.remove_settings.postprocess.background,
        "background_color": config_cli.rmbg_bg_color or app_config.remove_settings.postprocess.background_color,
        "mask_blur": config_cli.rmbg_blur if config_cli.rmbg_blur is not None else app_config.remove_settings.postprocess.mask_blur,
        "mask_offset": config_cli.rmbg_offset if config_cli.rmbg_offset is not None else app_config.remove_settings.postprocess.mask_offset,
        "smooth": config_cli.rmbg_smooth if config_cli.rmbg_smooth is not None else app_config.remove_settings.postprocess.smooth,
        "fill_holes": config_cli.rmbg_fill_holes or app_config.remove_settings.postprocess.fill_holes,
        "invert_output": config_cli.rmbg_invert or app_config.remove_settings.postprocess.invert_output,
    }
    
    logger.info(f"Запуск удаления фона для {len(image_paths)} изображений с моделью '{kwargs['model_name']}'.")
    
    results_iterator = remove_background(images=image_paths, output_dir=config_cli.rmbg_output, **kwargs)
    progress_bar = tqdm(results_iterator, total=len(image_paths), desc="Удаление фона")
    results = [res for res in progress_bar if res is not None]
    
    logger.info(f"Удаление фона успешно завершено. Обработано {len(results)} файлов.")

# ======================================================================================
# Блок 4: Точка входа в скрипт
# ======================================================================================
if __name__ == "__main__":
    main()