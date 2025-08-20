# img2mask/mask_sam2/run_mask_sam2.py

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
    
    from _rmbgtools import logger, load_config, segment_by_text, SAM2_MODELS, DINO_MODELS
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
        from _rmbgtools import logger, load_config, segment_by_text, SAM2_MODELS, DINO_MODELS
        from _rmbgtools.utils.cli_utils import expand_paths
    except ImportError as e:
        print(f"Критическая ошибка импорта в автономном режиме: {e}", file=sys.stderr)
        sys.exit(1)

# ======================================================================================
# Блок 2: Определение параметров и получение конфигурации
# ======================================================================================
def get_config() -> Namespace:
    parser = argparse.ArgumentParser(description="Инструмент для сегментации объектов по тексту.")
    parser.add_argument("--sam2_inputs", nargs='+', help="Путь к исходным файлам или папкам.")
    parser.add_argument("--sam2_output", help="Путь к папке для сохранения.")
    parser.add_argument("--sam2_config", help="Путь к файлу config.toml.")
    parser.add_argument("--sam2_prompt", help="Текстовое описание объекта.")
    parser.add_argument("--sam2_sam_model", choices=SAM2_MODELS, help="Переопределить модель SAM2.")
    parser.add_argument("--sam2_dino_model", choices=DINO_MODELS, help="Переопределить модель GroundingDINO.")
    parser.add_argument("--sam2_threshold", type=float, help="Порог уверенности детекции (0.0-1.0).")
    parser.add_argument("--sam2_threads", type=int, help="Переопределить кол-во потоков.")
    parser.add_argument("--sam2_device", choices=["auto", "cpu", "cuda"], help="Переопределить устройство.")
    parser.add_argument("--sam2_background", choices=["Alpha", "Color"], help="Тип фона.")
    parser.add_argument("--sam2_bg_color", type=str, help="Цвет фона в HEX.")
    parser.add_argument("--sam2_blur", type=int, help="Радиус размытия маски.")
    parser.add_argument("--sam2_offset", type=int, help="Смещение краев маски.")
    parser.add_argument("--sam2_smooth", type=float, help="Сила сглаживания краев.")
    parser.add_argument("--sam2_fill_holes", action="store_true", dest="sam2_fill_holes_true", help="Включить заполнение 'дыр'.")
    parser.add_argument("--sam2_no_fill_holes", action="store_false", dest="sam2_fill_holes_true", help="Отключить заполнение 'дыр'.")
    parser.add_argument("--sam2_invert", action="store_true", help="Инвертировать маску.")
    parser.add_argument("--sam2_verbose", action="store_true", help="Включить детальное логирование.")
    parser.set_defaults(sam2_fill_holes_true=None)

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    else:
        return parser.parse_args()

# ======================================================================================
# Блок 3: Основная функция main
# ======================================================================================
def main():
    config_cli = get_config()
    if config_cli.sam2_verbose: logger.setLevel(logging.DEBUG)
    config_path = config_cli.sam2_config or "config.toml"
    app_config = load_config(config_path)

    if not config_cli.sam2_inputs or not config_cli.sam2_output or not config_cli.sam2_prompt:
        logger.critical("Ошибка: Аргументы --sam2_inputs, --sam2_output, и --sam2_prompt обязательны.")
        sys.exit(1)

    image_paths = expand_paths(config_cli.sam2_inputs)
    if not image_paths:
        logger.error("Не найдено подходящих файлов изображений. Завершение.")
        sys.exit(1)

    kwargs = {
        "model_dir": app_config.global_settings.model_dir,
        "device": config_cli.sam2_device or app_config.global_settings.device,
        "num_threads": config_cli.sam2_threads or app_config.global_settings.num_threads,
        "sam_model_name": config_cli.sam2_sam_model or app_config.segment_settings.sam_model_name,
        "dino_model_name": config_cli.sam2_dino_model or app_config.segment_settings.dino_model_name,
        "threshold": config_cli.sam2_threshold if config_cli.sam2_threshold is not None else app_config.segment_settings.threshold,
        "background": config_cli.sam2_background or app_config.segment_settings.postprocess.background,
        "background_color": config_cli.sam2_bg_color or app_config.segment_settings.postprocess.background_color,
        "mask_blur": config_cli.sam2_blur if config_cli.sam2_blur is not None else app_config.segment_settings.postprocess.mask_blur,
        "mask_offset": config_cli.sam2_offset if config_cli.sam2_offset is not None else app_config.segment_settings.postprocess.mask_offset,
        "smooth": config_cli.sam2_smooth if config_cli.sam2_smooth is not None else app_config.segment_settings.postprocess.smooth,
        "fill_holes": config_cli.sam2_fill_holes_true if config_cli.sam2_fill_holes_true is not None else app_config.segment_settings.postprocess.fill_holes,
        "invert_output": config_cli.sam2_invert or app_config.segment_settings.postprocess.invert_output,
    }
    
    logger.info(f"Запуск сегментации для {len(image_paths)} изображений с промптом: '{config_cli.sam2_prompt}'")
    
    results_iterator = segment_by_text(
        images=image_paths, 
        output_dir=config_cli.sam2_output, 
        prompt=config_cli.sam2_prompt,
        **kwargs
    )
    
    progress_bar = tqdm(results_iterator, total=len(image_paths), desc="Сегментация")

    not_found_files, error_files, success_results = [], [], []
    for result in progress_bar:
        if result:
            status, value = result
            if status == "not_found": not_found_files.append(value)
            elif status == "error": error_files.append(value)
            else: success_results.append(result)

    if not_found_files:
        message = f"Объект '{config_cli.sam2_prompt}' не найден в следующих файлах:\n"
        message += "\n".join([f"  - {os.path.basename(f)}" for f in sorted(not_found_files)])
        logger.warning(message)
    
    if error_files:
        message = f"Произошла ошибка при обработке следующих файлов:\n"
        message += "\n".join([f"  - {os.path.basename(f)}" for f in sorted(error_files)])
        logger.error(message)

    logger.info("Сегментация успешно завершена.")

# ======================================================================================
# Блок 4: Точка входа в скрипт
# ======================================================================================
if __name__ == "__main__":
    main()