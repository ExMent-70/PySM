# run_img2mask.py

# 1. БЛОК: Импорты и настройка окружения
# ==============================================================================
import argparse
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, List
import concurrent.futures
import os

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else None



# --- НАЧАЛО ИЗМЕНЕНИЙ: Импортируем AVAILABLE_MODELS для динамических choices ---
from rmbg_library import (
    load_config, setup_logging, get_message, Config,
    RmbgModuleProcessor, BiRefNetModuleProcessor, get_compute_device,
    AVAILABLE_MODELS # <--- Новый импорт
)
# --- КОНЕЦ ИЗМЕНЕНИЙ ---
import torch




# 2. БЛОК: Определение параметров и получение конфигурации (ПЕРЕРАБОТАНО)
# ==============================================================================
def get_config() -> Namespace:
    """Определяет CLI-аргументы, которые будут видны в GUI PySM."""
    parser = argparse.ArgumentParser(description="Инструмент для удаления фона с помощью моделей RMBG и BiRefNet.")

    # --- Динамическое получение списков моделей для choices ---
    rmbg_models = [name for name, info in AVAILABLE_MODELS.items() if info.get("processor_module") == "rmbg_rmbg"]
    birefnet_models = [name for name, info in AVAILABLE_MODELS.items() if info.get("processor_module") == "rmbg_birefnet"]

    # --- Определение аргументов с новым префиксом img_rmbg_ ---
    parser.add_argument("--img_rmbg_input_dir", type=str, default="", help="Папка с входными изображениями.")
    parser.add_argument("--img_rmbg_output_dir", type=str, default="", help="Папка для сохранения результатов.")
    parser.add_argument("--img_rmbg_processor_type", type=str, default="rmbg", choices=["rmbg", "birefnet"], help="Тип процессора.")
    
    # --- Разделенные аргументы для выбора модели ---
    parser.add_argument("--img_rmbg_model_rmbg", type=str, default="RMBG-2.0", choices=rmbg_models, help="Модель для процессора 'rmbg'.")
    parser.add_argument("--img_rmbg_model_birefnet", type=str, default="BiRefNet-general", choices=birefnet_models, help="Модель для процессора 'birefnet'.")
    
    parser.add_argument("--img_rmbg_background", type=str, default="Alpha", choices=["Alpha", "Solid", "Original"], help="Тип фона.")
    parser.add_argument("--img_rmbg_mask_blur", type=int, default=0, help="Радиус размытия маски.")
    parser.add_argument("--img_rmbg_mask_offset", type=int, default=0, help="Смещение краев маски.")
    
    # --- Новые опции сохранения и инверсии ---
    parser.add_argument("--img_rmbg_save_options", type=str, nargs='+', default=["image", "mask"], choices=["image", "mask"], help="Что сохранять: 'image', 'mask'.")
    parser.add_argument("--img_rmbg_invert_mask", action=argparse.BooleanOptionalAction, default=False, help="Инвертировать финальную маску.")
    parser.add_argument(
        "--img_rmbg_fast_refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Применить быстрое смягчение краев маски."
    )    
    parser.add_argument("--img_rmbg_threads", type=int, default=0, help="Количество потоков. 0 = авто.")

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Фабрика процессоров (без изменений)
# ==============================================================================
def get_processor(processor_name: str, config: Config, device: torch.device) -> Any:
    logger = logging.getLogger(__name__)
    print(f"\nИнициализация процессора: <b>{processor_name}</b>")
    
    processor_class_map = {
        "rmbg": RmbgModuleProcessor,
        "birefnet": BiRefNetModuleProcessor,
    }
    
    ProcessorClass = processor_class_map.get(processor_name)
    if not ProcessorClass:
        logger.error(f"Unknown processor name specified: {processor_name}")
        return None

    try:
        processor_instance = ProcessorClass(config=config, device=device)
        print(f"Процессор <b>{processor_name}</b> успешно инициализирован.\n")
        return processor_instance
    except Exception as e:
        logger.critical(get_message("ERROR_PROCESSOR_INIT", processor_name=processor_name, exc=e), exc_info=True)
        return None


# 4. БЛОК: Функция-воркер для обработки одного файла (ИСПРАВЛЕНО)
# ==============================================================================
def process_single_image(file_path: Path, processor: Any, config: Config, model_name: str, args: Namespace) -> str:
    """
    Обрабатывает один файл изображения. Возвращает статус: 'success', 'skipped', 'error'.
    """
    from rmbg_library.rmbg_utils import load_image, save_image
    from rmbg_library import mask_ops
    
    try:
        image = load_image(file_path)
        if image is None: return "skipped"

        final_image, processed_mask = processor.process(image, filename_for_log=file_path.name)

        # --- НАЧАЛО КЛЮЧЕВОГО ИСПРАВЛЕНИЯ: Проверяем, что маска была создана ---
        if processed_mask is None:
            # Если процессор вернул None, значит, произошла внутренняя ошибка.
            # Ошибка уже залогирована внутри процессора.
            return f"ERROR on {file_path.name}: Failed to generate mask."
        # --- КОНЕЦ КЛЮЧЕВОГО ИСПРАВЛЕНИЯ ---

        if args.img_rmbg_fast_refine:
            logging.getLogger(__name__).debug(f"Applying fast refine to mask for {file_path.name}")
            processed_mask = mask_ops.fast_refine(processed_mask)

        save_opts = config.model.save_options
        
        if "image" in save_opts and final_image:
            suffix = ".png" if final_image.mode in ["RGBA", "L"] else file_path.suffix
            out_path = config.paths.output_dir / f"{file_path.stem}_{model_name}_output{suffix}"
            save_image(final_image, out_path)
        
        if "mask" in save_opts and processed_mask:
            out_path = config.paths.output_dir / f"{file_path.stem}_{model_name}_mask.png"
            save_image(processed_mask, out_path)
        
        return "success"
    except Exception as e:
        # Эта секция теперь будет ловить только ошибки самой функции-воркера, а не модели.
        logging.getLogger(__name__).exception(
            get_message("ERROR_PROCESSING_FILE", file_path=str(file_path), processor=config.processor_type, exc=e)
        )
        return f"ERROR on {file_path.name}: {e}"


# 5. БЛОК: Основная логика выполнения `main` (ПЕРЕРАБОТАНО)
# ==============================================================================
def main():
    """Главная функция-оркестратор."""
    total_start_time = time.monotonic()
    args = get_config()

    # --- НАЧАЛО БЛОКА ГИБРИДНОЙ КОНФИГУРАЦИИ ---
    
    # 1. Загружаем `config.toml` как базу с настройками по умолчанию
    config_path = Path(__file__).parent / "config.toml"
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить базовый config '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Переопределяем значения из `config` аргументами, полученными из GUI/CLI
    # Пути (обязательные параметры, поэтому всегда переопределяем)
    config.paths.input_dir = Path(args.img_rmbg_input_dir)
    config.paths.output_dir = Path(args.img_rmbg_output_dir)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Тип процессора
    config.processor_type = args.img_rmbg_processor_type
    
    # 3. Логика выбора имени модели
    if args.img_rmbg_processor_type == 'rmbg':
        model_name = args.img_rmbg_model_rmbg
    else: # birefnet
        model_name = args.img_rmbg_model_birefnet
    config.model.name = model_name

    # Параметры постобработки и сохранения
    config.model.background = args.img_rmbg_background
    config.model.mask_blur = args.img_rmbg_mask_blur
    config.model.mask_offset = args.img_rmbg_mask_offset
    config.model.invert_output = args.img_rmbg_invert_mask
    config.model.save_options = args.img_rmbg_save_options
    
    # --- КОНЕЦ БЛОКА ГИБРИДНОЙ КОНФИГУРАЦИИ ---

    if not config.paths.input_dir.is_dir():
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Папка с входными изображениями не найдена: {config.paths.input_dir}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(config.logging, log_dir=config.paths.output_dir)
    logger.info("--- RMBG/BiRefNet Tool Run (Multi-threaded) ---")
    logger.info(f"Final effective processor: {config.processor_type}")
    logger.info(f"Final effective model name: {config.model.name}")
    logger.info(f"Input directory: {config.paths.input_dir}")
    logger.info(f"Output directory: {config.paths.output_dir}")

    device = get_compute_device()
    logger.info(f"Using device: {device}")
    
    # 1. Инициализируем процессор ОДИН РАЗ до пула потоков
    processor = get_processor(config.processor_type, config, device)
    if processor is None:
        logger.critical("Failed to initialize processor. Exiting.")
        sys.exit(1)

    # 2. Собираем список файлов для обработки
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = sorted([p for p in config.paths.input_dir.glob("*") if p.is_file() and p.suffix.lower() in image_extensions])

    if not image_files:
        tqdm.write(get_message("INFO_NO_IMAGES_FOUND", input_dir=str(config.paths.input_dir)))
        sys.exit(0)

    logger.info(f"Найдено <b>{len(image_files)}</b> изображений для обработки")
    
    # 3. Определяем количество потоков
    num_threads = args.img_rmbg_threads
    if num_threads == 0:
        cpu_count = os.cpu_count() or 1
        num_threads = min(16, cpu_count * 2 + 1)
    logger.info(f"Using {num_threads} worker threads.")

    # 4. Запускаем многопоточную обработку
    stats = {"success": 0, "skipped": 0, "error": 0}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_file = {
                executor.submit(process_single_image, file, processor, config, model_name, args): file
                for file in image_files
            }
            
            progress_bar = tqdm(concurrent.futures.as_completed(future_to_file), total=len(image_files), desc=f"Processing ({model_name})")
            
            for future in progress_bar:
                try:
                    result = future.result()
                    if result == "success":
                        stats["success"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["error"] += 1
                        tqdm.write(result)
                except Exception as exc:
                    stats["error"] += 1
                    file_path = future_to_file[future]
                    tqdm.write(f"FATAL ERROR processing {file_path.name}: {exc}")
    finally:
        if hasattr(processor, "release") and callable(getattr(processor, "release")):
            processor.release()
            
    # 5. Финальный отчет
    total_time = time.monotonic() - total_start_time
    summary_message = (
        f"\nProcessing finished in {total_time:.2f} seconds.\n"
        f"Successfully processed: {stats['success']} files.\n"
        f"Skipped: {stats['skipped']} files.\n"
        f"Errors: {stats['error']} files."
    )
    tqdm.write(summary_message)
    #logger.info(summary_message.replace('\n', ' '))

    if IS_MANAGED_RUN:
        pysm_context.log_link(url_or_path=str(config.paths.output_dir), text="<br>Открыть папку с результатами")
        pysm_context.log_link(url_or_path=str(config.paths.input_dir), text="Открыть исходную папку")
# 6. БЛОК: Точка входа (переименован из 5)
# ==============================================================================
if __name__ == "__main__":
    main()