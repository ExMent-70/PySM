# run_session_create.py

# --- Блок 1: Импорты ---
# ==============================================================================
import argparse
import os
import pathlib
import re
import shutil
import sys
import logging
from argparse import Namespace

# Жесткий импорт зависимостей экосистемы PySM (без fallback-заглушек).
# Скрипт предназначен только для работы внутри среды выполнения.
from pysm_lib import pysm_context
from pysm_lib.pysm_context import ConfigResolver
from pysm_lib.pysm_progress_reporter import tqdm
#from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder



# --- Блок 2: Константы ---
# ==============================================================================
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
TEMPLATE_BASE_DIR_NAME = "_TEMPLATE_"
TEMPLATE_SESSION_DIR_NAME = "_TEMPLATE_SESSION_"
TEMPLATE_ALBUM_DIR_NAME = "_TEMPLATE_ALBUM_"
TEMPLATE_SESSION_FILE_NAME_BASE = "_TEMPLATE_SESSION_"
COSESSIONDB_EXT = ".cosessiondb"

INVALID_FOLDER_NAME_CHARS = r'[\<\>\:\"\/\\\|\?\*]'
RESERVED_FOLDER_NAMES = {
    ".", "..", "con", "prn", "aux", "nul", 
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", 
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"
}

logging.basicConfig(
    level=logging.INFO,
    #format="[%(levelname)s] %(message)s",
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- Блок 3: Вспомогательные функции ---
# ==============================================================================
def is_valid_foldername(name: str) -> tuple[bool, str]:
    """Проверяет корректность имени для создания папки."""
    if not isinstance(name, str) or not name.strip():
        return False, "Имя не может быть пустым."
    if re.search(INVALID_FOLDER_NAME_CHARS, name):
        return False, f"Имя содержит недопустимые символы: <>:\"/\\|?*"
    if name.strip().lower() in RESERVED_FOLDER_NAMES:
        return False, f"Имя '{name}' зарезервировано системой."
    return True, ""


def check_unresolved_macros(value: str, param_name: str) -> None:
    """Проверяет, остались ли неразрешенные макросы {имя_переменной} в строке."""
    if isinstance(value, str) and re.search(r'\{.*\}', value):
        logger.error(f"ОШИБКА: Параметр '{param_name}' содержит неразрешенный макрос: {value}")
        sys.exit(1)


def get_copy_function(overwrite: bool):
    """
    Возвращает функцию копирования файлов (callback для shutil.copytree),
    которая учитывает флаг перезаписи существующих файлов.
    """
    def _copy_func(src, dst, *, follow_symlinks=True):
        if not overwrite and os.path.exists(dst):
            logger.debug(f"    Пропуск существующего файла: <i>{os.path.basename(dst)}</i>")
            return dst
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
    return _copy_func


def copy_template_structure(
    template_src_dir: pathlib.Path, 
    target_dir: pathlib.Path, 
    overwrite_files: bool, 
    template_name: str
) -> None:
    """
    Копирует шаблон структуры папок в целевую директорию.
    Безопасно обрабатывает слияние с существующей директорией.
    """
    if not template_src_dir.is_dir():
        logger.error(f"ОШИБКА: Папка шаблона {template_name} не найдена: {template_src_dir}")
        sys.exit(1)

    dir_already_existed = target_dir.exists()
    
    try:
        # dirs_exist_ok=True позволяет сливать содержимое папок, если они уже существуют (Python 3.8+)
        shutil.copytree(
            template_src_dir,
            target_dir,
            dirs_exist_ok=True,
            copy_function=get_copy_function(overwrite_files)
        )
        logger.debug(f"  Структура папок {template_name} успешно скопирована в: {target_dir}")

    except Exception as e:
        logger.error(f"ОШИБКА при копировании структуры {template_name}: {e}")
        
        # Безопасный откат: удаляем только если мы сами создали папку на этом запуске
        if not dir_already_existed and target_dir.exists():
            logger.info(f"  Выполняется откат: удаление незавершенной папки {target_dir}", file=sys.stderr)
            shutil.rmtree(target_dir, ignore_errors=True)
        sys.exit(1)


# --- Блок 4: Получение конфигурации ---
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(
        description="Создает структуру рабочих папок из шаблонов (C1 Session, Album PSD).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--wf_session_name", type=str, default="{wf_session_name}", help="Имя для новой сессии (например, имя класса).")
    parser.add_argument("--wf_raw_path", type=str, default="{wf_raw_path}", help="Базовый путь к RAW (используется как default для psd).")
    parser.add_argument("--wf_session_path", type=str, default="{wf_session_path}", help="Папка, в которой будет создана сессия C1.")
    parser.add_argument("--wf_psd_path", type=str, default="{wf_raw_path}", help="Папка, в которой будет создана структура Альбома (PSD).")
    
    # Флаги для управления перезаписью файлов (по умолчанию False)
    parser.add_argument("--wf_overwrite_session", action="store_true", help="Перезаписывать существующие файлы шаблона сессии C1.")
    parser.add_argument("--wf_overwrite_album", action="store_true", help="Перезаписывать существующие файлы шаблона Альбома PSD.")

    resolver = ConfigResolver(parser)
    return resolver.resolve_all()


# --- Блок 5: Основная логика скрипта ---
# ==============================================================================
def main():
    """Основной рабочий процесс скрипта."""
    logger.info("\n<b>СОЗДАНИЕ СТРУКТУРЫ РАБОЧИХ ПАПОК</b>")   
    # 5.1. Получение и валидация параметров
    # --------------------------------------------------------------------------
    logger.debug("<b>Получение и валидация параметров</b>")
    config = get_config()

    # Проверка на наличие неразрешенных макросов {переменная}
    check_unresolved_macros(config.wf_session_name, "wf_session_name")
    check_unresolved_macros(config.wf_session_path, "wf_session_path")
    check_unresolved_macros(config.wf_psd_path, "wf_psd_path")

    # Валидация имени сессии
    is_valid, reason = is_valid_foldername(config.wf_session_name)
    if not is_valid:
        logger.error(f"ОШИБКА: Некорректное имя сессии/папки: {reason}")
        sys.exit(1)

    # Валидация существования базовых директорий на диске
    target_session_base = pathlib.Path(config.wf_session_path)
    target_psd_base = pathlib.Path(config.wf_psd_path)

    if not target_session_base.is_dir():
        logger.error(f"ОШИБКА: Корневая папка для исходных файлов не существует: {target_session_base}")
        sys.exit(1)
        
    if not target_psd_base.is_dir():
        logger.error(f"ОШИБКА: Корневая папка для PSD и INDD файлов не существует: {target_psd_base}")
        sys.exit(1)

    logger.debug(f"Имя новой структуры: {config.wf_session_name}")
    logger.debug(f"Корневая папка для исходных файлов: {target_session_base}")
    (f"Корневая папка для PSD и INDD файлов: {target_psd_base}")
    logger.debug(f"Флаги перезаписи: файлы фотосессии={config.wf_overwrite_session}, файлы альбома={config.wf_overwrite_album}")

    # 5.2. Создание структуры сессии C1
    # --------------------------------------------------------------------------
    logger.info("\n✅ Создание структуры папок для исходных файлов фотосессии")
    if config.wf_overwrite_session:
        logger.info("⚠️ <i>Все существующие файлы перезаписаны</i>")
    
    template_session_path = SCRIPT_DIR / TEMPLATE_BASE_DIR_NAME / TEMPLATE_SESSION_DIR_NAME
    final_session_path = target_session_base / config.wf_session_name

    copy_template_structure(
        template_src_dir=template_session_path,
        target_dir=final_session_path,
        overwrite_files=config.wf_overwrite_session,
        template_name="Source Folder"
    )

    # Переименование файла базы данных сессии C1
    try:
        original_db_file = final_session_path / (TEMPLATE_SESSION_FILE_NAME_BASE + COSESSIONDB_EXT)
        target_db_file = final_session_path / (config.wf_session_name + COSESSIONDB_EXT)
        
        if original_db_file.exists():
            if target_db_file.exists() and not config.wf_overwrite_session:
                logger.debug(f"Файл <i>{target_db_file.name}</i> уже существует.")
                original_db_file.unlink() # Удаляем остаток шаблона, чтобы не мусорить
            else:
                if target_db_file.exists():
                    target_db_file.unlink() # Безопасное удаление перед переименованием (перезапись)
                original_db_file.rename(target_db_file)
                logger.debug(f"Файл <i>_TEMPLATE_SESSION_.cosessiondb</i> переименован в: <i>{target_db_file.name}</i>")
    except Exception as e:
        logger.error(f"ОШИБКА при переименовании файла БД сессии: {e}")
        # Не прерываем весь скрипт, так как папки уже скопированы, но логируем ошибку


    # 5.3. Создание структуры Альбома (PSD)
    # --------------------------------------------------------------------------
    logger.info("\n✅ Создание структуры папок для рабочих файлов альбома (PSD, INDD и т.д.)")
    if config.wf_overwrite_album:
        logger.info("⚠️ <i>Все существующие файлы перезаписаны</i>")

    template_album_path = SCRIPT_DIR / TEMPLATE_BASE_DIR_NAME / TEMPLATE_ALBUM_DIR_NAME
    final_album_path = target_psd_base / config.wf_session_name

    copy_template_structure(
        template_src_dir=template_album_path,
        target_dir=final_album_path,
        overwrite_files=config.wf_overwrite_album,
        template_name="Album Folder"
    )
    logger.info("\n")
    # 1. Вывод структуры папок
    #tv_builder = StandardTreeBuilder(icon_size=28)
    #root_node_session = ResourceNode("Папка<br>фотосессии", final_session_path, "folder", "Папка с исходными файлами фотосессии (RAW-файлы)")
    #root_node_album = ResourceNode("Папка<br>альбома", final_album_path, "folder", "Папка с рабочими материалами альбома (файлы PSD, INDD и т.д.)")
    #tv_builder.add_section("<br>Рабочие папки и файлы", [root_node_session, root_node_album])
    #pysm_context.log_html(tv_builder.get_html())



    sys.exit(0)


# --- Блок 6: Точка входа ---
# ==============================================================================
if __name__ == "__main__":
    main()