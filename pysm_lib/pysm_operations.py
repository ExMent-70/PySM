# pysm_lib/pysm_operations.py

"""
Этот модуль является частью API PyScriptManager и предоставляет
высокоуровневые функции для выполнения файловых операций.
"""

# 1. БЛОК: Импорты
# ==============================================================================
import concurrent.futures
import pathlib
import shutil
import sys
import os


# Импортируем зависимые компоненты из нашей же библиотеки
from .pysm_progress_reporter import tqdm
from . import pysm_context
from .pysm_icons import icons
from .pysm_theme_api import theme_api



# 2. БЛОК: Приватные вспомогательные функции для `perform_directory_operation`
# ==============================================================================
def _get_unique_path_for_dir_op(path: pathlib.Path) -> pathlib.Path:
    """Генерирует уникальный путь, если исходный уже существует."""
    if not path.exists():
        return path
    parent, stem, ext = path.parent, path.stem, path.suffix
    i = 1
    while True:
        new_path = parent / f"{stem} ({i}){ext}"
        if not new_path.exists():
            return new_path
        i += 1


def _process_dir_item(
    source_path: pathlib.Path,
    source_root: pathlib.Path,
    dest_root: pathlib.Path,
    mode: str,
    on_conflict: str,
) -> tuple[str, str]:
    """Обрабатывает один файл в рамках операции с директорией."""
    try:
        relative_path = source_path.relative_to(source_root)
        dest_path = dest_root / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            if on_conflict == "skip":
                return "skipped", f"{relative_path}"
            elif on_conflict == "rename":
                dest_path = _get_unique_path_for_dir_op(dest_path)

        if mode == "copy":
            shutil.copy2(source_path, dest_path)
        else:
            shutil.move(str(source_path), str(dest_path))
        return "success", f"{relative_path}"
    except Exception as e:
        return "error", f"ERROR on {source_path.name}: {e}"




# --- НАЧАЛО ИЗМЕНЕНИЙ: Полностью переработанная функция _cleanup_empty_dirs ---
def _cleanup_empty_dirs(path_to_clean: pathlib.Path):
    """
    Рекурсивно удаляет все пустые поддиректории, включая те,
    которые становятся пустыми после удаления их дочерних элементов.
    """
    cleaned_count = 0
    # Собираем все директории внутри path_to_clean
    all_dirs = [d[0] for d in os.walk(str(path_to_clean))]
    
    # Сортируем их по длине пути в обратном порядке (от самых глубоких к корню)
    # Это гарантирует, что мы сначала удалим дочерние пустые папки
    all_dirs.sort(key=len, reverse=True)

    for dirpath in all_dirs:
        try:
            # os.rmdir() сработает, только если директория пуста
            os.rmdir(dirpath)
            cleaned_count += 1
        except OSError:
            # Игнорируем ошибку, если директория не пуста
            continue

    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} empty subdirectories in the source folder.")
# --- КОНЕЦ ИЗМЕНЕНИЙ ---

# 3. БЛОК: Публичная функция API для операций с директориями
# ==============================================================================
def perform_directory_operation(
    source_dir_str: str,
    dest_dir_str: str,
    mode: str,
    on_conflict: str,
    threads: int,
    copy_base_folder: bool,
    include_patterns: list[str],
) -> int:
    """
    Выполняет многопоточную операцию (копирование/перемещение) с директорией.

    :return: 0 при успехе, 1 при наличии ошибок.
    """
    text_main = theme_api.get_parsed_style("script_stdout", default="color: #2c3e50")
    text_color = text_main.get("color")
    name_style = f"color: {text_color};"
    icon_error = icons.ERROR()
    icon_info = icons.INFO()
    icon_sub = icons.ARROW_SUB()
    icon_ok = icons.OK()
    icon_folder = icons.FOLDER()
    icon_warning = icons.WARNING()

    if mode == "move":
        action_desc = "<b>ПЕРЕМЕЩЕНИЕ ФАЙЛОВ И ПАПОК</b>"
    else:
        action_desc = "<b>КОПИРОВАНИЕ ФАЙЛОВ И ПАПОК</b>"
    print(action_desc)

    
    #print(f"\nРежим работы: <b>{mode}</b>,<br> Количество потоков: <b>{threads}</b>,<br> Действие при конфликте: <b>{on_conflict}</b>")

    source_dir = pathlib.Path(source_dir_str)
    dest_dir = pathlib.Path(dest_dir_str)

    if not source_dir.is_dir():
        html = (f'<div style="{name_style}"><br>{icon_error} Операция отменена. Исходная папка не найдена:</div>')
        html = html + (f'<div style="{name_style}"><i>{source_dir}</i><br></div>')
        pysm_context.log_html(html)       
        return 1

    final_dest_root = dest_dir / source_dir.name if copy_base_folder else dest_dir
    html=""
    html = html + (f'<div style="{name_style}">{icon_sub} корневую папку: <b>{copy_base_folder}</b></div>')
    html =  html +  (f'<div style="{name_style}">{icon_sub} только файлы: <b>{include_patterns}</b></div>')
    pysm_context.log_html(html)       


    items_to_process = []
    for pattern in include_patterns:
        items_to_process.extend(list(source_dir.rglob(pattern)))
    items_to_process = sorted(
        list(set(item for item in items_to_process if item.is_file()))
    )

    if not items_to_process:
        html = (f'<div style="{name_style}">{icon_error} В целевой папке файлы <b>{include_patterns}</b> не найдены<br></div>')
        pysm_context.log_html(html) 
        if copy_base_folder and not final_dest_root.exists():
            final_dest_root.mkdir(parents=True, exist_ok=True)
            html = (f'<div style="{name_style}">{icon_info} Создана целевая папка: <b>{final_dest_root}</b></div>')
            pysm_context.log_html(html) 
        return 0

    if mode == "move":
        action_desc = "для перемещения"
    else:
        action_desc = "для копирования"
    html = (f'<div style="{name_style}">{icon_info} Найдено файлов {action_desc}: <b>{len(items_to_process)}</b></div>')
    pysm_context.log_html(html) 

    stats = {"success": 0, "error": 0, "skipped": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_item = {
            executor.submit(
                _process_dir_item, item, source_dir, final_dest_root, mode, on_conflict
            ): item
            for item in items_to_process
        }
        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_item),
            total=len(items_to_process),
            desc="Processing",
            unit="file",
            dynamic_ncols=True,
        )
        for future in progress_bar:
            try:
                status, message = future.result()
                stats[status] += 1
                if status == "error":
                    tqdm.write(f"[FAIL] {message}")
                progress_bar.set_postfix(
                    ok=stats["success"], failed=stats["error"], skipped=stats["skipped"]
                )
            except Exception as e:
                stats["error"] += 1
                tqdm.write(f"[FATAL] An unexpected error occurred: {e}")


    # --- НАЧАЛО ИЗМЕНЕНИЙ: Полностью переработана логика завершения для 'move' ---
    if mode == "move" and stats["error"] == 0 and stats["success"] > 0:
        # Вместо удаления всей папки, мы аккуратно удаляем только пустые поддиректории,
        # которые могли остаться после перемещения файлов.
        # Это предотвращает потерю данных, не соответствующих фильтру.
        #print("Перемещение файлов выполнено. Удаление пустых папкок...")
        _cleanup_empty_dirs(source_dir)
        
    elif mode == "move" and stats["error"] > 0:
        html = (f'<div style="{name_style}">{icon_warning}Перемещение выполнено с ошибками. Структура исходного каталога не изменилась</div>')
        pysm_context.log_html(html) 

    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    if mode == "move":
        action_desc = "перемещено"
    else:
        action_desc = "скопировано"
    sum_str = f"<b>{stats['success']}</b> {action_desc}, <b>{stats['skipped']}</b> пропущено, <b>{stats['error']}</b> ошибок"
    html = (f'<div style="{name_style}">{icon_ok}{sum_str}</div>')
    pysm_context.log_html(html) 

    #print("[Directory Operation Finished]")
    pysm_context.log_link(
        url_or_path=str(source_dir), # Передаем строку, а не объект Path
        text=f"{icon_folder} Исходная папка",
    )
    pysm_context.log_link(
        url_or_path=str(final_dest_root), # Передаем строку, а не объект Path
        text=f"{icon_folder} Целевая папка<br>",
    )    
    return 1 if stats["error"] > 0 else 0


# 4. БЛОК: Публичная функция API для операций с файлами
# ==============================================================================
def perform_file_operation(
    operation: str,
    source_path_str: str,
    destination_path_str: str,
    overwrite: bool,
    create_parents: bool,
) -> int:
    """
    Выполняет одиночную операцию с файлом (copy, move, rename, delete).

    :return: 0 при успехе, 1 при ошибке.
    """
    tqdm.write(f"--- File Operation Starting: {operation} ---")

    try:
        src = pathlib.Path(source_path_str) if source_path_str else None
        dst = pathlib.Path(destination_path_str) if destination_path_str else None

        if operation in ["copy", "move", "rename", "delete"]:
            if not src or not src.exists():
                raise FileNotFoundError(f"Source path does not exist: {src}")
            if not src.is_file():
                raise TypeError(f"Source path is not a file: {src}")

        if operation == "delete":
            tqdm.write(f"Deleting file: {src}")
            src.unlink()

        elif operation in ["copy", "move", "rename"]:
            if not dst:
                raise ValueError(
                    "Destination path must be specified for this operation."
                )
            if dst.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination file exists and overwrite is false: {dst}"
                )
            if create_parents:
                dst.parent.mkdir(parents=True, exist_ok=True)

            tqdm.write(f"Processing file from '{src}' to '{dst}'")
            if operation == "copy":
                shutil.copy2(src, dst)
            else:  # move и rename для shutil - одна и та же операция
                shutil.move(str(src), str(dst))

        tqdm.write(f"--- Operation '{operation}' completed successfully. ---")
        return 0

    except Exception as e:
        tqdm.write(
            f"--- ERROR during '{operation}' operation: {type(e).__name__}: {e} ---"
        )
        return 1
