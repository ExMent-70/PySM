# pysm_lib/pysm_operations.py

"""
Этот модуль является частью API PyScriptManager и предоставляет
высокоуровневые функции для выполнения файловых операций.
"""

# 1. БЛОК: Импорты
# ==============================================================================
import concurrent.futures
from datetime import datetime
import pathlib
import re
import shutil
import sys
import os
import uuid


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


# 3. БЛОК: Приватные вспомогательные функции для `perform_batch_rename_operation`
# ==============================================================================
_WINDOWS_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_windows_filename(filename: str, replacement: str = "_") -> str:
    """Заменяет символы, которые нельзя использовать в имени файла Windows."""
    safe_name = _WINDOWS_INVALID_FILENAME_CHARS.sub(replacement, filename)
    safe_name = safe_name.rstrip(" .")
    return safe_name


def _format_batch_rename_index(index: int, digits: int) -> str:
    """Форматирует порядковый номер для шаблона переименования."""
    if digits <= 0:
        return str(index)
    return str(index).zfill(digits)


def _render_batch_rename_template(
    template: str,
    source_path: pathlib.Path,
    index: int,
    index_digits: int,
    prefix: str,
    suffix: str,
    lowercase_extension: bool,
) -> str:
    """Подставляет внутренние токены batch rename в имя файла."""
    stat_result = source_path.stat()
    modified_dt = datetime.fromtimestamp(stat_result.st_mtime)
    ext = source_path.suffix.lower() if lowercase_extension else source_path.suffix
    replacements = {
        "%index%": _format_batch_rename_index(index, index_digits),
        "%stem%": source_path.stem,
        "%name%": source_path.name,
        "%ext%": ext,
        "%prefix%": prefix or "",
        "%suffix%": suffix or "",
        "%created_date%": modified_dt.strftime("%Y-%m-%d"),
        "%created_time%": modified_dt.strftime("%H-%M-%S"),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


# 4. БЛОК: Публичная функция API для операций с директориями
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


# 5. БЛОК: Публичная функция API для операций с файлами
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


# 6. БЛОК: Публичная функция API для пакетного переименования файлов
# ==============================================================================
def perform_batch_rename_operation(
    source_dir_str: str,
    include_patterns: list[str],
    rename_template: str,
    start_index: int,
    index_digits: int,
    prefix: str = "",
    suffix: str = "",
    recursive: bool = False,
    on_conflict: str = "error",
    dry_run: bool = False,
    sort_method: str = "modified_time",
    lowercase_extension: bool = False,
    sanitize_filename: bool = True,
) -> int:
    """
    Переименовывает набор файлов по шаблону после выбранной сортировки.

    Внутренние токены шаблона используют синтаксис %token%, чтобы не конфликтовать
    с переменными контекста PySM вида {var_name}.

    :return: 0 при успехе, 1 при ошибке.
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

    print("<b>ПАКЕТНОЕ ПЕРЕИМЕНОВАНИЕ ФАЙЛОВ</b>")

    source_dir = pathlib.Path(source_dir_str)
    if not source_dir.is_dir():
        html = f'<div style="{name_style}"><br>{icon_error} Операция отменена. Исходная папка не найдена:</div>'
        html += f'<div style="{name_style}"><i>{source_dir}</i><br></div>'
        pysm_context.log_html(html)
        return 1

    if not rename_template:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_error} Операция отменена. Шаблон имени не задан.</div>'
        )
        return 1

    if pathlib.PureWindowsPath(rename_template).name != rename_template:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_error} Шаблон должен задавать только имя файла, без папок: <b>{rename_template}</b></div>'
        )
        return 1

    include_patterns = include_patterns or ["*"]
    html = ""
    html += f'<div style="{name_style}">{icon_sub} шаблон имени: <b>{rename_template}</b></div>'
    html += f'<div style="{name_style}">{icon_sub} паттерны файлов: <b>{include_patterns}</b></div>'
    html += f'<div style="{name_style}">{icon_sub} метод сортировки: <b>{sort_method}</b></div>'
    html += f'<div style="{name_style}">{icon_sub} рекурсивно: <b>{recursive}</b></div>'
    html += f'<div style="{name_style}">{icon_sub} dry-run: <b>{dry_run}</b></div>'
    pysm_context.log_html(html)

    items_to_process = []
    for pattern in include_patterns:
        iterator = source_dir.rglob(pattern) if recursive else source_dir.glob(pattern)
        items_to_process.extend(list(iterator))
    items_to_process = list(set(item for item in items_to_process if item.is_file()))

    if sort_method == "created_time":
        items_to_process = sorted(
            items_to_process,
            key=lambda item: (item.stat().st_ctime_ns, item.name.lower()),
        )
    elif sort_method == "modified_time":
        items_to_process = sorted(
            items_to_process,
            key=lambda item: (item.stat().st_mtime_ns, item.name.lower()),
        )
    elif sort_method == "name":
        items_to_process = sorted(
            items_to_process,
            key=lambda item: str(item.relative_to(source_dir)).lower(),
        )
    elif sort_method == "none":
        items_to_process = list(items_to_process)
    else:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_error} Неизвестный метод сортировки: <b>{sort_method}</b></div>'
        )
        return 1

    if not items_to_process:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_warning} Файлы по паттернам <b>{include_patterns}</b> не найдены.</div>'
        )
        return 0

    pysm_context.log_html(
        f'<div style="{name_style}">{icon_info} Найдено файлов для переименования: <b>{len(items_to_process)}</b></div>'
    )

    candidate_entries = []
    assigned_targets: set[pathlib.Path] = set()
    plan = []
    stats = {"planned": 0, "skipped": 0, "error": 0}

    for offset, source_path in enumerate(items_to_process):
        current_index = start_index + offset
        rendered_name = _render_batch_rename_template(
            template=rename_template,
            source_path=source_path,
            index=current_index,
            index_digits=index_digits,
            prefix=prefix,
            suffix=suffix,
            lowercase_extension=lowercase_extension,
        )

        if sanitize_filename:
            rendered_name = _sanitize_windows_filename(rendered_name)

        if not rendered_name:
            tqdm.write(f"[FAIL] Empty target filename for {source_path}")
            stats["error"] += 1
            continue

        if pathlib.PureWindowsPath(rendered_name).name != rendered_name:
            tqdm.write(f"[FAIL] Target filename contains path separators: {rendered_name}")
            stats["error"] += 1
            continue

        target_path = source_path.with_name(rendered_name)
        candidate_entries.append((source_path, target_path))

    renamed_sources = {
        source_path.resolve()
        for source_path, target_path in candidate_entries
        if source_path.resolve() != target_path.resolve()
    }

    for source_path, target_path in candidate_entries:
        target_resolved = target_path.resolve()

        while True:
            target_already_assigned = target_resolved in assigned_targets
            target_is_current_source = source_path.resolve() == target_resolved
            target_is_external_existing = (
                target_path.exists()
                and not target_is_current_source
                and target_resolved not in renamed_sources
            )
            if not target_already_assigned and not target_is_external_existing:
                break
            if on_conflict == "skip":
                stats["skipped"] += 1
                target_path = None
                break
            if on_conflict == "rename":
                target_path = _get_unique_path_for_dir_op(target_path)
                target_resolved = target_path.resolve()
                continue
            stats["error"] += 1
            tqdm.write(f"[FAIL] Target file already exists: {target_path}")
            target_path = None
            break

        if target_path is None:
            continue

        assigned_targets.add(target_resolved)
        if source_path.resolve() == target_resolved:
            stats["skipped"] += 1
            continue

        plan.append((source_path, target_path))
        stats["planned"] += 1

    if stats["error"] > 0:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_error} План переименования содержит ошибки: <b>{stats["error"]}</b>. Файлы не изменены.</div>'
        )
        return 1

    if not plan:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_ok} Нет файлов, требующих переименования. Пропущено: <b>{stats["skipped"]}</b></div>'
        )
        return 0

    preview_lines = []
    for source_path, target_path in plan[:10]:
        preview_lines.append(f"{source_path.name} -> {target_path.name}")
    preview_html = "<br>".join(preview_lines)
    if len(plan) > 10:
        preview_html += f"<br>... и ещё {len(plan) - 10}"
    pysm_context.log_html(
        f'<div style="{name_style}">{icon_info} План переименования:<br><code>{preview_html}</code></div>'
    )

    if dry_run:
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_ok} Dry-run завершён. Запланировано переименований: <b>{len(plan)}</b>, пропущено: <b>{stats["skipped"]}</b></div>'
        )
        pysm_context.log_link(
            url_or_path=str(source_dir),
            text=f"{icon_folder} Исходная папка<br>",
        )
        return 0

    temp_plan = []
    operation_id = uuid.uuid4().hex

    try:
        prepare_progress = tqdm(
            enumerate(plan, start=1),
            total=len(plan),
            desc="Preparing rename",
            unit="file",
        )
        for step_index, (source_path, target_path) in prepare_progress:
            temp_path = source_path.with_name(
                f".pysm_rename_tmp_{operation_id}_{step_index}{source_path.suffix}"
            )
            while temp_path.exists():
                temp_path = source_path.with_name(
                    f".pysm_rename_tmp_{uuid.uuid4().hex}_{step_index}{source_path.suffix}"
                )
            source_path.rename(temp_path)
            temp_plan.append((temp_path, source_path, target_path))

        progress_bar = tqdm(temp_plan, total=len(temp_plan), desc="Renaming", unit="file")
        for temp_path, _source_path, target_path in progress_bar:
            temp_path.rename(target_path)

    except Exception as e:
        tqdm.write(f"[FATAL] Batch rename failed: {type(e).__name__}: {e}")
        for temp_path, source_path, _target_path in reversed(temp_plan):
            if temp_path.exists() and not source_path.exists():
                try:
                    temp_path.rename(source_path)
                except Exception as rollback_error:
                    tqdm.write(f"[FAIL] Rollback failed for {temp_path}: {rollback_error}")
        pysm_context.log_html(
            f'<div style="{name_style}">{icon_error} Переименование прервано: <b>{type(e).__name__}</b>: {e}</div>'
        )
        return 1

    pysm_context.log_html(
        f'<div style="{name_style}">{icon_ok} Переименовано: <b>{len(plan)}</b>, пропущено: <b>{stats["skipped"]}</b></div>'
    )
    pysm_context.log_link(
        url_or_path=str(source_dir),
        text=f"{icon_folder} Исходная папка<br>",
    )
    return 0
