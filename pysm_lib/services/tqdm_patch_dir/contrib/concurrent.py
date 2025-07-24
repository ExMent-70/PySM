# pysm_lib/services/tqdm_patch_dir/contrib/concurrent.py
"""
Поддельный подмодуль concurrent для совместимости.
"""

# --- НАЧАЛО ИЗМЕНЕНИЙ ---
# КОММЕНТАРИЙ: Изменен импорт, чтобы разорвать цикл.
# Мы импортируем конкретную реализацию, а не родительский пакет.
# Четыре точки означают подъем на 4 уровня вверх от текущего файла
# до корня pysm_lib.
from ....pysm_progress_reporter import tqdm
# --- КОНЕЦ ИЗМЕНЕНИЙ ---


def thread_map(func, *iterables, **tqdm_kwargs):
    """
    Простая заглушка для thread_map.
    Вместо многопоточности, выполняет обычный map и оборачивает его в наш tqdm
    для отображения прогресса.
    """
    iterable = list(zip(*iterables))
    tqdm_kwargs.pop("max_workers", None)
    tqdm_kwargs.pop("chunksize", None)
    progress_iterator = tqdm(iterable, **tqdm_kwargs)
    return list(map(lambda x: func(*x), progress_iterator))
