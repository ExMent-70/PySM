# pysm_lib/context_loader.py

import sys
import runpy
import os
from .locale_manager import LocaleManager

locale_manager = LocaleManager()

if os.environ.get("PY_SCRIPT_MANAGER_ACTIVE") == "1":
    try:
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Импортируем модули-заглушки напрямую
        from .services.tqdm_patch_dir import auto, notebook
        from .services.tqdm_patch_dir.contrib import concurrent

        # Создаем пустой объект-заглушку для самого 'contrib'
        class ContribModule:
            pass
        
        contrib_mod = ContribModule()
        contrib_mod.concurrent = concurrent # Присваиваем ему наш модуль concurrent

        # Создаем главный объект-заглушку
        class TqdmPatchModule:
            pass
        
        tqdm_patch_dir = TqdmPatchModule()
        tqdm_patch_dir.auto = auto
        tqdm_patch_dir.notebook = notebook
        tqdm_patch_dir.contrib = contrib_mod
        # Функции tqdm и trange будут доступны через импорт `from tqdm import tqdm`
        # благодаря следующему шагу.
        from .services.tqdm_patch_dir import tqdm, trange
        tqdm_patch_dir.tqdm = tqdm
        tqdm_patch_dir.trange = trange
        
        # Регистрируем все заглушки в sys.modules
        sys.modules["tqdm"] = tqdm_patch_dir
        sys.modules["tqdm.auto"] = auto
        sys.modules["tqdm.notebook"] = notebook
        sys.modules["tqdm.contrib"] = contrib_mod
        sys.modules["tqdm.contrib.concurrent"] = concurrent
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    except ImportError as e:
        print(f"PySM Loader Warning: Could not patch tqdm: {e}", file=sys.stderr)


if len(sys.argv) < 2:
    print(locale_manager.get("context_loader.error.script_path_missing"), file=sys.stderr)
    sys.exit(99)

path_to_run = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(path_to_run, run_name="__main__")