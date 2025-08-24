# pysm_lib/context_loader.py

import sys
import runpy
import os
import types
from .locale_manager import LocaleManager

locale_manager = LocaleManager()

if os.environ.get("PY_SCRIPT_MANAGER_ACTIVE") == "1":
    try:
        # --- НАЧАЛО ИЗМЕНЕНИЙ: Создание полной структуры прокси-пакета ---
        from importlib.util import find_spec
        from .services.tqdm_patch_dir import tqdm, trange, auto, notebook, contrib

        # 1. Создаем в памяти прокси-объекты для КАЖДОГО уровня, который может быть импортирован
        tqdm_proxy = types.ModuleType('tqdm')
        contrib_proxy = types.ModuleType('tqdm.contrib')
        
        # 2. Наполняем главный прокси-модуль 'tqdm'
        tqdm_proxy.tqdm = tqdm
        tqdm_proxy.trange = trange
        tqdm_proxy.auto = auto
        tqdm_proxy.notebook = notebook
        tqdm_proxy.contrib = contrib_proxy  # 'contrib' теперь тоже является модулем

        # 3. Наполняем подмодуль 'tqdm.contrib'
        contrib_proxy.concurrent = contrib.concurrent

        # 4. Копируем метаданные (__spec__) из НАСТОЯЩИХ модулей в наши прокси.
        #    Это удовлетворит и torch, и huggingface_hub.
        real_tqdm_spec = find_spec('tqdm')
        if real_tqdm_spec:
            tqdm_proxy.__spec__ = real_tqdm_spec
            # Указываем, что это пакет, чтобы из него можно было импортировать
            tqdm_proxy.__path__ = real_tqdm_spec.submodule_search_locations

        real_contrib_spec = find_spec('tqdm.contrib')
        if real_contrib_spec:
            contrib_proxy.__spec__ = real_contrib_spec
            contrib_proxy.__path__ = real_contrib_spec.submodule_search_locations
        
        # 5. Регистрируем ВСЮ нашу структуру в sys.modules.
        sys.modules['tqdm'] = tqdm_proxy
        sys.modules['tqdm.contrib'] = contrib_proxy
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    except Exception as e:
        print(f"PySM Loader Warning: Could not patch tqdm: {e}", file=sys.stderr)


if len(sys.argv) < 2:
    print(locale_manager.get("context_loader.error.script_path_missing"), file=sys.stderr)
    sys.exit(99)

path_to_run = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(path_to_run, run_name="__main__")