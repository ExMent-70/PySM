# pysm_lib/services/tqdm_patch_dir/__init__.py
"""
Главный __init__.py для нашего поддельного пакета tqdm.
Он предоставляет объекты, которые будут подставлены в sys.modules.
"""

from ...pysm_progress_reporter import JsonProgressReporter

class tqdm:
    def __new__(cls, *args, **kwargs):
        return JsonProgressReporter(*args, **kwargs)

    @staticmethod
    def write(*args, **kwargs):
        return JsonProgressReporter.write(*args, **kwargs)

def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)

class _TqdmSubModule:
    def __init__(self):
        self.tqdm = tqdm
        self.trange = trange

auto = _TqdmSubModule()
notebook = _TqdmSubModule()
