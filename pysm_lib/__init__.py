# pysm_lib/__init__.py

# --- НОВЫЙ БЛОК: Делаем экземпляр контекста доступным на уровне пакета ---
from .pysm_context import pysm_context

__all__ = ["pysm_context"]
# Это позволяет делать импорт: from pysm_lib import pysm_context
