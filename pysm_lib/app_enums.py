# pysm_lib/app_enums.py

from enum import Enum, auto


class SetRunMode:
    """
    Определяет режимы запуска для набора скриптов.
    Используется строковыми значениями для совместимости с JSON.
    """

    SEQUENTIAL_FULL = "sequential_full"
    SEQUENTIAL_STEP = "sequential_step"
    SINGLE_FROM_SET = "single_from_set"
    CONDITIONAL_FULL = "conditional_full"
    CONDITIONAL_STEP = "conditional_step"


class AppState(Enum):
    """Определяет общее состояние приложения."""

    IDLE = auto()
    SET_RUNNING_AUTO = auto()
    SET_RUNNING_STEP_WAIT = auto()
    SET_STOPPING = auto()
    SCANNING_SCRIPTS = auto()


class ScriptRunStatus(Enum):
    """Определяет статус выполнения конкретного экземпляра скрипта в наборе."""

    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    ERROR = auto()
    SKIPPED = auto()
