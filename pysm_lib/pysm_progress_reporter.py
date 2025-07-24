# pysm_lib/pysm_progress_reporter.py
import os
import sys
import json
from typing import Optional, Any

from .locale_manager import LocaleManager

locale_manager = LocaleManager()
IS_RUNNING_UNDER_PYSM = os.environ.get("PY_SCRIPT_MANAGER_ACTIVE") == "1"


class JsonProgressReporter:
    def __init__(
        self,
        iterable: Optional[Any] = None,
        total: Optional[int] = None,
        desc: str = locale_manager.get("progress_reporter.default_description"),
        unit: str = "it",
        ncols: Optional[int] = None,
        bar_format: Optional[str] = None,
        leave: bool = True,
        **kwargs: Any,
    ):
        self.actual_iterable = iterable
        self.total: int
        self.n: int = 0  # Используем 'n' для совместимости с tqdm
        self.desc: str = desc  # Используем 'desc' для совместимости
        self.unit: str = unit
        self._pysm_initial_sent: bool = False
        self._closed: bool = False

        if total is not None:
            self.total = total
        elif iterable is not None:
            try:
                self.total = len(iterable)
            except (TypeError, AttributeError):
                self.total = 0
        else:
            self.total = 0

        # Переименовываем переменные для совместимости
        self.current = self.n
        self.description = self.desc

        if IS_RUNNING_UNDER_PYSM and not self._pysm_initial_sent:
            # Отправляем начальное состояние даже для total=0, чтобы показать неопределенный бар
            self._send_progress_json()
            self._pysm_initial_sent = True

    def _send_progress_json(self):
        if self._closed:
            return

        # Обновляем значения перед отправкой
        self.n = self.current
        self.desc = self.description

        progress_data = {
            "type": "progress",
            "current": self.n,
            "total": self.total,
            "text": f"{self.desc}: {self.n}/{self.total} {locale_manager.get('progress_reporter.unit_of', unit=self.unit)}",
        }

        # Специальный режим для неопределенного прогресс-бара
        if self.total == 0:
            progress_data["text"] = self.desc

        print(json.dumps(progress_data), file=sys.stderr, flush=True)

    def update(self, n: int = 1) -> bool:
        if self._closed:
            return False

        if not self._pysm_initial_sent:
            self._send_progress_json()
            self._pysm_initial_sent = True

        self.current += n
        if self.total > 0:
            self.current = min(self.current, self.total)

        self.current = max(0, self.current)

        self._send_progress_json()
        return True

    def reset(self, total: Optional[int] = None):
        """
        Сбрасывает счетчик и опционально устанавливает новый total.
        Имитирует tqdm.reset().
        """
        self.current = 0
        if total is not None:
            self.total = total
        self._pysm_initial_sent = False
        self._closed = False
        self.update(0)

    def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
        if desc is not None:
            self.description = desc
        if refresh:
            self._send_progress_json()

    def set_postfix(
        self, ordered_dict: Optional[dict] = None, refresh: bool = True, **kwargs: Any
    ) -> None:
        postfix_str = ""
        if ordered_dict:
            postfix_str = ", ".join([f"{k}: {v}" for k, v in ordered_dict.items()])
        elif kwargs:
            postfix_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])

        current_base_description = self.description.split(" (")[0]
        full_description = (
            f"{current_base_description} ({postfix_str})"
            if postfix_str
            else current_base_description
        )

        if self.description != full_description:
            self.description = full_description
            if refresh:
                self._send_progress_json()

    def write(
        s: str, file: Optional[Any] = None, end: str = "\n", nolock: bool = False
    ) -> None:
        """
        Статический метод для вывода сообщений в stderr.
        Не зависит от экземпляра класса.
        """
        # Важно! tqdm.write не должен вызывать JSON-прогресс.
        # Просто печатаем сообщение в stderr, чтобы оно появилось в консоли, но не мешало JSON.
        print(s, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._closed:
            return

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # КОММЕНТАРИЙ: Команда сброса прогресс-бара также должна идти в stderr.
        print(
            json.dumps({"type": "progress", "current": 0, "total": 0, "text": ""}),
            file=sys.stderr,
            flush=True,
        )
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        self._closed = True

    def __iter__(self) -> Any:
        if self.actual_iterable is None:
            self.current = 0
            return self
        else:
            self._iterator = iter(self.actual_iterable)
            return self

    def __next__(self) -> Any:
        if self.actual_iterable is None:
            if self.current < self.total:
                self.update(1)
                return self.current - 1
            else:
                raise StopIteration
        else:
            try:
                val = next(self._iterator)
                self.update(1)
                return val
            except StopIteration:
                self.close()
                raise

    def __len__(self) -> int:
        return self.total

    def __enter__(self) -> "JsonProgressReporter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False


def tqdm(
    iterable: Optional[Any] = None,
    use_tqdm_if_not_managed: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Возвращает либо JsonProgressReporter, либо настоящий tqdm, в зависимости от окружения.
    Имитирует основной интерфейс tqdm.
    """
    if IS_RUNNING_UNDER_PYSM:
        return JsonProgressReporter(iterable=iterable, **kwargs)
    else:
        if use_tqdm_if_not_managed:
            try:
                from tqdm import tqdm as original_tqdm

                return original_tqdm(iterable=iterable, **kwargs)
            except ImportError:
                return JsonProgressReporter(iterable=iterable, **kwargs)
        else:
            return JsonProgressReporter(iterable=iterable, **kwargs)


def trange(*args, **kwargs):
    """Совместимость с tqdm.trange"""
    return tqdm(range(*args), **kwargs)


tqdm.write = JsonProgressReporter.write
