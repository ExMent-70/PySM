"""Small progress protocol shared by processing and model downloads."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class ProgressReporter(Protocol):
    def set_description(self, desc: str, refresh: bool = True) -> None: ...

    def set_postfix(
        self,
        ordered_dict: dict[str, int] | None = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def update(self, value: int = 1) -> Any: ...

    def close(self) -> None: ...


ProgressFactory = Callable[..., ProgressReporter]
