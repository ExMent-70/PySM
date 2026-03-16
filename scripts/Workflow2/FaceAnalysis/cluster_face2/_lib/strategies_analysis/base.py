# cluster_face/_lib/strategies_analysis/base.py

from abc import ABC, abstractmethod
from argparse import Namespace
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis_manager import AnalysisDataManager

logger = logging.getLogger(__name__)

class AnalysisStrategy(ABC):
    """
    Абстрактный базовый класс для стратегий анализа лиц.
    """

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Имя режима, соответствующее аргументу CLI (cleaning, face, matches)."""
        pass

    @abstractmethod
    def run(self, config: Namespace, data_manager: "AnalysisDataManager") -> None:
        """
        Запуск логики стратегии.
        
        Args:
            config: Объект с аргументами командной строки (содержит ВСЕ параметры).
                    Стратегия должна брать только свои (с префиксом a_).
            data_manager: Инициализированный менеджер данных с загруженным JSON и Embeddings.
        """
        pass
    
    def log_header(self):
        """Выводит заголовок режима в лог."""
        logger.info(f"<br><b>РЕЖИМ РАБОТЫ: {self.mode_name.upper()}</b>")