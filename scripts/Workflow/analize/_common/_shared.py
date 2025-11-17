# analize/_common/_shared.py
"""
Этот модуль содержит общие классы и Pydantic-модели,
используемые на разных этапах конвейера анализа лиц.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import toml
from pydantic import BaseModel, Field

# --- Блок 1: Настройка логирования ---
# ==============================================================================
logger = logging.getLogger(__name__)


# --- Блок 2: Pydantic-модели для конфигурации ---
# ==============================================================================
class DbscanConfig(BaseModel):
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "cosine"


class HdbscanConfig(BaseModel):
    min_cluster_size: int = 5
    min_samples: Optional[int] = None
    metric: str = "cosine"
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = False


class PortraitClusteringConfig(BaseModel):
    dbscan: DbscanConfig = Field(default_factory=DbscanConfig)
    hdbscan: HdbscanConfig = Field(default_factory=HdbscanConfig)


class ClusteringConfig(BaseModel):
    portrait: PortraitClusteringConfig = Field(default_factory=PortraitClusteringConfig)


class MatchingConfig(BaseModel):
    match_threshold: float = 0.5
    use_auto_threshold: bool = False
    percentile: int = 10


class AppConfig(BaseModel):
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)


# --- Блок 3: Класс для управления конфигурацией ---
# ==============================================================================
class ConfigManager:
    """
    Класс для загрузки, валидации и доступа к параметрам
    из TOML-файла конфигурации.
    """
    def __init__(self, config_path: Path):
        """
        Инициализирует менеджер конфигурации.

        Args:
            config_path: Путь к TOML-файлу конфигурации.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            Exception: При ошибках парсинга или валидации.
        """
        if not config_path.is_file():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        try:
            self.config = toml.load(config_path)
            AppConfig(**self.config)  # Валидация структуры
            logger.info(f"Конфигурация успешно загружена и валидирована из {config_path.name}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке или валидации конфигурации: {e}", exc_info=True)
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Возвращает значение из конфигурации по ключу-пути.
        Например: 'clustering.portrait.dbscan.eps'

        Args:
            key_path: Строковый путь до ключа, разделенный точками.
            default: Значение по умолчанию, если ключ не найден.

        Returns:
            Значение из конфигурации или значение по умолчанию.
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default


# --- Блок 4: Класс для загрузки эмбеддингов ---
# ==============================================================================
class EmbeddingLoader:
    """
    Класс для загрузки файлов эмбеддингов (.npy) и
    соответствующих им индексных файлов (.json).
    """
    def __init__(self, embeddings_dir: Path):
        """
        Инициализирует загрузчик.

        Args:
            embeddings_dir: Путь к директории, содержащей файлы эмбеддингов.
        """
        if not embeddings_dir.is_dir():
            raise FileNotFoundError(f"Директория с эмбеддингами не найдена: {embeddings_dir}")
        self.embeddings_dir = embeddings_dir

    def load(self, data_type: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Загружает пару файлов: эмбеддинги и их индекс.

        Args:
            data_type: Тип данных для загрузки ('portrait' или 'group').

        Returns:
            Кортеж (numpy.ndarray, dict) в случае успеха, иначе (None, None).
        """
        npy_path = self.embeddings_dir / f"{data_type}_embeddings.npy"
        idx_path = self.embeddings_dir / f"{data_type}_index.json"

        if not npy_path.exists() or not idx_path.exists():
            logger.warning(f"Файлы для '{data_type}' не найдены в {self.embeddings_dir}")
            return None, None
        try:
            embeddings = np.load(npy_path)
            with idx_path.open("r", encoding="utf-8") as f:
                index_data = json.load(f)

            # Приведение ключей индекса к корректному формату для групповых фото
            if data_type == "group":
                index_data = {
                    f"{key.split('::')[0]}::{int(key.split('::')[1])}": val
                    for key, val in index_data.items()
                }

            logger.info(f"Загружены эмбеддинги для '{data_type}': {embeddings.shape[0]} векторов.")
            return embeddings, index_data
        except Exception as e:
            logger.error(f"Ошибка загрузки эмбеддингов для '{data_type}': {e}", exc_info=True)
            return None, None