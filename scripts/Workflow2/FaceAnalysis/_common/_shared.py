# analize/_common/_shared.py
"""
Этот модуль содержит общие классы и Pydantic-модели,
используемые на разных этапах конвейера анализа лиц.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import toml
from pydantic import BaseModel, Field

from .status_icons import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

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
            raise FileNotFoundError(f"{icon_error} Файл конфигурации не найден: {config_path}")
        
        try:
            self.config = toml.load(config_path)
            AppConfig(**self.config)  # Валидация структуры
            logger.info(f"{icon_ok} Конфигурация успешно загружена и валидирована из {config_path.name}")
        except Exception as e:
            logger.error(f"{icon_error} Ошибка при загрузке или валидации конфигурации: {e}", exc_info=True)
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Возвращает значение из конфигурации по ключу-пути.
        Например: 'clustering.portrait.dbscan.eps'
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default


# --- Блок 4: Класс для работы с эмбеддингами ---
# ==============================================================================
class EmbeddingLoader:
    """
    Класс для загрузки И СОХРАНЕНИЯ файлов эмбеддингов (.npy) и
    соответствующих им индексных файлов (.json).
    """
    def __init__(self, embeddings_dir: Path):
        """
        Инициализирует менеджер.
        Args:
            embeddings_dir: Путь к директории, содержащей файлы эмбеддингов.
        """
        if not embeddings_dir.exists():
             pass 
        self.embeddings_dir = embeddings_dir

    def load(self, data_type: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Загружает пару файлов: эмбеддинги и их индекс.
        
        Args:
            data_type: Тип данных. В новой архитектуре используйте 'faces'.
                       (Ищет файлы {data_type}_embeddings.npy и {data_type}_index.json)
        
        Returns:
            Tuple[embeddings, index_data]:
                embeddings: np.ndarray (N, 512)
                index_data: Dict. 
                    Для типа 'faces': { "filename": [idx1, idx2, ...] }
                    Для старых типов: может отличаться.
        """
        npy_path = self.embeddings_dir / f"{data_type}_embeddings.npy"
        idx_path = self.embeddings_dir / f"{data_type}_index.json"

        if not npy_path.exists() or not idx_path.exists():
            logger.warning(f"{icon_error} Файлы для '{data_type}' не найдены в {self.embeddings_dir}")
            return None, None
        try:
            embeddings = np.load(npy_path)
            with idx_path.open("r", encoding="utf-8") as f:
                index_data = json.load(f)

            # Обработка Legacy формата (старая структура group)
            if data_type == "group":
                new_index = {}
                for key, val in index_data.items():
                    try:
                        # Старый ключ: "filename::face_index"
                        if "::" in key:
                            fname, fidx = key.rsplit('::', 1)
                            new_index[f"{fname}::{int(fidx)}"] = val
                        else:
                            new_index[key] = val
                    except ValueError:
                        continue 
                index_data = new_index
            
            # Для нового типа 'faces' (Unified Storage) индекс загружается "как есть":
            # { "img1.jpg": [0], "img2.jpg": [1, 2] }
            
            logger.info(f"️{icon_info} Загружено <b>{embeddings.shape[0]}</b> векторов для <i>{data_type}</i>")
            return embeddings, index_data
        except Exception as e:
            logger.error(f"{icon_error} Ошибка загрузки эмбеддингов для '{data_type}': {e}", exc_info=True)
            return None, None

    def save(self, data_type: str, embeddings: np.ndarray, index: Dict[str, Any]) -> bool:
        """
        Сохраняет массив эмбеддингов и индексный словарь на диск.
        """
        try:
            if not self.embeddings_dir.exists():
                self.embeddings_dir.mkdir(parents=True, exist_ok=True)

            npy_path = self.embeddings_dir / f"{data_type}_embeddings.npy"
            idx_path = self.embeddings_dir / f"{data_type}_index.json"

            # 1. Сохраняем .npy
            np.save(npy_path, embeddings)

            # 2. Сохраняем .json
            with idx_path.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

            logger.info(f"{icon_save} Сохранено <b>{embeddings.shape[0]}</b> векторов для <i>{data_type}</i>")
            return True
        except Exception as e:
            logger.critical(f"{icon_save_error} Критическая ошибка при сохранении эмбеддингов '{data_type}': {e}", exc_info=True)
            return False