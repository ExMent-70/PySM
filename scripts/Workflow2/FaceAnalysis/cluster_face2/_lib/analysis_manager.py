# cluster_face/_lib/analysis_manager.py

import json
import logging
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Обеспечиваем доступ к _common (на уровень выше, чем cluster_face)
try:
    # cluster_face/_lib/analysis_manager.py -> cluster_face/_lib -> cluster_face -> project_root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

try:
    from _common._shared import EmbeddingLoader
except ImportError:
    EmbeddingLoader = None

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

logger = logging.getLogger(__name__)

class AnalysisDataManager:
    """
    Управляет загрузкой и сохранением данных для всех стратегий анализа лиц.
    Работает с info_faces.json и _Embeddings (npy).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_path = self.data_dir / "info_faces.json"
        self.embeddings_dir = self.data_dir / "_Embeddings"
        
        self.json_data: Dict[str, Any] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.index_map: Dict[str, List[int]] = {}

        # Проверка существования критических путей
        if not self.data_dir.exists():
            raise FileNotFoundError(f"{icon_error} Папка данных не найдена: {self.data_dir}")

    def load_data(self) -> bool:
        """
        Загружает JSON и Эмбеддинги.
        Возвращает True, если загрузка прошла успешно.
        """
        if not self.json_path.exists():
            logger.error(f"{icon_error} Файл метаданных не найден: {self.json_path}")
            return False

        # 1. Загрузка JSON
        try:
            with self.json_path.open("r", encoding="utf-8") as f:
                self.json_data = json.load(f)
        except Exception as e:
            logger.error(f"{icon_error} Ошибка чтения JSON: {e}")
            return False

        # 2. Загрузка Эмбеддингов
        if EmbeddingLoader is None:
            logger.critical(f"{icon_error} Модуль EmbeddingLoader не найден (проблема с импортом _common).")
            return False

        try:
            loader = EmbeddingLoader(self.embeddings_dir)
            # Загружаем категорию 'faces' (единое хранилище)
            self.embeddings, self.index_map = loader.load("faces")
            
            if self.embeddings is None:
                logger.error(f"{icon_error} Эмбеддинги не найдены в {self.embeddings_dir}")
                return False
                
        except Exception as e:
            logger.error(f"{icon_error} Ошибка загрузки эмбеддингов: {e}")
            return False

        return True

    def save_json(self, backup: bool = True) -> bool:
        """
        Сохраняет текущее состояние self.json_data в файл.
        """
        try:
            if backup and self.json_path.exists():
                shutil.copy(self.json_path, self.json_path.with_suffix(".json.bak"))

            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self.json_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"<br>{icon_save} файл <i>{self.json_path.name}</i> сохранён")
            return True
        except Exception as e:
            logger.critical(f"{icon_save_error} Ошибка сохранения JSON: {e}")
            return False

    def get_subset_embeddings(self, filter_func) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Вспомогательный метод для получения подмножества эмбеддингов.
        
        Args:
            filter_func: Функция (filename, file_data) -> bool. 
                         Если True, эмбеддинги этого файла включаются в выборку.
                         
        Returns:
            (filenames, global_indices, embedding_matrix)
        """
        filenames = []
        global_indices = []
       
        if self.embeddings is None:
            return [], [], np.array([])

        for fname, info in self.json_data.items():
            if filter_func(fname, info):
                indices = self.index_map.get(fname, [])
                
                # В большинстве стратегий мы работаем с лицами.
                # Если фильтр пропустил файл, берем его индексы.
                # Важно: стратегии должны сами разбираться, какое именно лицо им нужно,
                # но этот хелпер собирает все лица из подходящих файлов.
                
                for idx in indices:
                    if idx < len(self.embeddings):
                        filenames.append(fname)
                        global_indices.append(idx)

        if not global_indices:
            return [], [], np.array([])

        return filenames, global_indices, self.embeddings[global_indices]