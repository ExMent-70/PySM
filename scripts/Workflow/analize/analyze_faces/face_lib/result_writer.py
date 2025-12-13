# analize/analyze_faces/face_lib/result_writer.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisResultWriter:
    """
    Класс для накопления результатов анализа и их надежного сохранения.
    Решает проблему с кодировкой (UTF-8) и централизует логику вывода.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.embeddings_dir = output_dir / "_Embeddings"
        
        self.portrait_meta: Dict[str, Any] = {}
        self.group_meta: Dict[str, Any] = {}
        
        self.portrait_embeddings: List[np.ndarray] = []
        self.group_embeddings: List[np.ndarray] = []
        
        self.portrait_index: Dict[str, int] = {}
        self.group_index: Dict[str, int] = {}

    def add_result(self, filename: str, meta: List[Dict], embeddings: List[np.ndarray], original_shape: tuple):
        """Добавляет результат анализа одного файла в общий пул."""
        file_data = {
            "filename": filename,
            "faces": meta,
            "original_shape": original_shape
        }
        
        is_portrait = len(meta) == 1
        
        if is_portrait:
            self.portrait_meta[filename] = file_data
            self.portrait_index[filename] = len(self.portrait_embeddings)
            self.portrait_embeddings.append(embeddings[0])
        else:
            self.group_meta[filename] = file_data
            for i, embedding in enumerate(embeddings):
                # Формируем уникальный ключ для лица в группе: filename::face_index
                key = f"{filename}::{i}"
                self.group_index[key] = len(self.group_embeddings)
                self.group_embeddings.append(embedding)

    def save_all(self, json_manager: Any):
        """
        Сохраняет все накопленные данные на диск.
        
        Args:
            json_manager: Экземпляр JsonDataManager для сохранения основных JSON (совместимость со старым кодом).
        """
        logger.info(f"Сохранение результатов: {len(self.portrait_meta)} портретных, {len(self.group_meta)} групповых.")
        
        # 1. Сохранение основных метаданных через существующий менеджер (он внутри просто дампит json)
        # ВАЖНО: JsonDataManager нужно будет проверить отдельно, но пока используем интерфейс как было.
        json_manager.portrait_data = self.portrait_meta
        json_manager.group_data = self.group_meta
        json_manager.save_data()

        # 2. Сохранение эмбеддингов и индексов
        if self.portrait_embeddings or self.group_embeddings:
            self.embeddings_dir.mkdir(exist_ok=True, parents=True)

        if self.portrait_embeddings:
            np.save(self.embeddings_dir / "portrait_embeddings.npy", np.array(self.portrait_embeddings))
            self._save_json_safe(self.embeddings_dir / "portrait_index.json", self.portrait_index)

        if self.group_embeddings:
            np.save(self.embeddings_dir / "group_embeddings.npy", np.array(self.group_embeddings))
            self._save_json_safe(self.embeddings_dir / "group_index.json", self.group_index)

    def _save_json_safe(self, path: Path, data: Any):
        """Вспомогательный метод для сохранения JSON с корректной кодировкой UTF-8."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Не удалось сохранить JSON файл {path}: {e}")