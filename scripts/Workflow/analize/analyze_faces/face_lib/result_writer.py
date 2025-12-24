# analize/analyze_faces/face_lib/result_writer.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisResultWriter:
    """
    Класс для накопления результатов анализа и их надежного сохранения.
    Решает проблему с кодировкой (UTF-8) и разделяет "тяжелые" ландмарки
    от основных метаданных.
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
        Разделяет основные данные и тяжелые ландмарки по разным файлам.
        
        Args:
            json_manager: Экземпляр JsonDataManager для сохранения основных JSON.
        """
        logger.info(f"Сохранение результатов: {len(self.portrait_meta)} портретных, {len(self.group_meta)} групповых.")
        
        # 1. Разделение данных на "легкие" (основные) и "тяжелые" (ландмарки)
        p_main, p_land = self._split_landmarks(self.portrait_meta)
        g_main, g_land = self._split_landmarks(self.group_meta)

        # 2. Сохранение основных метаданных через существующий менеджер
        json_manager.portrait_data = p_main
        json_manager.group_data = g_main
        json_manager.save_data()

        # 3. Сохранение отдельных файлов с ландмарками
        if p_land:
            self._save_json_safe(self.output_dir / "info_portrait_landmarks.json", p_land)
            logger.info(f"- ландмарки портретов: <i>info_portrait_landmarks.json</i>")
            
        if g_land:
            self._save_json_safe(self.output_dir / "info_group_landmarks.json", g_land)
            logger.info(f"- ландмарки групп: <i>info_group_landmarks.json</i>")

        # 4. Сохранение эмбеддингов и индексов
        if self.portrait_embeddings or self.group_embeddings:
            self.embeddings_dir.mkdir(exist_ok=True, parents=True)

        if self.portrait_embeddings:
            np.save(self.embeddings_dir / "portrait_embeddings.npy", np.array(self.portrait_embeddings))
            self._save_json_safe(self.embeddings_dir / "portrait_index.json", self.portrait_index)

        if self.group_embeddings:
            np.save(self.embeddings_dir / "group_embeddings.npy", np.array(self.group_embeddings))
            self._save_json_safe(self.embeddings_dir / "group_index.json", self.group_index)

    def _split_landmarks(self, meta_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Разделяет исходные метаданные на два словаря:
        1. Main: Все данные БЕЗ landmark_2d_106 и landmark_3d_68.
        2. Landmarks: Только landmark_2d_106 и landmark_3d_68 (с сохранением структуры имен файлов).
        """
        main_data = {}
        landmarks_data = {}
        
        # Ключи, которые нужно вынести в отдельный файл
        keys_to_move = ["landmark_2d_106", "landmark_3d_68"]

        for filename, content in meta_data.items():
            # Копируем структуру файла
            main_entry = content.copy()
            main_entry["faces"] = []
            
            land_entry = {
                "filename": filename,
                "faces": []
            }
            
            has_landmarks = False
            
            for face in content["faces"]:
                # Создаем копии словарей лица
                main_face = face.copy()
                land_face = {}
                
                for key in keys_to_move:
                    if key in main_face:
                        land_face[key] = main_face.pop(key)
                        has_landmarks = True
                
                main_entry["faces"].append(main_face)
                land_entry["faces"].append(land_face)
            
            main_data[filename] = main_entry
            if has_landmarks:
                landmarks_data[filename] = land_entry
                
        return main_data, landmarks_data

    def _save_json_safe(self, path: Path, data: Any):
        """Вспомогательный метод для сохранения JSON с корректной кодировкой UTF-8."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Не удалось сохранить JSON файл {path}: {e}")