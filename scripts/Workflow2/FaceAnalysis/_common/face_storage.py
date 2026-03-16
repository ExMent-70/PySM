# analize/_common/face_storage.py

import json
import logging
import shutil
import struct
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from .status_icons import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

logger = logging.getLogger(__name__)

class FaceStorageManager:
    """
    Управляет инкрементальным сохранением результатов анализа лиц.
    
    Стратегия:
    1. В процессе работы данные сбрасываются (append) во временные файлы:
       - temp_faces.jsonl (основные метаданные)
       - temp_landmarks.jsonl (тяжелые ландмарки 106/68 точек)
       - temp_embeddings.bin (плоский бинарный массив float32)
       - temp_index.jsonl (связь имени файла и индексов эмбеддингов)
    
    2. Метод finalize() собирает эти данные в итоговые файлы:
       - info_faces.json
       - info_faces_landmarks.json
       - _Embeddings/faces_embeddings.npy
       - _Embeddings/faces_index.json
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.embeddings_dir = output_dir / "_Embeddings"
        
        # Создаем необходимые директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Пути к временным файлам
        self._temp_dir = self.output_dir / "_temp_processing"
        self._temp_dir.mkdir(exist_ok=True)

        self._temp_faces_path = self._temp_dir / "temp_faces.jsonl"
        self._temp_land_path = self._temp_dir / "temp_landmarks.jsonl"
        self._temp_emb_bin = self._temp_dir / "temp_embeddings.bin"
        self._temp_idx_path = self._temp_dir / "temp_index.jsonl"

        # Счетчик записанных эмбеддингов (для формирования глобального индекса)
        self._total_embeddings_count = 0
        
        # Очистка предыдущих временных файлов при новом запуске
        self._cleanup_temp_files()
        logger.debug(f"ℹ️ FaceStorageManager инициализирован. Временная папка: {self._temp_dir}")

    def save_batch(self, batch_results: List[Tuple[str, List[Dict], List[np.ndarray], Tuple[int, int]]]):
        """
        Сохраняет пакет обработанных данных во временные файлы.

        Args:
            batch_results: Список кортежей, где каждый кортеж содержит:
                - filename (str): Имя файла.
                - meta_list (List[Dict]): Список словарей с метаданными лиц.
                - embeddings_list (List[np.ndarray]): Список векторов (512,).
                - original_shape (Tuple[int, int]): Размеры исходного изображения.
        """
        if not batch_results:
            return

        try:
            # Открываем файлы в режиме добавления (append)
            with open(self._temp_faces_path, "a", encoding="utf-8") as f_faces, \
                 open(self._temp_land_path, "a", encoding="utf-8") as f_land, \
                 open(self._temp_idx_path, "a", encoding="utf-8") as f_idx, \
                 open(self._temp_emb_bin, "ab") as f_emb:

                for filename, meta_list, emb_list, orig_shape in batch_results:
                    
                    # 1. Подготовка метаданных
                    main_faces_data = []
                    land_faces_data = []
                    has_landmarks = False

                    # Разделяем "легкие" данные и "тяжелые" ландмарки
                    for face_meta in meta_list:
                        main_face, land_face = self._split_face_data(face_meta)
                        main_faces_data.append(main_face)
                        land_faces_data.append(land_face)
                        if land_face:
                            has_landmarks = True

                    # 2. Формирование записей для JSON Lines
                    # Основная запись
                    record_main = {
                        "filename": filename,
                        "face_count": len(meta_list),
                        "original_shape": orig_shape,
                        "faces": main_faces_data
                    }
                    f_faces.write(json.dumps(record_main, ensure_ascii=False) + "\n")

                    # Запись ландмарков (только если они есть)
                    if has_landmarks:
                        record_land = {
                            "filename": filename,
                            "faces": land_faces_data
                        }
                        f_land.write(json.dumps(record_land, ensure_ascii=False) + "\n")

                    # 3. Сохранение эмбеддингов и индекса
                    current_indices = []
                    for emb in emb_list:
                        # Записываем сырые байты float32
                        f_emb.write(emb.astype(np.float32).tobytes())
                        current_indices.append(self._total_embeddings_count)
                        self._total_embeddings_count += 1
                    
                    # Запись индекса: filename -> [idx1, idx2, ...]
                    if current_indices:
                        record_idx = {filename: current_indices}
                        f_idx.write(json.dumps(record_idx, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении батча данных: {e}", exc_info=True)
            raise

    def finalize(self) -> bool:
        """
        Собирает итоговые файлы из временных и очищает мусор.
        Вызывается в конце работы скрипта.
        """
        logger.info("<b>Сборка итоговых результатов из временных файлов...</b>")
        
        try:
            # 1. Сборка основного JSON (info_faces.json)
            final_faces = {}
            if self._temp_faces_path.exists():
                with open(self._temp_faces_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            final_faces[record["filename"]] = record
            
            self._save_json(self.output_dir / "info_faces.json", final_faces)
            logger.info(f"{icon_save} файл <i>info_faces.json</i> сохранён (обработано <b>{len(final_faces)}</b> изображений)")

            # 2. Сборка JSON с ландмарками (info_faces_landmarks.json)
            final_landmarks = {}
            if self._temp_land_path.exists():
                with open(self._temp_land_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            final_landmarks[record["filename"]] = record
            
            if final_landmarks:
                self._save_json(self.output_dir / "info_faces_landmarks.json", final_landmarks)
                logger.info(f"{icon_save} файл <i>info_faces_landmarks.json</i> сохранён")

            # 3. Сборка индекса эмбеддингов
            final_index = {}
            if self._temp_idx_path.exists():
                with open(self._temp_idx_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            final_index.update(record)
            
            if final_index:
                self._save_json(self.embeddings_dir / "faces_index.json", final_index)

            # 4. Конвертация бинарных эмбеддингов в .npy
            if self._temp_emb_bin.exists() and self._total_embeddings_count > 0:
                # Читаем весь бинарный файл как плоский массив
                raw_data = np.fromfile(self._temp_emb_bin, dtype=np.float32)
                # Решейпим в (N, 512)
                if raw_data.size != self._total_embeddings_count * 512:
                    logger.error(f"{icon_error} Несовпадение размеров данных эмбеддингов! Ожидалось <b>{self._total_embeddings_count * 512}</b>, получено <b>{raw_data.size}</b>")
                else:
                    embeddings_array = raw_data.reshape((self._total_embeddings_count, 512))
                    np.save(self.embeddings_dir / "faces_embeddings.npy", embeddings_array)
                    logger.info(f"{icon_save} файл <i>faces_embeddings.npy</i> сохранён (всего <b>{self._total_embeddings_count}</b> лиц)")

            # 5. Очистка
            self._cleanup_temp_files(remove_dir=True)
            return True

        except Exception as e:
            logger.critical(f"{icon_error} Критическая ошибка при финализации данных: {e}", exc_info=True)
            return False

    def _split_face_data(self, face_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Разделяет метаданные одного лица на основные и ландмарки.
        """
        main_face = face_meta.copy()
        land_face = {}
        
        # Ключи, которые уходят в файл ландмарков
        keys_to_move = ["landmark_2d_106", "landmark_3d_68"]
        
        has_extracted = False
        for key in keys_to_move:
            if key in main_face:
                land_face[key] = main_face.pop(key)
                has_extracted = True
        
        # Если ландмарков не было, возвращаем пустой словарь для land_face
        return main_face, (land_face if has_extracted else {})

    def _save_json(self, path: Path, data: Any):
        """Сохраняет данные в JSON с отступами."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"{icon_error} Ошибка записи JSON {path}: {e}")

    def _cleanup_temp_files(self, remove_dir: bool = False):
        """Удаляет временные файлы и папку."""
        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
                if not remove_dir:
                    self._temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"{icon_warning}️ Не удалось очистить временные файлы: {e}")