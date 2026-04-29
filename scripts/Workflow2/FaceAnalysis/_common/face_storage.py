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
    """

    # ИСПРАВЛЕНИЕ: Добавлен аргумент clear_existing
    def __init__(self, output_dir: Path, clear_existing: bool = False):
        self.output_dir = output_dir
        self.embeddings_dir = output_dir / "_Embeddings"
        self.clear_existing = clear_existing
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._temp_dir = self.output_dir / "_temp_processing"
        self._temp_dir.mkdir(exist_ok=True)

        self._temp_faces_path = self._temp_dir / "temp_faces.jsonl"
        self._temp_land_path = self._temp_dir / "temp_landmarks.jsonl"
        self._temp_emb_bin = self._temp_dir / "temp_embeddings.bin"
        self._temp_idx_path = self._temp_dir / "temp_index.jsonl"

        # Инициализируем счетчик размером существующего .npy, чтобы индексы не пересекались при добавлении файлов
        self._total_embeddings_count = 0
        existing_npy = self.embeddings_dir / "faces_embeddings.npy"
        
        # ИСПРАВЛЕНИЕ: Читаем старые данные ТОЛЬКО если мы в режиме Sync (clear_existing=False)
        if not self.clear_existing and existing_npy.exists():
            try:
                old_emb = np.load(existing_npy, mmap_mode='r')
                self._total_embeddings_count = old_emb.shape[0]
            except Exception:
                pass
        
        self._cleanup_temp_files()
        logger.debug(f"ℹ️ FaceStorageManager инициализирован. Временная папка: {self._temp_dir}")

    def save_batch(self, batch_results: List[Tuple[str, List[Dict], List[np.ndarray], Tuple[int, int]]]):
        if not batch_results:
            return

        try:
            with open(self._temp_faces_path, "a", encoding="utf-8") as f_faces, \
                 open(self._temp_land_path, "a", encoding="utf-8") as f_land, \
                 open(self._temp_idx_path, "a", encoding="utf-8") as f_idx, \
                 open(self._temp_emb_bin, "ab") as f_emb:

                for filename, meta_list, emb_list, orig_shape in batch_results:
                    main_faces_data = list()
                    land_faces_data = list()
                    has_landmarks = False

                    for face_meta in meta_list:
                        main_face, land_face = self._split_face_data(face_meta)
                        main_faces_data.append(main_face)
                        land_faces_data.append(land_face)
                        if land_face:
                            has_landmarks = True

                    record_main = {
                        "filename": filename,
                        "face_count": len(meta_list),
                        "original_shape": orig_shape,
                        "faces": main_faces_data
                    }
                    f_faces.write(json.dumps(record_main, ensure_ascii=False) + "\n")

                    if has_landmarks:
                        record_land = {
                            "filename": filename,
                            "faces": land_faces_data
                        }
                        f_land.write(json.dumps(record_land, ensure_ascii=False) + "\n")

                    current_indices = list()
                    for emb in emb_list:
                        f_emb.write(emb.astype(np.float32).tobytes())
                        current_indices.append(self._total_embeddings_count)
                        self._total_embeddings_count += 1
                    
                    if current_indices:
                        record_idx = {filename: current_indices}
                        f_idx.write(json.dumps(record_idx, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении батча данных: {e}", exc_info=True)
            raise

    def finalize(self) -> bool:
        logger.info("<b>Сборка итоговых результатов из временных файлов...</b>")
        
        try:
            # 1. Сборка основного JSON (info_faces.json)
            final_faces = dict()
            target_json = self.output_dir / "info_faces.json"
            
            # ИСПРАВЛЕНИЕ: Грузим старые данные только если это не режим полной перезаписи
            if not self.clear_existing and target_json.exists():
                try:
                    with open(target_json, "r", encoding="utf-8") as f:
                        final_faces = json.load(f)
                except Exception:
                    pass

            if self._temp_faces_path.exists():
                with open(self._temp_faces_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            final_faces[record["filename"]] = record
            
            if final_faces:
                self._save_json(target_json, final_faces)
                logger.info(f"{icon_save} файл <i>info_faces.json</i> сохранён (всего записей: <b>{len(final_faces)}</b>)")

            # 2. Сборка JSON с ландмарками
            final_landmarks = dict()
            target_land_json = self.output_dir / "info_faces_landmarks.json"
            
            if not self.clear_existing and target_land_json.exists():
                try:
                    with open(target_land_json, "r", encoding="utf-8") as f:
                        final_landmarks = json.load(f)
                except Exception:
                    pass

            if self._temp_land_path.exists():
                with open(self._temp_land_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            final_landmarks[record["filename"]] = record
            
            if final_landmarks:
                self._save_json(target_land_json, final_landmarks)
                logger.info(f"{icon_save} файл <i>info_faces_landmarks.json</i> сохранён")

            # 3. Сборка индекса эмбеддингов
            final_index = dict()
            target_idx_json = self.embeddings_dir / "faces_index.json"
            
            if not self.clear_existing and target_idx_json.exists():
                try:
                    with open(target_idx_json, "r", encoding="utf-8") as f:
                        final_index = json.load(f)
                except Exception:
                    pass

            if self._temp_idx_path.exists():
                with open(self._temp_idx_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            final_index.update(record)
            
            if final_index:
                self._save_json(target_idx_json, final_index)

            # 4. Конвертация бинарных эмбеддингов в .npy
            if self._temp_emb_bin.exists():
                raw_data = np.fromfile(self._temp_emb_bin, dtype=np.float32)
                new_count = raw_data.size // 512
                if raw_data.size > 0 and raw_data.size % 512 == 0:
                    new_embeddings = raw_data.reshape((new_count, 512))
                    target_npy = self.embeddings_dir / "faces_embeddings.npy"
                    
                    # ИСПРАВЛЕНИЕ: Склеиваем с массивом только в режиме Sync
                    if not self.clear_existing and target_npy.exists():
                        try:
                            old_embeddings = np.load(target_npy)
                            combined_embeddings = np.vstack((old_embeddings, new_embeddings))
                        except Exception:
                            combined_embeddings = new_embeddings
                    else:
                        # В режиме Create просто записываем новую матрицу
                        combined_embeddings = new_embeddings
                        
                    np.save(target_npy, combined_embeddings)
                    logger.info(f"{icon_save} файл <i>faces_embeddings.npy</i> обновлен (всего <b>{self._total_embeddings_count}</b> лиц)")

            # 5. Очистка
            self._cleanup_temp_files(remove_dir=True)
            return True

        except Exception as e:
            logger.critical(f"{icon_error} Критическая ошибка при финализации данных: {e}", exc_info=True)
            return False

    def _split_face_data(self, face_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        main_face = face_meta.copy()
        land_face = dict()
        keys_to_move = list(["landmark_2d_106", "landmark_3d_68"])
        has_extracted = False
        for key in keys_to_move:
            if key in main_face:
                land_face[key] = main_face.pop(key)
                has_extracted = True
        return main_face, (land_face if has_extracted else dict())

    def _save_json(self, path: Path, data: Any):
        try:
            # ИСПРАВЛЕНИЕ: indent=2 обеспечивает компактный размер
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"{icon_error} Ошибка записи JSON {path}: {e}")

    def _cleanup_temp_files(self, remove_dir: bool = False):
        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
                if not remove_dir:
                    self._temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"{icon_warning}️ Не удалось очистить временные файлы: {e}")