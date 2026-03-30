# analize/cluster_editor/_lib/strategies/cleaning.py

import logging
import json
import re
import numpy as np
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy

# Попытка импорта общих модулей (предполагается структура проекта)
try:
    from _common._shared import EmbeddingLoader
except ImportError:
    EmbeddingLoader = None

logger = logging.getLogger(__name__)

class CleaningModeStrategy(EditorStrategy):
    """
    Стратегия для режима 'cleaning'.
    Предназначена для разметки 'мусора' (trash) и временной группировки.
    Метод save() выполняет деструктивное удаление отсеянных данных.
    """

    @property
    def mode_name(self) -> str:
        return "cleaning"
    
    def show_face_details_panel(self) -> bool:
        return False

    def get_window_title(self, session_name: str) -> str:
        return f"ОЧИСТКА (Удаление мусора) - {session_name}"

    def get_name_prefix(self, cluster_id: str) -> str:
        if cluster_id.isdigit():
            return f"Temp {cluster_id} - "
        return ""

    def _strip_name_prefix(self, name: str) -> str:
        # Убираем "Temp X - "
        if name and name.startswith("Temp ") and " - " in name:
            return name.split(" - ", 1)[1]
        return name

    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        
        for record in records.values():
            for face in record.faces:
                if face.is_trash:
                    cid, cname = "trash", "🗑️ КОРЗИНА"
                else:
                    cid = str(face.temp_cluster_label if face.temp_cluster_label is not None else -1)
                    raw_name = face.temp_child_name
                    
                    # Логика дефолтного имени
                    if raw_name == f"Temp_Cluster_{cid}":
                        base_name = "Auto"
                    else:
                        base_name = raw_name or "Auto"
                    
                    if cid == "-1":
                        cname = "Одиночные (Noise)"
                    else:
                        prefix = self.get_name_prefix(cid)
                        # Избегаем дублирования префикса
                        if not base_name.startswith("Temp "):
                            cname = prefix + base_name
                        else:
                            cname = base_name

                # Временная подмена имени для UI
                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)

        return dict(clusters)

    def get_files_for_cluster(self, cluster_id: str, records: Dict[str, ImageRecord]) -> List[str]:
        result = set()
        for filename, record in records.items():
            for face in record.faces:
                if cluster_id == "trash":
                    if face.is_trash: result.add(filename)
                else:
                    current_cid = str(face.temp_cluster_label if face.temp_cluster_label is not None else -1)
                    if not face.is_trash and current_cid == cluster_id:
                        result.add(filename)
        
        # --- ИСПРАВЛЕНИЕ: Натуральная сортировка ---
        def natural_keys(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
            
        return sorted(list(result), key=natural_keys)

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                        records: Dict[str, ImageRecord], 
                        face_selection_map: Optional[Dict[str, Any]] = None,
                        target_name: Optional[str] = None) -> None:
        
        new_id_val = int(target_id) if target_id.lstrip('-').isdigit() else None
        clean_name = self._strip_name_prefix(target_name) if target_name else None

        for filename in filenames:
            record = records.get(filename)
            if not record: continue
            
            # --- ИСПРАВЛЕНИЕ: Обработка списка индексов из face_selection_map ---
            indices = range(len(record.faces))
            if face_selection_map and filename in face_selection_map:
                selection = face_selection_map[filename]
                # Если передали список индексов - используем его, иначе оборачиваем в список
                if isinstance(selection, list):
                    indices = selection
                else:
                    indices = [selection]
            
            for idx in indices:
                if idx >= len(record.faces): continue
                face = record.faces[idx]
                
                if target_id == "trash":
                    face.quality_status = "trash"
                    face.temp_cluster_label = None
                else:
                    face.quality_status = "ok"
                    face.temp_cluster_label = new_id_val
                    if clean_name:
                        face.temp_child_name = clean_name

    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        if cluster_id in ["trash", "-1"]: return
        
        files = self.get_files_for_cluster(cluster_id, records)
        clean_name = self._strip_name_prefix(new_name)
        
        for fname in files:
            record = records[fname]
            for face in record.faces:
                if str(face.temp_cluster_label) == cluster_id and not face.is_trash:
                    face.temp_child_name = clean_name

    def save(self, records: Dict[str, ImageRecord], paths_config: Dict[str, Path]) -> bool:
        """
        Специализированное сохранение: удаляет 'мусор' из JSON и пересобирает вектора.
        """
        logger.info("Cleaning Strategy: Starting destructive save...")
        
        json_path = paths_config.get("json_path")
        embeddings_dir = paths_config.get("embeddings_dir")
        vector_cache = paths_config.get("vector_cache", {})

        if not json_path or not embeddings_dir:
            logger.error("Missing paths for cleaning save")
            return False

        if not EmbeddingLoader:
            logger.error("EmbeddingLoader unavailable")
            return False

        new_json_data = {}
        new_vectors = []
        new_index_map = {}
        files_to_remove_from_memory = []

        for filename, record in list(records.items()):
            valid_faces = []
            file_indices = []
            
            for i, face in enumerate(record.faces):
                if face.is_trash: continue
                
                # --- ИСПРАВЛЕНИЕ: Надежный поиск вектора (Способ Б) ---
                vector = None
                
                # 1. Сначала пробуем ключ, который сформировал DataManager (он уже учитывает face_index)
                if face.embedding_key and face.embedding_key in vector_cache:
                    vector = vector_cache[face.embedding_key]
                else:
                    # 2. Fallback: строим ключи сами
                    # Если есть face_index, используем его (самое надежное)
                    if face.face_index is not None:
                        key_by_index = f"{filename}::{face.face_index}"
                        if key_by_index in vector_cache:
                            vector = vector_cache[key_by_index]
                    
                    # 3. Fallback: если одиночный файл
                    if vector is None and len(record.faces) == 1 and filename in vector_cache:
                        vector = vector_cache[filename]
                    
                    # 4. Fallback: используем loop index (только для старых файлов без индекса)
                    if vector is None:
                        key_loop = f"{filename}::{i}"
                        if key_loop in vector_cache:
                            vector = vector_cache[key_loop]

                if vector is not None:
                    new_idx = len(new_vectors)
                    new_vectors.append(vector)
                    file_indices.append(new_idx)
                    valid_faces.append(face)
                else:
                    # Логируем с face_index для отладки
                    idx_info = f"idx={face.face_index}" if face.face_index is not None else f"loop={i}"
                    logger.warning(f"Vector missing for valid face: {filename} ({idx_info})")
                    valid_faces.append(face)
            
            record.faces = valid_faces
            record.face_count = len(valid_faces)
            
            if valid_faces:
                new_json_data[filename] = record.to_dict()
                if file_indices:
                    new_index_map[filename] = file_indices
                record.commit_changes()
            else:
                files_to_remove_from_memory.append(filename)

        for fname in files_to_remove_from_memory:
            del records[fname]

        # 2. Сохранение векторов
        try:
            emb_loader = EmbeddingLoader(embeddings_dir)
            if new_vectors:
                arr = np.array(new_vectors, dtype=np.float32)
                emb_loader.save("faces", arr, new_index_map)
                
                # Обновляем кэш векторов для продолжения работы без перезапуска
                # Поскольку мы переписали файл, старые индексы (face_index) могут стать невалидными 
                # для НОВЫХ векторов в памяти, но мы перестраиваем кэш.
                vector_cache.clear()
                for fname, indices in new_index_map.items():
                    for i, idx in enumerate(indices):
                        if idx < len(arr):
                            # ВАЖНО: После очистки (Cleaning) вектора переупакованы.
                            # Теперь их ключи снова 0, 1, 2...
                            # Нам нужно обновить face_index у объектов в памяти, чтобы они соответствовали новому файлу!
                            vector_cache[f"{fname}::{i}"] = arr[idx]
                            if len(indices) == 1: vector_cache[fname] = arr[idx]
                            
                            # Обновляем face_index у объектов Face в памяти
                            if fname in records:
                                rec = records[fname]
                                if i < len(rec.faces):
                                    rec.faces[i].face_index = i # Новый индекс в новом файле
                                    rec.faces[i].embedding_key = f"{fname}::{i}"

        except Exception as e:
            logger.critical(f"Cleaning: NPY Save error: {e}")
            return False

        # 3. Сохранение JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(new_json_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.critical(f"Cleaning: JSON Save error: {e}")
            return False
            
        return True