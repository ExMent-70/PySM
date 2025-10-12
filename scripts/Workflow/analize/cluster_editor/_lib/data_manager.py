# 1. БЛОК: data_manager.py (ПОЛНЫЙ ОБНОВЛЕННЫЙ КОД)
# ==============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_manager.py
===============
Модуль, содержащий класс ClusterDataManager для управления бизнес-логикой
и состоянием данных редактора кластеров.
"""

import logging
import ijson
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .data_models import ImageRecord, Face

logger = logging.getLogger(__name__)

class ClusterDataManager:
    """
    Класс, инкапсулирующий всю логику работы с данными о кластерах.
    Отвечает за загрузку, сохранение, модификацию и предоставление данных.
    """
    def __init__(self, portrait_json_path: Path, group_json_path: Path):
        self.portrait_json_path = portrait_json_path
        self.group_json_path = group_json_path
        
        self.records: Dict[str, ImageRecord] = {}
        self.newly_created_clusters: List[Dict] = []
        self._has_changes = False
        self._cluster_indices: Dict[str, Dict[str, List[str]]] = {
            'face': defaultdict(list),
            'location': defaultdict(list)
        }
        # --- ИЗМЕНЕНИЕ: Индекс теперь хранит кортежи (filename, distance) ---
        self.matches_index: Dict[str, List[tuple[str, float]]] = defaultdict(list)
        self._cluster_id_to_name_cache: Dict[str, str] = {}

    def _build_indices(self):
        """Строит индексы для быстрого поиска файлов по ID кластера."""
        self._cluster_indices['face'].clear()
        self._cluster_indices['location'].clear()
        self._cluster_id_to_name_cache.clear()

        for record in self.records.values():
            if record.image_type == 'group':
                face_cluster_id = "group"
            elif record.faces:
                face = record.faces[0]
                face_cluster_id = str(face.cluster_label if face.cluster_label is not None else -1)
                if face_cluster_id not in self._cluster_id_to_name_cache and face.child_name:
                    self._cluster_id_to_name_cache[face_cluster_id] = face.child_name
            else:
                face_cluster_id = "-1"
            self._cluster_indices['face'][face_cluster_id].append(record.filename)

            location_cluster_id = str(record.location_cluster if record.location_cluster is not None else -1)
            self._cluster_indices['location'][location_cluster_id].append(record.filename)
    
    def _build_matches_index(self):
        """
        Строит индекс совпадений, сохраняя имя файла и дистанцию.
        Структура: {'portrait_cluster_id': [('group_photo1.jpg', 0.1), ...]}
        """
        self.matches_index.clear()
        temp_matches = defaultdict(list)

        for record in self.records.values():
            if record.image_type != 'group':
                continue

            for face in record.faces:
                label = face.extra_data.get('matched_portrait_cluster_label')
                # --- ИЗМЕНЕНИЕ: Извлекаем match_distance ---
                distance = face.extra_data.get('match_distance')
                if label is not None and distance is not None:
                    temp_matches[str(label)].append((record.filename, float(distance)))
        
        # Сортируем списки по дистанции
        self.matches_index = {
            cluster_id: sorted(file_dist_pairs, key=lambda x: x[1])
            for cluster_id, file_dist_pairs in temp_matches.items()
        }

    def has_changes(self) -> bool:
        if self._has_changes:
            return True
        return any(record.is_changed for record in self.records.values())

    def load_data(self) -> tuple[bool, str]:
        if not self.portrait_json_path.is_file() or not self.group_json_path.is_file():
            msg = "Один или оба JSON-файла ('info_...') не найдены."
            return False, msg
        
        self.records.clear()

        def _parse_stream(file_path: Path, image_type: str):
            with open(file_path, 'r', encoding='utf-8') as f:
                for filename, image_data_gen in ijson.kvitems(f, ''):
                    image_data = dict(image_data_gen)
                    raw_faces = image_data.get("faces", [])
                    
                    parsed_faces = []
                    for face_data in raw_faces:
                        known_fields = {'bbox', 'cluster_label', 'child_name'}
                        light_data = {k: face_data[k] for k in known_fields if k in face_data}
                        extra_data = {k: v for k, v in face_data.items() if k not in known_fields}
                        
                        face_obj = Face(**light_data)
                        face_obj.extra_data = extra_data
                        parsed_faces.append(face_obj)

                    self.records[filename] = ImageRecord(
                        filename=filename, image_type=image_type, faces=parsed_faces,
                        raw_faces_data=raw_faces, location_cluster=image_data.get("location_cluster"),
                        location_name=image_data.get("location_name"),
                        original_shape=tuple(image_data.get("original_shape", [0, 0]))
                    )
        try:
            _parse_stream(self.portrait_json_path, 'portrait')
            _parse_stream(self.group_json_path, 'group')
        except Exception as e:
            error_message = f"Ошибка при потоковом чтении данных:\n\n{e}\n\n{traceback.format_exc()}"
            return False, error_message

        self._build_indices()
        self._build_matches_index()
        logger.info(f"Загружено {len(self.records)} записей об изображениях. Построены индексы.")
        self._has_changes = False
        return True, ""


    def get_all_location_names(self) -> List[str]:
        """Собирает все уникальные имена локаций из загруженных данных.

        Итерирует по всем записям в памяти, собирает непустые значения
        из поля `location_name` в множество для обеспечения уникальности,
        а затем возвращает отсортированный список.

        Returns:
            List[str]: Отсортированный список уникальных имен локаций.
        """
        location_names = set()
        for record in self.records.values():
            if record.location_name:
                location_names.add(record.location_name)
        return sorted(list(location_names))

    def save_data(self) -> bool:
        try:
            with open(self.portrait_json_path, 'r', encoding='utf-8') as f:
                full_portrait_data = json.load(f)
            with open(self.group_json_path, 'r', encoding='utf-8') as f:
                full_group_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Не удалось прочитать исходные JSON для слияния при сохранении: {e}")
            return False

        for record in self.records.values():
            if not record.is_changed:
                continue

            target_data_dict = None
            if record.filename in full_portrait_data:
                target_data_dict = full_portrait_data
            elif record.filename in full_group_data:
                target_data_dict = full_group_data
            
            if target_data_dict:
                target_data_dict[record.filename]['location_cluster'] = record.location_cluster
                target_data_dict[record.filename]['location_name'] = record.location_name
                
                for i, light_face in enumerate(record.faces):
                    if i < len(target_data_dict[record.filename].get('faces', [])):
                        target_data_dict[record.filename]['faces'][i]['cluster_label'] = light_face.cluster_label
                        target_data_dict[record.filename]['faces'][i]['child_name'] = light_face.child_name

        try:
            with open(self.portrait_json_path, 'w', encoding='utf-8') as f:
                json.dump(full_portrait_data, f, ensure_ascii=False, indent=2)
            
            with open(self.group_json_path, 'w', encoding='utf-8') as f:
                json.dump(full_group_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.critical(f"Критическая ошибка при записи JSON: {e}")
            return False

        for record in self.records.values():
            record.is_changed = False
        self._has_changes = False
        self.newly_created_clusters = []
        logger.info("Изменения успешно сохранены.")
        return True

    def generate_and_save_matches_json(self, output_path: Path) -> tuple[bool, str]:
        output_data = {}
        
        try:
            sorted_cluster_ids = sorted(self.matches_index.keys(), key=int)
        except ValueError:
            sorted_cluster_ids = sorted(self.matches_index.keys())

        for cluster_id in sorted_cluster_ids:
            # matches_index теперь содержит список кортежей (filename, distance)
            matches = self.matches_index[cluster_id]
            if not matches:
                continue
            
            child_name = self._cluster_id_to_name_cache.get(cluster_id)
            if not child_name:
                portrait_files = self._cluster_indices['face'].get(cluster_id, [])
                if portrait_files:
                    first_record = self.records.get(portrait_files[0])
                    if first_record and first_record.faces:
                        child_name = first_record.faces[0].child_name
            if not child_name:
                child_name = f"Кластер {cluster_id}"
            
            group_photos_list = []
            for filename, distance in matches:
                group_photos_list.append({
                    "filename": filename,
                    "min_distance": distance, # Используем реальную дистанцию
                    "num_faces": 1 
                })

            output_data[cluster_id] = {
                "child_name": child_name.split('-', 1)[-1] if child_name.startswith("0") else child_name,
                "group_photos": group_photos_list
            }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            msg = f"Файл совпадений успешно сгенерирован и сохранен:\n{output_path}"
            logger.info(msg)
            return True, msg
        except (IOError, TypeError) as e:
            error_msg = f"Ошибка при сохранении файла совпадений: {e}"
            return False, error_msg

    def get_clusters(self, mode_config: Dict) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = {}
        is_face_mode = mode_config["mode_name"] == 'face'
        
        index_to_use = self._cluster_indices['face' if is_face_mode else 'location']
        
        for cluster_id, filenames in index_to_use.items():
            if not filenames: continue
            
            first_record = self.records[filenames[0]]
            face = first_record.faces[0] if first_record.faces else Face(bbox=[])
            
            cluster_name = ""
            if is_face_mode:
                if first_record.image_type == 'group': cluster_name = "_Group_Photos"
                else:
                    cluster_name = face.child_name or f"Кластер {cluster_id}"
                    if cluster_id == "-1": cluster_name = "99-Noise"
                    elif cluster_name.startswith("Unknown"):
                        if not cluster_name.startswith("98-"): cluster_name = f"98-{cluster_name}"
                    elif cluster_id not in ["-1", "group"]:
                        prefix = mode_config['name_prefix_logic'](cluster_id)
                        current_name_no_prefix = cluster_name.split('-', 1)[-1]
                        cluster_name = prefix + current_name_no_prefix
            else:
                cluster_name = first_record.location_name or f"Локация {cluster_id}"
            
            face.filename = first_record.filename
            face.effective_name = cluster_name
            
            clusters[cluster_id] = [face] * len(filenames)

        return clusters

    def get_files_for_cluster(self, mode_config: Dict, cluster_id: str) -> List[str]:
        index_key = 'face' if mode_config["mode_name"] == 'face' else 'location'
        return sorted(self._cluster_indices[index_key].get(cluster_id, []))

    def get_group_matches_for_cluster(self, cluster_id: str) -> List[str]:
        # --- ИЗМЕНЕНИЕ: Возвращаем только имена файлов, так как дистанция не нужна для отображения ---
        return [filename for filename, distance in self.matches_index.get(cluster_id, [])]

    def move_images_to_cluster(self, mode_config: Dict, target_id: str, target_name: str, filenames: List[str]):
        is_face_mode = mode_config["mode_name"] == 'face'
        new_id_val = int(target_id) if target_id.isdigit() else target_id
        
        index_key = 'face' if is_face_mode else 'location'
        index = self._cluster_indices[index_key]

        for filename in filenames:
            record = self.records.get(filename)
            if not record: continue

            record.is_changed = True
            
            old_id = ""
            if is_face_mode:
                old_id = "group" if record.image_type == 'group' else str(record.faces[0].cluster_label or -1)
            else:
                old_id = str(record.location_cluster or -1)
            if filename in index[old_id]:
                index[old_id].remove(filename)

            if is_face_mode:
                if target_id == "group":
                    if record.image_type == 'portrait':
                        record.image_type = 'group'
                        if record.faces: record.faces[0].cluster_label = None; record.faces[0].child_name = None
                else:
                    if record.image_type == 'group': record.image_type = 'portrait'
                    if record.faces: record.faces[0].cluster_label = new_id_val; record.faces[0].child_name = target_name
            else:
                record.location_cluster = new_id_val
                record.location_name = target_name
            
            index[target_id].append(filename)
        
        self.newly_created_clusters = [c for c in self.newly_created_clusters if c['id'] != target_id]
        self._has_changes = True

    def rename_cluster(self, mode_config: Dict, cluster_id: str, new_name: str):
        is_face_mode = mode_config["mode_name"] == 'face'
        prefix = mode_config["name_prefix_logic"](cluster_id)
        final_new_name = prefix + new_name

        files_to_rename = self.get_files_for_cluster(mode_config, cluster_id)
        for filename in files_to_rename:
            record = self.records[filename]
            record.is_changed = True
            if is_face_mode:
                if record.faces: record.faces[0].child_name = final_new_name
            else:
                record.location_name = final_new_name
        self._has_changes = True
    
    def create_cluster(self, mode_config: Dict, new_name: str):
        is_face_mode = mode_config["mode_name"] == 'face'
        index_key = 'face' if is_face_mode else 'location'
        
        existing_ids = set(self._cluster_indices[index_key].keys())
        for cluster_data in self.newly_created_clusters:
            existing_ids.add(cluster_data["id"])

        numeric_ids = {0}
        for cid in existing_ids:
            if cid.isdigit(): numeric_ids.add(int(cid))

        new_id = max(numeric_ids) + 1
        new_id_str = str(new_id)

        prefix = mode_config['name_prefix_logic'](new_id_str)
        final_new_name = prefix + new_name.strip()

        self.newly_created_clusters.append({"id": new_id_str, "name": final_new_name})
        self._has_changes = True

    def delete_newly_created_cluster(self, cluster_id: str):
        self.newly_created_clusters = [c for c in self.newly_created_clusters if c['id'] != cluster_id]
        self._has_changes = True