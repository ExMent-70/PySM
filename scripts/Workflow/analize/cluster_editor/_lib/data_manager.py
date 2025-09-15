# 1. БЛОК: Файл _lib/data_manager.py (ПОЛНАЯ ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)
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
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict, fields
from collections import defaultdict

from .data_models import ImageRecord, Face
from _common.json_data_manager import JsonDataManager

logger = logging.getLogger(__name__)

class ClusterDataManager:
    """
    Класс, инкапсулирующий всю логику работы с данными о кластерах.
    Отвечает за загрузку, сохранение, модификацию и предоставление данных.
    """
    def __init__(self, portrait_json_path: Path, group_json_path: Path):
        self._json_manager = JsonDataManager(portrait_json_path, group_json_path)
        self.records: Dict[str, ImageRecord] = {}
        self.newly_created_clusters: List[Dict] = []
        self._has_changes = False
        # --- НОВОЕ: Индексы для быстрого доступа ---
        self._cluster_indices: Dict[str, Dict[str, List[str]]] = {
            'face': defaultdict(list),
            'location': defaultdict(list)
        }

    def _build_indices(self):
        """Строит индексы для быстрого поиска файлов по ID кластера."""
        self._cluster_indices['face'].clear()
        self._cluster_indices['location'].clear()

        for record in self.records.values():
            # Индекс для режима "лиц"
            if record.image_type == 'group':
                face_cluster_id = "group"
            elif record.faces:
                face_cluster_id = str(record.faces[0].cluster_label if record.faces[0].cluster_label is not None else -1)
            else:
                face_cluster_id = "-1"
            self._cluster_indices['face'][face_cluster_id].append(record.filename)

            # Индекс для режима "локаций"
            location_cluster_id = str(record.location_cluster if record.location_cluster is not None else -1)
            self._cluster_indices['location'][location_cluster_id].append(record.filename)

    def has_changes(self) -> bool:
        if self._has_changes:
            return True
        return any(record.is_changed for record in self.records.values())

    def load_data(self) -> bool:
        if not self._json_manager.load_data():
            logger.error("Не удалось загрузить JSON-файлы.")
            return False

        self.records.clear()
        known_face_fields = {f.name for f in fields(Face) if f.name not in ["filename", "effective_name"]}

        def parse_image_data(filename: str, data: Dict, image_type: str):
            parsed_faces = []
            for face_data in data.get("faces", []):
                known_data = {k: v for k, v in face_data.items() if k in known_face_fields}
                extra_data = {k: v for k, v in face_data.items() if k not in known_face_fields}
                known_data['extra_data'] = extra_data
                parsed_faces.append(Face(**known_data))

            self.records[filename] = ImageRecord(
                filename=filename,
                image_type=image_type,
                faces=parsed_faces,
                location_cluster=data.get("location_cluster"),
                location_name=data.get("location_name"),
                original_shape=tuple(data.get("original_shape", [0, 0]))
            )

        for filename, data in self._json_manager.portrait_data.items():
            parse_image_data(filename, data, 'portrait')
        for filename, data in self._json_manager.group_data.items():
            parse_image_data(filename, data, 'group')
            
        # --- НОВОЕ: Строим индексы после загрузки ---
        self._build_indices()
        
        logger.info(f"Загружено {len(self.records)} записей об изображениях.")
        self._has_changes = False
        return True

    def save_data(self) -> bool:
        portrait_data: Dict[str, Any] = {}
        group_data: Dict[str, Any] = {}

        for record in self.records.values():
            record_dict = {
                "faces": [],
                "original_shape": record.original_shape,
                "location_cluster": record.location_cluster,
                "location_name": record.location_name
            }
            for face in record.faces:
                face_dict = asdict(face, dict_factory=lambda x: {k: v for (k, v) in x if v is not None and k not in ['filename', 'effective_name', 'extra_data']})
                if face.extra_data:
                    face_dict.update(face.extra_data)
                record_dict['faces'].append(face_dict)

            if record.image_type == 'portrait':
                portrait_data[record.filename] = record_dict
            else:
                group_data[record.filename] = record_dict

        self._json_manager.portrait_data = portrait_data
        self._json_manager.group_data = group_data
        
        if not self._json_manager.save_data():
            logger.error("Не удалось сохранить JSON-файлы.")
            return False

        for record in self.records.values():
            record.is_changed = False
        self._has_changes = False
        self.newly_created_clusters = []
        logger.info("Данные успешно сохранены.")
        return True

    def get_clusters(self, mode_config: Dict) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = {}
        name_key = mode_config["json_field_name"]
        is_face_mode = mode_config["mode_name"] == 'face'
        
        index_to_use = self._cluster_indices['face' if is_face_mode else 'location']
        
        for cluster_id, filenames in index_to_use.items():
            if not filenames: continue
            
            # Берем информацию из первой записи для отображения имени и превью
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
            
            # Для подсчета мы просто используем длину списка из индекса
            clusters[cluster_id] = [face] * len(filenames)

        return clusters

    def get_files_for_cluster(self, mode_config: Dict, cluster_id: str) -> List[str]:
        """Возвращает список имен файлов, используя индекс. Мгновенно."""
        index_key = 'face' if mode_config["mode_name"] == 'face' else 'location'
        return sorted(self._cluster_indices[index_key].get(cluster_id, []))

    def move_images_to_cluster(self, mode_config: Dict, target_id: str, target_name: str, filenames: List[str]):
        is_face_mode = mode_config["mode_name"] == 'face'
        new_id_val = int(target_id) if target_id.isdigit() else target_id
        
        index_key = 'face' if is_face_mode else 'location'
        index = self._cluster_indices[index_key]

        for filename in filenames:
            record = self.records.get(filename)
            if not record: continue

            record.is_changed = True
            
            # --- Обновление индекса ---
            # 1. Находим старый ID и удаляем из старого списка
            old_id = ""
            if is_face_mode:
                old_id = "group" if record.image_type == 'group' else str(record.faces[0].cluster_label or -1)
            else:
                old_id = str(record.location_cluster or -1)
            if filename in index[old_id]:
                index[old_id].remove(filename)

            # 2. Обновляем модель
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
            
            # 3. Добавляем в новый список в индексе
            index[target_id].append(filename)
        
        self.newly_created_clusters = [c for c in self.newly_created_clusters if c['id'] != target_id]
        self._has_changes = True

    def rename_cluster(self, mode_config: Dict, cluster_id: str, new_name: str):
        is_face_mode = mode_config["mode_name"] == 'face'
        prefix = mode_config["name_prefix_logic"](cluster_id)
        final_new_name = prefix + new_name

        # Используем индекс для быстрого поиска нужных записей
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