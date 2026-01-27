# analize/cluster_editor/_lib/data_manager.py
"""
Модуль управления данными для редактора кластеров.

Обеспечивает:
1. Загрузку и сохранение метаданных из JSON-файлов.
2. Управление эмбеддингами.
3. Логику перемещения изображений (включая ручное сопоставление в режиме matches).
"""

import logging
import ijson
import json
import traceback
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Добавляем путь к корню проекта
try:
    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

from .data_models import ImageRecord, Face
from _common._shared import EmbeddingLoader

logger = logging.getLogger(__name__)


class ClusterDataManager:
    """
    Класс-менеджер, инкапсулирующий всю работу с данными сессии.
    """

    def __init__(self, portrait_json_path: Path, group_json_path: Optional[Path] = None):
        self.portrait_json_path = portrait_json_path
        self.group_json_path = group_json_path

        self.records: Dict[str, ImageRecord] = {}
        self.newly_created_clusters: List[Dict] = []
        
        # Индексы для быстрого поиска: {cluster_id: [filename, ...]}
        self._cluster_indices: Dict[str, Dict[str, List[str]]] = {
            'face': defaultdict(list),
            'location': defaultdict(list)
        }
        
        # Индекс совпадений для режима 'matches'
        self.matches_index: Dict[str, List[tuple[str, float]]] = defaultdict(list)
        self._cluster_id_to_name_cache: Dict[str, str] = {}
        
        # Кэш векторов
        self.vector_cache: Dict[str, np.ndarray] = {}
        self._has_unsaved_matches: bool = False

        # --- НОВОЕ ПОЛЕ: Список файлов с ошибками сопоставления ---
        self.error_matches_files: List[str] = []

    def load_data(self) -> tuple[bool, str]:
        """Загружает данные из JSON файлов и файлы эмбеддингов в память."""
        if not self.portrait_json_path.is_file():
            return False, "Эталонный JSON-файл ('info_portrait_faces.json') не найден."
        
        self.records.clear()
        self.vector_cache.clear()
        self.error_matches_files = []

        # 1. Загрузка векторов (код без изменений)
        p_emb_dir = self.portrait_json_path.parent / "_Embeddings"
        g_emb_dir = (self.group_json_path.parent / "_Embeddings") if self.group_json_path else None
        
        try:
            loader = EmbeddingLoader(p_emb_dir)
            p_vecs, p_idx = loader.load("portrait")
            if p_vecs is not None and p_idx is not None:
                for fname, row_idx in p_idx.items():
                    if row_idx < len(p_vecs):
                        self.vector_cache[fname] = p_vecs[row_idx]

            if g_emb_dir and g_emb_dir.exists():
                g_loader = EmbeddingLoader(g_emb_dir)
                g_vecs, g_idx = g_loader.load("group")
                if g_vecs is not None and g_idx is not None:
                    for key, row_idx in g_idx.items():
                        if row_idx < len(g_vecs):
                            self.vector_cache[key] = g_vecs[row_idx]
        except Exception as e:
            logger.error(f"Ошибка при загрузке векторов: {e}")

        # 2. Загрузка JSON
        try:
            with open(self.portrait_json_path, 'r', encoding='utf-8') as f:
                items = ijson.kvitems(f, '', use_float=True)
                for filename, data in items:
                    record = ImageRecord.from_dict(filename, 'portrait', dict(data))
                    if record.faces:
                        record.faces[0].embedding_key = filename
                    self.records[filename] = record
            
            if self.group_json_path and self.group_json_path.is_file():
                self._load_group_json_internal(self.group_json_path)

        except Exception as e:
            return False, f"Ошибка при чтении данных:\n{e}\n{traceback.format_exc()}"
        
        self._build_indices(after_load=True)
        self._build_matches_index() 
        logger.info(f"Загружено {len(self.records)} записей.")
        return True, ""

    def _load_group_json_internal(self, path: Path):
        """Вспомогательный метод для загрузки group json и error matches."""
        with open(path, 'r', encoding='utf-8') as f:
            items = ijson.kvitems(f, '', use_float=True)
            for filename, data in items:
                record = ImageRecord.from_dict(filename, 'group', dict(data))
                for i, face in enumerate(record.faces):
                    face.embedding_key = f"{filename}::{i}"
                self.records[filename] = record
        
        # Загрузка error_matches.json, если он есть рядом
        error_path = path.parent / "error_matches.json"
        if error_path.exists():
            try:
                with open(error_path, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                    # Сохраняем список имен файлов, где были проблемы
                    self.error_matches_files = [
                        item["filename"] for item in error_data.get("unmatched_files", [])
                    ]
                logger.info(f"Загружен список ошибок сопоставления: {len(self.error_matches_files)} файлов.")
            except Exception as e:
                logger.warning(f"Не удалось прочитать error_matches.json: {e}")
                self.error_matches_files = []
        else:
            self.error_matches_files = []

    def reload_group_data(self, group_json_path: Path) -> bool:
        """Перезагружает данные о групповых фотографиях."""
        if not group_json_path.is_file():
            logger.error(f"Файл групповых данных не найден: {group_json_path}")
            return False

        filenames_to_delete = [
            fname for fname, record in self.records.items() 
            if record.original_image_type == 'group'
        ]
        for fname in filenames_to_delete:
            del self.records[fname]
        
        try:
            self._load_group_json_internal(group_json_path)
        except Exception as e:
            logger.error(f"Ошибка при чтении нового файла групповых данных: {e}")
            self.records.clear()
            return False

        self._build_indices(after_load=True)
        self._build_matches_index()
        self._has_unsaved_matches = False
        logger.info("Групповые данные перезагружены.")
        return True

    def save_data(self) -> bool:
        """Сохраняет JSON-файлы и эмбеддинги."""
        try:
            with open(self.portrait_json_path, 'r', encoding='utf-8') as f:
                full_portrait_data = json.load(f)
            if self.group_json_path:
                with open(self.group_json_path, 'r', encoding='utf-8') as f:
                    full_group_data = json.load(f)
            else:
                full_group_data = {}
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Не удалось прочитать исходные JSON: {e}")
            return False

        for record in self.records.values():
            if not record.is_changed: continue
            
            target_dict = full_portrait_data if record.image_type == 'portrait' else full_group_data
            source_dict = full_portrait_data if record.original_image_type == 'portrait' else full_group_data
            
            if record.image_type != record.original_image_type:
                if record.filename in source_dict:
                    del source_dict[record.filename]
                target_dict[record.filename] = record.to_dict()
            elif record.filename in target_dict:
                target_dict[record.filename].update(record.to_dict())

        try:
            with open(self.portrait_json_path, 'w', encoding='utf-8') as f:
                json.dump(full_portrait_data, f, ensure_ascii=False, indent=2)
            if self.group_json_path:
                with open(self.group_json_path, 'w', encoding='utf-8') as f:
                    json.dump(full_group_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.critical(f"Ошибка записи JSON: {e}")
            return False
        
        if not self._rebuild_and_save_embeddings():
            logger.error("Ошибка сохранения эмбеддингов!")
            return False

        for record in self.records.values():
            record.commit_changes()
        self.newly_created_clusters = []
        logger.info("Все изменения сохранены.")
        return True

    def _rebuild_and_save_embeddings(self) -> bool:
        """Пересобирает массивы эмбеддингов."""
        new_p_vectors = []
        new_p_index = {}
        new_g_vectors = []
        new_g_index = {}
        
        for filename, record in self.records.items():
            if record.image_type == 'portrait':
                if not record.faces: continue
                face = record.faces[0]
                if face.embedding_key and face.embedding_key in self.vector_cache:
                    vector = self.vector_cache[face.embedding_key]
                    new_idx = len(new_p_vectors)
                    new_p_vectors.append(vector)
                    new_p_index[filename] = new_idx
                    face.embedding_key = filename 
                else:
                    logger.warning(f"Потерян вектор для портрета {filename}")

            elif record.image_type == 'group':
                for i, face in enumerate(record.faces):
                    if face.embedding_key and face.embedding_key in self.vector_cache:
                        vector = self.vector_cache[face.embedding_key]
                        new_idx = len(new_g_vectors)
                        new_g_vectors.append(vector)
                        new_key = f"{filename}::{i}"
                        new_g_index[new_key] = new_idx
                        face.embedding_key = new_key
                    else:
                        logger.warning(f"Потерян вектор для лица {i} в группе {filename}")

        emb_dir = self.portrait_json_path.parent / "_Embeddings"
        loader = EmbeddingLoader(emb_dir)
        
        ok_p = True
        if new_p_vectors:
            ok_p = loader.save("portrait", np.array(new_p_vectors), new_p_index)
        
        ok_g = True
        if self.group_json_path:
            g_emb_dir = self.group_json_path.parent / "_Embeddings"
            g_loader = EmbeddingLoader(g_emb_dir)
            if new_g_vectors:
                ok_g = g_loader.save("group", np.array(new_g_vectors), new_g_index)
            
        return ok_p and ok_g

    def _strip_name_prefix(self, name: str) -> str:
        if name and '-' in name and name.split('-', 1)[0].isdigit():
            return name.split('-', 1)[1]
        return name

    def move_images_to_cluster(self, mode_config: Dict, target_id: str, target_name: str, 
                               filenames: List[str], face_selection_map: Dict[str, int] = None):
        """Перемещает изображения (логика face/location)."""
        is_face_mode = mode_config["mode_name"] == 'face'
        new_id_val = int(target_id) if target_id.isdigit() else None
        clean_target_name = self._strip_name_prefix(target_name)
        
        for filename in filenames:
            record = self.records.get(filename)
            if not record: continue
            
            if is_face_mode:
                target_face_index = 0
                if face_selection_map and filename in face_selection_map:
                    target_face_index = face_selection_map[filename]

                if target_id == "group":
                    if record.image_type == 'portrait':
                        record.image_type = 'group'
                        if record.removed_faces:
                            record.faces.extend(record.removed_faces)
                            record.removed_faces = [] 
                        for face in record.faces:
                            face.cluster_label = None
                            face.child_name = None
                else:
                    if record.image_type == 'group':
                        record.image_type = 'portrait'
                    if record.faces:
                        if 0 <= target_face_index < len(record.faces):
                            selected_face = record.faces[target_face_index]
                            faces_to_remove = [
                                face for i, face in enumerate(record.faces) 
                                if i != target_face_index
                            ]
                            record.removed_faces.extend(faces_to_remove)
                            selected_face.cluster_label = new_id_val
                            selected_face.child_name = clean_target_name
                            record.faces = [selected_face]
            else:
                record.location_cluster = new_id_val
                record.location_name = target_name
        
        self._build_indices()

    # --- НОВЫЙ МЕТОД: Ручная привязка совпадения ---
    def assign_manual_match(self, filename: str, target_cluster_id: str, target_cluster_name: str, face_index: int):
        """
        Устанавливает ручное сопоставление для конкретного лица на групповом фото.
        """
        record = self.records.get(filename)
        if not record or not record.faces: return
        
        if 0 <= face_index < len(record.faces):
            face = record.faces[face_index]
            face.extra_data['matched_portrait_cluster_label'] = int(target_cluster_id)
            # Очищаем имя от префикса (NN-)
            clean_name = self._strip_name_prefix(target_cluster_name)
            face.extra_data['matched_child_name'] = clean_name
            # Помечаем как ручное (дистанция 0.0 или спец. флаг)
            face.extra_data['match_distance'] = 0.0
            
            # Обновляем индексы
            self._build_matches_index()
            self._has_unsaved_matches = True

    # --- НОВЫЙ МЕТОД: Удаление сопоставления ---
    def unassign_manual_match(self, filename: str, current_cluster_id: str):
        """
        Удаляет сопоставление лица с указанным кластером.
        """
        record = self.records.get(filename)
        if not record: return
        
        target_id_int = int(current_cluster_id)
        
        for face in record.faces:
            # Сбрасываем только если привязка совпадает с текущим кластером
            if face.extra_data.get('matched_portrait_cluster_label') == target_id_int:
                face.extra_data['matched_portrait_cluster_label'] = None
                face.extra_data['matched_child_name'] = None
                face.extra_data['match_distance'] = None
        
        self._build_matches_index()
        self._has_unsaved_matches = True

    def rename_cluster(self, mode_config: Dict, cluster_id: str, new_name: str):
        is_face_mode = mode_config["mode_name"] == 'face'
        files_to_rename = self.get_files_for_cluster(mode_config, cluster_id)
        for filename in files_to_rename:
            record = self.records[filename]
            if is_face_mode:
                if record.faces: record.faces[0].child_name = new_name
            else:
                record.location_name = new_name

    def is_cluster_changed(self, mode_name: str, cluster_id: str) -> bool:
        if any(c['id'] == cluster_id for c in self.newly_created_clusters):
            return True
        is_face_mode = mode_name == 'face'
        
        for record in self.records.values():
            if not record.is_changed: continue
            
            # Для режима matches нас интересуют изменения в extra_data (привязки)
            if mode_name == 'matches':
                # Это сложнее отследить через record.is_changed, так как extra_data не всегда триггерит его.
                # Но мы будем полагаться на то, что сохранение перегенерирует matches.json
                # из актуального индекса.
                # Для UI подсветки можно проверить, есть ли в этом кластере фото с manual match (0.0).
                pass

            current_id, original_id = None, None
            if is_face_mode:
                current_id = "group" if record.image_type == 'group' else str(record.faces[0].cluster_label if record.faces and record.faces[0].cluster_label is not None else -1)
                orig_type = record.original_image_type
                if orig_type == 'group': original_id = "group"
                else:
                    if record.faces:
                        orig_lbl = record.faces[0].original_cluster_label
                        original_id = str(orig_lbl if orig_lbl is not None else -1)
                    else: original_id = "-1"
            else:
                current_id = str(record.location_cluster if record.location_cluster is not None else -1)
                original_id = str(record.original_location_cluster if record.original_location_cluster is not None else -1)

            if cluster_id == current_id or cluster_id == original_id:
                return True
        return False
    
    def get_all_location_names(self) -> List[str]:
        return sorted({r.location_name for r in self.records.values() if r.location_name})

    def generate_and_save_matches_json(self, output_path: Path) -> tuple[bool, str]:
        output_data = {}
        all_portrait_cluster_ids = [
            cid for cid in self._cluster_indices['face'].keys() 
            if cid not in ["-1", "group"]
        ]
        try:
            sorted_cluster_ids = sorted(all_portrait_cluster_ids, key=int)
        except ValueError:
            sorted_cluster_ids = sorted(all_portrait_cluster_ids)

        for cluster_id in sorted_cluster_ids:
            matches = self.matches_index.get(cluster_id, [])
            child_name = self._cluster_id_to_name_cache.get(cluster_id)
            if not child_name:
                files = self._cluster_indices['face'].get(cluster_id, [])
                if files and self.records.get(files[0]) and self.records[files[0]].faces:
                    child_name = self.records[files[0]].faces[0].child_name
            if not child_name: child_name = f"Кластер {cluster_id}"
            
            group_photos = [{"filename": fn, "min_distance": dist, "num_faces": 1} for fn, dist in matches]
            
            output_data[cluster_id] = {
                "child_name": child_name.split('-', 1)[-1] if child_name[0].isdigit() and '-' in child_name else child_name,
                "group_photos": group_photos
            }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            self._has_unsaved_matches = False
            return True, f"Файл совпадений успешно сгенерирован и сохранен:\n{output_path}"
        except (IOError, TypeError) as e:
            return False, f"Ошибка сохранения: {e}"


    # --- НОВЫЙ МЕТОД: Генерация файла ошибок ---
    def generate_and_save_error_matches_json(self, output_path: Path) -> tuple[bool, str]:
        """
        Генерирует и сохраняет файл error_matches.json на основе текущего состояния данных.
        В файл попадают групповые фото, на которых есть лица без привязки.
        """
        unmatched_files = []
        total_unmatched_faces = 0

        for filename, record in self.records.items():
            if record.image_type != 'group':
                continue
            
            # Ищем лица на фото, у которых нет привязки
            unmatched_faces_on_photo = []
            for i, face in enumerate(record.faces):
                if face.extra_data.get('matched_portrait_cluster_label') is None:
                    unmatched_faces_on_photo.append({
                        "face_index": i,
                        # Если дистанции нет, ставим 1.0 (максимальное различие)
                        "nearest_match_distance": face.extra_data.get('match_distance', 1.0)
                    })
            
            if unmatched_faces_on_photo:
                total_unmatched_faces += len(unmatched_faces_on_photo)
                unmatched_files.append({
                    "filename": filename,
                    "unmatched_count": len(unmatched_faces_on_photo),
                    "faces": unmatched_faces_on_photo
                })

        output_data = {
            "description": "Список групповых фотографий с лицами, не сопоставленными ни с одним кластером (обновлено вручную).",
            "unmatched_files": unmatched_files,
            "total_unmatched_faces": total_unmatched_faces
        }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            return True, "OK"
        except IOError as e:
            return False, str(e)

    # --- НОВЫЙ МЕТОД: Комплексное сохранение для режима matches ---
    def save_matches_mode_data(self, matches_path: Path, error_path: Path) -> tuple[bool, str]:
        """
        Сохраняет и файл совпадений, и файл ошибок.
        Сбрасывает флаг изменений только если оба сохранения прошли успешно.
        """
        # 1. Сохраняем matches_portrait_to_group.json
        ok_matches, msg_matches = self.generate_and_save_matches_json(matches_path)
        if not ok_matches:
            return False, f"Ошибка при сохранении matches.json: {msg_matches}"

        # 2. Сохраняем error_matches.json
        ok_errors, msg_errors = self.generate_and_save_error_matches_json(error_path)
        if not ok_errors:
            return False, f"Ошибка при сохранении error_matches.json: {msg_errors}"

        # 3. Сбрасываем флаг изменений
        self._has_unsaved_matches = False
        return True, "Данные сопоставления и список ошибок успешно сохранены."


    def get_clusters(self, mode_config: Dict) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        is_face_mode = mode_config["mode_name"] == 'face'
        
        for record in self.records.values():
            face = record.faces[0] if record.faces else Face(bbox=[])
            
            if is_face_mode:
                if record.image_type == 'group': cluster_id = "group"
                else: cluster_id = str(face.cluster_label if face.cluster_label is not None else -1)
            else:
                cluster_id = str(record.location_cluster if record.location_cluster is not None else -1)

            if not clusters[cluster_id]:
                if is_face_mode:
                    if record.image_type == 'group': cluster_name = "_Group_Photos"
                    else:
                        cluster_name = face.child_name or f"Кластер {cluster_id}"
                        if cluster_id == "-1": cluster_name = "99-Noise"
                        elif cluster_name.startswith("Unknown"):
                             if not cluster_name.startswith("98-"): cluster_name = f"98-{cluster_name}"
                        elif cluster_id not in ["-1", "group"]:
                            prefix = mode_config['name_prefix_logic'](cluster_id)
                            clean_name = cluster_name.split('-', 1)[-1] if cluster_name[0].isdigit() and '-' in cluster_name else cluster_name
                            cluster_name = prefix + clean_name
                else:
                    cluster_name = record.location_name or f"Локация {cluster_id}"
                face.effective_name = cluster_name
            
            face.filename = record.filename
            clusters[cluster_id].append(face)
        return dict(clusters)

    def get_files_for_cluster(self, mode_config: Dict, cluster_id: str) -> List[str]:
        # --- ИЗМЕНЕНИЕ: Обработка спец-кластера 'error_matches' ---
        if cluster_id == "error_matches":
            # Возвращаем только те файлы из списка ошибок, которые реально есть в базе
            # и у которых до сих пор есть несопоставленные лица.
            valid_files = []
            for fname in self.error_matches_files:
                record = self.records.get(fname)
                if not record: continue
                # Проверяем, есть ли на фото лица БЕЗ привязки
                has_unmatched_faces = any(
                    f.extra_data.get('matched_portrait_cluster_label') is None 
                    for f in record.faces
                )
                if has_unmatched_faces:
                    valid_files.append(fname)
            return sorted(valid_files)
        # -----------------------------------------------------------
        
        index_key = 'face' if mode_config["mode_name"] == 'face' else 'location'
        return sorted(self._cluster_indices[index_key].get(cluster_id, []))

    def get_group_matches_for_cluster(self, cluster_id: str) -> List[str]:
        # --- ИЗМЕНЕНИЕ: Поддержка error_matches в этом методе тоже ---
        if cluster_id == "error_matches":
            return self.get_files_for_cluster({"mode_name": "face"}, "error_matches") # config не важен
        # -------------------------------------------------------------
        return [filename for filename, _ in self.matches_index.get(cluster_id, [])]
    
    def create_cluster(self, mode_config: Dict, new_name: str):
        index_key = 'face' if mode_config["mode_name"] == 'face' else 'location'
        existing_ids = set(self._cluster_indices[index_key].keys())
        for cluster_data in self.newly_created_clusters:
            existing_ids.add(cluster_data["id"])
        numeric_ids = {int(cid) for cid in existing_ids if cid.isdigit()}
        numeric_ids.add(0)
        new_id = max(numeric_ids) + 1
        new_id_str = str(new_id)
        prefix = mode_config['name_prefix_logic'](new_id_str)
        final_new_name = prefix + new_name.strip()
        self.newly_created_clusters.append({"id": new_id_str, "name": final_new_name})

    def delete_newly_created_cluster(self, cluster_id: str):
        self.newly_created_clusters = [c for c in self.newly_created_clusters if c['id'] != cluster_id]

    def _build_indices(self, after_load: bool = False):
        self._cluster_indices['face'].clear()
        self._cluster_indices['location'].clear()
        if after_load: self._cluster_id_to_name_cache.clear()

        for record in self.records.values():
            if record.image_type == 'group':
                face_cluster_id = "group"
            elif record.faces:
                face = record.faces[0]
                face_cluster_id = str(face.cluster_label if face.cluster_label is not None else -1)
                if after_load and face_cluster_id not in self._cluster_id_to_name_cache and face.child_name:
                    self._cluster_id_to_name_cache[face_cluster_id] = face.child_name
            else:
                face_cluster_id = "-1"
            self._cluster_indices['face'][face_cluster_id].append(record.filename)

            location_cluster_id = str(record.location_cluster if record.location_cluster is not None else -1)
            self._cluster_indices['location'][location_cluster_id].append(record.filename)

    def _build_matches_index(self):
        self.matches_index.clear()
        temp_matches = defaultdict(list)
        for record in self.records.values():
            if record.image_type != 'group': continue
            for face in record.faces:
                label = face.extra_data.get('matched_portrait_cluster_label')
                distance = face.extra_data.get('match_distance')
                if label is not None and distance is not None:
                    temp_matches[str(label)].append((record.filename, float(distance)))
        self.matches_index = {
            cid: sorted(pairs, key=lambda x: x[1]) for cid, pairs in temp_matches.items()
        }

    def has_changes(self) -> bool:
# --- Учет флага matches ---
        if self._has_unsaved_matches:
            return True

        if self.newly_created_clusters:
            return True
        return any(record.is_changed for record in self.records.values())