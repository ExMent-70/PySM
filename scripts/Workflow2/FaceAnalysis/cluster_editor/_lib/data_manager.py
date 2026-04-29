# analize/cluster_editor/_lib/data_manager.py

import logging
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .data_models import ImageRecord, Face
from .strategies import get_strategy

try:
    from _common._shared import EmbeddingLoader
except ImportError:
    EmbeddingLoader = None

logger = logging.getLogger(__name__)

class ClusterDataManager:
    def __init__(self, working_dir: Path, reference_dir: Optional[Path] = None, mode: str = "face"):
        self.working_dir = working_dir
        self.reference_dir = reference_dir if reference_dir else working_dir
        
        self.info_json_path = self.working_dir / "info_faces.json"
        self.embeddings_dir = self.working_dir / "_Embeddings"

        # Основное хранилище данных: {filename: ImageRecord}
        self.records: Dict[str, ImageRecord] = dict()
        
        # Список новых пустых кластеров (созданных кнопкой "Создать кластер")
        self.newly_created_clusters: List[Dict] = list()
        
        # Ручные обложки для локаций (legacy/context support)
        self.manual_covers: Dict[str, str] = dict()
        self._has_unsaved_covers = False

        try:
            self.strategy = get_strategy(mode)
            logger.info(f"<b>РЕЖИМ РАБОТЫ: {self.strategy.mode_name.upper()}</b>")
        except ValueError as e:
            logger.critical(f"Failed to initialize strategy: {e}")
            raise

    def load_data(self) -> tuple[bool, str]:
        """Загружает JSON. Векторы больше не загружаются в память при старте!"""
        self.strategy.invalidate_cache()
        if not self.info_json_path.exists():
            return False, f"Файл не найден: {self.info_json_path}"
        
        self.records.clear()
        self.manual_covers.clear()
        self._has_unsaved_covers = False

        try:
            with open(self.info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for filename, file_data in data.items():
                    record = ImageRecord.from_dict(filename, file_data)
                    
                    if record.face_count == 1: 
                        record.image_type = 'portrait'
                    else: 
                        record.image_type = 'group'
                    
                    record.original_image_type = record.image_type
                    
                    all_faces = record.faces + record.removed_faces
                    for i, face in enumerate(all_faces):
                        target_idx = i
                        if face.face_index is not None:
                            target_idx = face.face_index
                        
                        if record.face_count == 1 and len(all_faces) == 1:
                             face.embedding_key = filename
                        else:
                             face.embedding_key = f"{filename}::{target_idx}"
                        
                        face.commit_changes()
                        
                    record.commit_changes()
                    self.records[filename] = record
        except Exception as e:
            return False, f"JSON load error: {e}"
        
        if self.strategy.mode_name == 'matches' and self.reference_dir != self.working_dir:
            self._load_reference_clusters()

        return True, ""

    def switch_working_session(self, new_json_path: Path):
        self.working_dir = new_json_path.parent
        self.info_json_path = new_json_path
        self.embeddings_dir = self.working_dir / "_Embeddings"
        logger.info(f"Switched working session to: {self.working_dir}")

    def _load_reference_clusters(self):
        ref_json = self.reference_dir / "info_faces.json"
        if not ref_json.exists(): 
            return

        try:
            with open(ref_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for filename, file_data in data.items():
                    if file_data.get("face_count") != 1: continue
                    if filename in self.records: continue 

                    record = ImageRecord.from_dict(filename, file_data)
                    record.image_type = 'portrait'
                    
                    for face in record.faces:
                        face.extra_data['is_reference'] = True
                    
                    self.records[filename] = record
                    
        except Exception as e:
            logger.error(f"Error loading reference JSON: {e}")

    def get_clusters(self, mode_config: Dict = None) -> Dict[str, List[Face]]:
        clusters = self.strategy.get_clusters(self.records)
        for new_c in self.newly_created_clusters:
            cid = new_c["id"]
            if cid not in clusters:
                f = Face(bbox=list(), child_name=new_c["name"])
                f.effective_name = new_c["name"]
                clusters[cid] =[f]
        return clusters

    def get_files_for_cluster(self, mode_config: Dict, cluster_id: str) -> List[str]:
        return self.strategy.get_files_for_cluster(cluster_id, self.records)

    def move_images_to_cluster(self, mode_config: Dict, target_id: str, target_name: str, 
                               filenames: List[str], face_selection_map: Dict[str, int] = None):
        self.strategy.move_images(
            source_id="", 
            target_id=target_id,
            filenames=filenames,
            records=self.records,
            face_selection_map=face_selection_map,
            target_name=target_name
        )

    def rename_cluster(self, mode_config: Dict, cluster_id: str, new_name: str):
        self.strategy.rename_cluster(cluster_id, new_name, self.records)
        for c in self.newly_created_clusters:
            if c["id"] == cluster_id:
                prefix = self.strategy.get_name_prefix(cluster_id)
                clean_new_name = new_name
                if prefix and new_name.startswith(prefix):
                    clean_new_name = new_name[len(prefix):]
                c["name"] = prefix + clean_new_name

    def create_cluster(self, mode_config: Dict, new_name: str):
        clusters = self.strategy.get_clusters(self.records)
        existing_ids = set(clusters.keys())
        for c in self.newly_created_clusters: 
            existing_ids.add(c["id"])
        
        max_id = 0
        for cid in existing_ids:
            if cid.isdigit(): 
                val = int(cid)
                if val > max_id: max_id = val
                
        new_id = str(max_id + 1)
        prefix = self.strategy.get_name_prefix(new_id)
        
        self.newly_created_clusters.append({
            "id": new_id, 
            "name": prefix + new_name.strip()
        })

    def delete_newly_created_cluster(self, cluster_id: str):
        self.newly_created_clusters =[c for c in self.newly_created_clusters if c['id'] != cluster_id]

    def is_cluster_changed(self, mode: str, cluster_id: str) -> bool:
        if any(c['id'] == cluster_id for c in self.newly_created_clusters): 
            return True
        files = self.strategy.get_files_for_cluster(cluster_id, self.records)
        for fname in files:
            rec = self.records.get(fname)
            if rec and rec.is_changed:
                return True
        return False

    def has_changes(self) -> bool:
        if self.newly_created_clusters: return True
        if self._has_unsaved_covers: return True
        return any(rec.is_changed for rec in self.records.values())

    # --- Saving Logic ---

    def _standard_json_save(self) -> bool:
        output_data = dict()
        for filename, record in self.records.items():
            if any(f.extra_data.get('is_reference') for f in record.faces):
                continue
                
            record.face_count = len(record.faces)
            output_data[filename] = record.to_dict()
        try:
            if self.info_json_path.exists():
                shutil.copy(self.info_json_path, self.info_json_path.with_suffix(".json.bak"))
            with open(self.info_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            for record in self.records.values():
                record.commit_changes()
            self.newly_created_clusters = list()
            self._has_unsaved_covers = False
            return True
        except Exception as e:
            logger.critical(f"Standard Save error: {e}")
            return False

    def _cleaning_save(self) -> bool:
        """Специфичное деструктивное сохранение: удаление лиц и пересборка NPY."""
        logger.info("DataManager: Starting destructive save for cleaning...")
        
        if not EmbeddingLoader:
            logger.error("EmbeddingLoader unavailable")
            return False

        # 1. Загружаем вектора "на лету"
        try:
            emb_loader = EmbeddingLoader(self.embeddings_dir)
            vecs, idx_map = emb_loader.load("faces")
        except Exception as e:
            logger.error(f"Error loading embeddings for cleaning: {e}")
            return False

        # Строим быстрый словарь векторов для поиска
        vector_lookup = dict()
        if vecs is not None and idx_map is not None:
            for fname, indices in idx_map.items():
                for i, row_idx in enumerate(indices):
                    if row_idx < len(vecs):
                        vector_lookup[f"{fname}::{i}"] = vecs[row_idx]
                        if len(indices) == 1 and i == 0:
                            vector_lookup[fname] = vecs[row_idx]

        new_json_data = dict()
        new_vectors = list()
        new_index_map = dict()
        files_to_remove = list()

        for filename, record in list(self.records.items()):
            valid_faces = list()
            file_indices = list()
            
            for i, face in enumerate(record.faces):
                if face.is_trash: continue
                
                # Поиск вектора
                vector = None
                if face.embedding_key and face.embedding_key in vector_lookup:
                    vector = vector_lookup[face.embedding_key]
                elif face.face_index is not None:
                    key = f"{filename}::{face.face_index}"
                    if key in vector_lookup: vector = vector_lookup[key]
                elif len(record.faces) == 1 and filename in vector_lookup:
                    vector = vector_lookup[filename]
                else:
                    key = f"{filename}::{i}"
                    if key in vector_lookup: vector = vector_lookup[key]

                if vector is not None:
                    new_idx = len(new_vectors)
                    new_vectors.append(vector)
                    file_indices.append(new_idx)
                    
                    face.face_index = len(valid_faces)
                    face.embedding_key = f"{filename}::{face.face_index}"
                    valid_faces.append(face)
                else:
                    logger.warning(f"Vector missing for face, removing to prevent corruption: {filename}")

            record.faces = valid_faces
            record.face_count = len(valid_faces)
            
            if valid_faces:
                new_json_data[filename] = record.to_dict()
                if file_indices:
                    new_index_map[filename] = file_indices
                record.commit_changes()
            else:
                files_to_remove.append(filename)

        for fname in files_to_remove:
            del self.records[fname]

        # 2. Перезапись NPY
        try:
            if new_vectors:
                arr = np.array(new_vectors, dtype=np.float32)
                emb_loader.save("faces", arr, new_index_map)
        except Exception as e:
            logger.critical(f"Cleaning: NPY Save error: {e}")
            return False

        # 3. Перезапись JSON
        try:
            with open(self.info_json_path, 'w', encoding='utf-8') as f:
                json.dump(new_json_data, f, ensure_ascii=False, indent=2)
            self.newly_created_clusters = list()
        except Exception as e:
            logger.critical(f"Cleaning: JSON Save error: {e}")
            return False

        self.strategy.invalidate_cache()    
        return True

    def save_data(self) -> bool:
        if self.strategy.mode_name == 'cleaning':
            return self._cleaning_save()
        else:
            paths_config = {
                "json_path": self.info_json_path,
                "embeddings_dir": self.embeddings_dir
            }
            self.strategy.save(self.records, paths_config)
            return self._standard_json_save()

    # --- Legacy Wrappers & Clean APIs ---

    def ingest_location_covers(self, context_covers: Dict[str, str]):
        if not context_covers: return
        name_to_id = dict()
        for record in self.records.values():
            if record.location_cluster is not None and record.location_name:
                name_to_id[record.location_name] = str(record.location_cluster)
        
        for loc_name, filename in context_covers.items():
            if loc_name in name_to_id:
                cid = name_to_id[loc_name]
                if filename in self.records:
                    self.manual_covers[cid] = filename

    def set_location_cover(self, cluster_id: str, filename: str):
        if self.strategy.mode_name == 'location':
            self.manual_covers[cluster_id] = filename
            self._has_unsaved_covers = True

    def get_representative_file(self, mode_config: Dict, cluster_id: str) -> Optional[str]:
        if self.strategy.mode_name == 'location' and cluster_id in self.manual_covers:
            cover = self.manual_covers[cluster_id]
            record = self.records.get(cover)
            if record and str(record.location_cluster) == cluster_id:
                return cover
            else:
                del self.manual_covers[cluster_id]

        files = self.strategy.get_files_for_cluster(cluster_id, self.records)
        if not files: return None
        
        first_file = files[0]
        faces = self.records[first_file].faces
        return self.strategy.get_preview_image(cluster_id, faces, self.records)

    def get_location_covers_dict(self) -> Dict[str, str]:
        result = dict()
        if self.strategy.mode_name != 'location': return result
        clusters = self.get_clusters()
        for cid, faces in clusters.items():
            if not faces: continue
            loc_name = faces[0].effective_name 
            cover_file = self.get_representative_file(dict(), cid)
            if cover_file and loc_name:
                result[loc_name] = cover_file
        return result
    
    def assign_manual_match(self, filename: str, target_cluster_id: str, target_cluster_name: str, face_index: int):
        self.strategy.move_images(
            source_id="", 
            target_id=target_cluster_id, 
            filenames=[filename], 
            records=self.records,
            face_selection_map={filename: face_index},
            target_name=target_cluster_name
        )
        
    def unassign_manual_match(self, filename: str, current_cluster_id: str):
        self.strategy.move_images(
            source_id=current_cluster_id,
            target_id="error_matches",
            filenames=[filename],
            records=self.records
        )
        
    def set_cluster_gender(self, cluster_id: str, gender: str):
        """Принудительно устанавливает пол (gender_faceonnx) для всех лиц портретного кластера."""
        if self.strategy.mode_name != 'face': 
            return
            
        files = self.get_files_for_cluster(dict(), cluster_id)
        for fname in files:
            record = self.records.get(fname)
            if not record: 
                continue
            for face in record.faces:
                if str(face.cluster_label if face.cluster_label is not None else -1) == cluster_id:
                    face.extra_data['gender_faceonnx'] = gender