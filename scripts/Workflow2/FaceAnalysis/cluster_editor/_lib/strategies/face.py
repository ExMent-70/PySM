
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy, natural_sort_key

logger = logging.getLogger(__name__)

class FaceModeStrategy(EditorStrategy):
    @property
    def mode_name(self) -> str:
        return "face"

    def get_window_title(self, session_name: str) -> str:
        return f"Редактор ЛИЦ (Портреты) - {session_name}"

    def get_name_prefix(self, cluster_id: str) -> str:
        if cluster_id.isdigit():
            return f"{int(cluster_id):02d}-"
        return ""

    def normalize_cluster_name(self, name: str) -> str:
        if name and '-' in name:
            parts = name.split('-', 1)
            if parts[0].isdigit():
                return parts[1].strip()
        return name

    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        for record in records.values():
            if record.image_type == 'group' or record.face_count != 1:
                cid = "group"
                cname = "_Group_Photos"
                face = record.faces[0] if record.faces else Face(bbox=list())
                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)
                continue

            if record.faces:
                face = record.faces[0]
                cid = str(face.cluster_label if face.cluster_label is not None else -1)
                
                if cid == "-1":
                    cname = "99-Noise"
                elif cid != "-1":
                    prefix = self.get_name_prefix(cid)
                    label = (
                        self.student_label(face.student_id)
                        or f"Не назначен [Cluster {cid}]"
                    )
                    cname = prefix + label
                else:
                    cname = f"Cluster {cid}"

                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)
        return dict(clusters)

    def _build_files_cache(self, records: Dict[str, ImageRecord]) -> Dict[str, List[str]]:
        cache = defaultdict(set)
        for filename, record in records.items():
            if record.image_type == 'group' or record.face_count != 1:
                cache["group"].add(filename)
            else:
                if record.face_count == 1 and record.faces:
                    face = record.faces[0]
                    fid = str(face.cluster_label if face.cluster_label is not None else -1)
                    cache[fid].add(filename)
        return {k: sorted(list(v), key=natural_sort_key) for k, v in cache.items()}

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None:
        
        self.invalidate_cache()
        new_id_val = int(target_id) if target_id.lstrip('-').isdigit() else None
        target_student_id = target_name or None

        for filename in filenames:
            record = records.get(filename)
            if not record: continue
            
            if target_id == "group":
                if record.image_type == 'portrait' or record.face_count == 1:
                    record.image_type = 'group'
                    if record.removed_faces:
                        record.faces.extend(record.removed_faces)
                        record.removed_faces = list()
                        record.faces.sort(key=lambda f: f.face_index if f.face_index is not None else float('inf'))

                    record.face_count = len(record.faces)
                    for f in record.faces: 
                        f.cluster_label = None
                        f.student_id = None
            else:
                if record.image_type == 'group' or record.face_count > 1:
                    record.image_type = 'portrait'
                
                target_idx = 0
                if face_selection_map and filename in face_selection_map:
                    target_idx = face_selection_map[filename]
                
                if record.faces and 0 <= target_idx < len(record.faces):
                    selected_face = record.faces[target_idx]
                    faces_to_remove =[f for i, f in enumerate(record.faces) if i != target_idx]
                    record.removed_faces.extend(faces_to_remove)
                    
                    record.faces =[selected_face]
                    record.face_count = 1
                    
                    if target_id == "-1":
                        selected_face.cluster_label = -1
                        selected_face.student_id = None
                    else:
                        selected_face.cluster_label = new_id_val
                        selected_face.student_id = target_student_id

    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        if cluster_id in ["group", "-1"]: return
        self.invalidate_cache()
        files = self.get_files_for_cluster(cluster_id, records)
        for fname in files:
            record = records[fname]
            if record.face_count == 1 and record.faces:
                record.faces[0].student_id = new_name
