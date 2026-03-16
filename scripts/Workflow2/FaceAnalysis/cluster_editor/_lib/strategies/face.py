# analize/cluster_editor/_lib/strategies/face.py

import logging
import re
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy

logger = logging.getLogger(__name__)

class FaceModeStrategy(EditorStrategy):
    """
    Стратегия для режима 'face' (Портретная кластеризация).
    Управляет разделением на портреты и групповые фото, а также
    индивидуальными метками лиц.
    """

    @property
    def mode_name(self) -> str:
        return "face"

    def get_window_title(self, session_name: str) -> str:
        return f"Редактор ЛИЦ (Портреты) - {session_name}"

    def get_name_prefix(self, cluster_id: str) -> str:
        """Возвращает префикс 'XX-' для числовых ID."""
        if cluster_id.isdigit():
            return f"{int(cluster_id):02d}-"
        return ""

    def _strip_name_prefix(self, name: str) -> str:
        if name and '-' in name:
            parts = name.split('-', 1)
            if parts[0].isdigit():
                return parts[1].strip()
        return name

    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)

        for record in records.values():
            # 1. Групповые фото
            if record.image_type == 'group' or record.face_count != 1:
                cid = "group"
                cname = "_Group_Photos"
                face = record.faces[0] if record.faces else Face(bbox=[])
                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)
                continue

            # 2. Портреты
            if record.faces:
                face = record.faces[0]
                cid = str(face.cluster_label if face.cluster_label is not None else -1)
                
                raw_name = face.child_name or f"Cluster {cid}"
                if cid == "-1":
                    cname = "99-Noise"
                elif cid != "-1":
                    prefix = self.get_name_prefix(cid)
                    cname = prefix + self._strip_name_prefix(raw_name)
                else:
                    cname = raw_name

                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)

        return dict(clusters)

    def get_files_for_cluster(self, cluster_id: str, records: Dict[str, ImageRecord]) -> List[str]:
        result = []
        for filename, record in records.items():
            if cluster_id == "group":
                if record.image_type == 'group' or record.face_count != 1:
                    result.append(filename)
            else:
                if record.face_count == 1 and record.faces:
                    face = record.faces[0]
                    fid = str(face.cluster_label if face.cluster_label is not None else -1)
                    if fid == cluster_id:
                        result.append(filename)
        
        # --- ИСПРАВЛЕНИЕ: Натуральная сортировка ---
        # Разбивает строку на текст и числа: 'photo-10.jpg' -> ['photo-', 10, '.jpg']
        # Это гарантирует правильный порядок.
        def natural_keys(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
            
        return sorted(result, key=natural_keys)

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None:
        
        new_id_val = int(target_id) if target_id.lstrip('-').isdigit() else None
        clean_name = self._strip_name_prefix(target_name) if target_name else None

        for filename in filenames:
            record = records.get(filename)
            if not record: continue
            
            # --- Сценарий A: Перенос В корзину "Group" ---
            if target_id == "group":
                if record.image_type == 'portrait' or record.face_count == 1:
                    record.image_type = 'group'
                    
                    if record.removed_faces:
                        # 1. Объединяем списки
                        record.faces.extend(record.removed_faces)
                        record.removed_faces = []
                        
                        # 2. СПОСОБ Б: Сортировка по face_index
                        # Гарантирует соответствие индекса в списке индексу в файле векторов
                        record.faces.sort(key=lambda f: f.face_index if f.face_index is not None else float('inf'))

                    record.face_count = len(record.faces)
                    for f in record.faces: 
                        f.cluster_label = None
                        f.child_name = None
            
            # --- Сценарий B: Перенос В обычный кластер (или Noise) ---
            else:
                if record.image_type == 'group' or record.face_count > 1:
                    record.image_type = 'portrait'
                
                target_idx = 0
                if face_selection_map and filename in face_selection_map:
                    target_idx = face_selection_map[filename]
                
                if record.faces and 0 <= target_idx < len(record.faces):
                    selected_face = record.faces[target_idx]
                    
                    faces_to_remove = [f for i, f in enumerate(record.faces) if i != target_idx]
                    record.removed_faces.extend(faces_to_remove)
                    
                    record.faces = [selected_face]
                    record.face_count = 1
                    
                    if target_id == "-1":
                        selected_face.cluster_label = -1
                        selected_face.child_name = "Noise"
                    else:
                        selected_face.cluster_label = new_id_val
                        if clean_name:
                            selected_face.child_name = clean_name

    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        if cluster_id in ["group", "-1"]: return

        files = self.get_files_for_cluster(cluster_id, records)
        clean_name = self._strip_name_prefix(new_name)
        
        for fname in files:
            record = records[fname]
            if record.face_count == 1 and record.faces:
                record.faces[0].child_name = clean_name

    def save(self, records: Dict[str, ImageRecord], paths_config: Dict[str, Path]) -> bool:
        return True