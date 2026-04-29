# analize/cluster_editor/_lib/strategies/cleaning.py

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy, natural_sort_key

logger = logging.getLogger(__name__)

class CleaningModeStrategy(EditorStrategy):
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
                    
                    if raw_name == f"Temp_Cluster_{cid}":
                        base_name = "Auto"
                    else:
                        base_name = raw_name or "Auto"
                    
                    if cid == "-1":
                        cname = "Одиночные (Noise)"
                    else:
                        prefix = self.get_name_prefix(cid)
                        if not base_name.startswith("Temp "):
                            cname = prefix + base_name
                        else:
                            cname = base_name

                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)
        return dict(clusters)

    def _build_files_cache(self, records: Dict[str, ImageRecord]) -> Dict[str, List[str]]:
        cache = defaultdict(set)
        for filename, record in records.items():
            for face in record.faces:
                if face.is_trash:
                    cache["trash"].add(filename)
                else:
                    cid = str(face.temp_cluster_label if face.temp_cluster_label is not None else -1)
                    cache[cid].add(filename)
        return {k: sorted(list(v), key=natural_sort_key) for k, v in cache.items()}

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                        records: Dict[str, ImageRecord], 
                        face_selection_map: Optional[Dict[str, Any]] = None,
                        target_name: Optional[str] = None) -> None:
        
        self.invalidate_cache()
        new_id_val = int(target_id) if target_id.lstrip('-').isdigit() else None
        clean_name = self._strip_name_prefix(target_name) if target_name else None

        for filename in filenames:
            record = records.get(filename)
            if not record: continue
            
            indices = list(range(len(record.faces)))
            if face_selection_map and filename in face_selection_map:
                selection = face_selection_map[filename]
                if isinstance(selection, list):
                    indices = selection
                else:
                    indices =[selection]
            
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
        self.invalidate_cache()
        files = self.get_files_for_cluster(cluster_id, records)
        clean_name = self._strip_name_prefix(new_name)
        
        for fname in files:
            record = records[fname]
            for face in record.faces:
                if str(face.temp_cluster_label) == cluster_id and not face.is_trash:
                    face.temp_child_name = clean_name

    def save(self, records: Dict[str, ImageRecord], paths_config: Dict[str, Path]) -> bool:
        # Вся логика сохранения перенесена в DataManager, чтобы не нарушать архитектуру
        return True