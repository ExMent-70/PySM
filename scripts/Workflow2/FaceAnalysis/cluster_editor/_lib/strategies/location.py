
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy, natural_sort_key

logger = logging.getLogger(__name__)

class LocationModeStrategy(EditorStrategy):
    @property
    def mode_name(self) -> str:
        return "location"

    def get_window_title(self, session_name: str) -> str:
        return f"Редактор ЛОКАЦИЙ - {session_name}"

    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        for record in records.values():
            cid = str(record.location_cluster if record.location_cluster is not None else -1)
            cname = record.location_name or f"Loc {cid}"
            
            face = record.faces[0] if record.faces else Face(bbox=list())
            face.effective_name = cname
            face.filename = record.filename
            clusters[cid].append(face)
        return dict(clusters)

    def _build_files_cache(self, records: Dict[str, ImageRecord]) -> Dict[str, List[str]]:
        cache = defaultdict(set)
        for filename, record in records.items():
            lid = str(record.location_cluster if record.location_cluster is not None else -1)
            cache[lid].add(filename)
        return {k: sorted(list(v), key=natural_sort_key) for k, v in cache.items()}

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None:
        
        self.invalidate_cache()
        new_id_val = int(target_id) if target_id.lstrip('-').isdigit() else None
        
        for filename in filenames:
            record = records.get(filename)
            if record:
                record.location_cluster = new_id_val
                if target_id == "-1":
                    record.location_name = None
                elif target_name:
                    record.location_name = target_name

    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        self.invalidate_cache()
        files = self.get_files_for_cluster(cluster_id, records)
        for fname in files:
            records[fname].location_name = new_name
