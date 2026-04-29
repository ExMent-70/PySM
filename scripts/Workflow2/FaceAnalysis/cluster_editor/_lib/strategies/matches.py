# analize/cluster_editor/_lib/strategies/matches.py

import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy, natural_sort_key

logger = logging.getLogger(__name__)

class MatchesModeStrategy(EditorStrategy):
    @property
    def mode_name(self) -> str:
        return "matches"

    def get_window_title(self, session_name: str) -> str:
        return f"Сопоставление (Портреты -> Группы) - {session_name}"
    
    def get_name_prefix(self, cluster_id: str) -> str:
        if cluster_id.isdigit(): return f"{int(cluster_id):02d}-"
        return ""

    def _strip_name_prefix(self, name: str) -> str:
        if name and '-' in name and name.split('-', 1)[0].isdigit(): 
            return name.split('-', 1)[1].strip()
        return name

    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        for record in records.values():
            if record.image_type == 'portrait' and record.faces:
                face = record.faces[0]
                cid = str(face.cluster_label if face.cluster_label is not None else -1)
                
                if cid in ["-1", "group"]: continue
                
                raw_name = face.child_name or f"Cluster {cid}"
                prefix = self.get_name_prefix(cid)
                cname = prefix + self._strip_name_prefix(raw_name)
                
                face.effective_name = cname
                face.filename = record.filename
                clusters[cid].append(face)
        return dict(clusters)

    def _build_files_cache(self, records: Dict[str, ImageRecord]) -> Dict[str, List[str]]:
        cache = defaultdict(set)
        for filename, record in records.items():
            if record.face_count > 1 or record.image_type == 'group':
                if any(f.extra_data.get('matched_portrait_cluster_label') is None for f in record.faces):
                    cache["error_matches"].add(filename)
            
            for face in record.faces:
                lbl = face.extra_data.get('matched_portrait_cluster_label')
                if lbl is not None:
                    cache[str(lbl)].add(filename)
        return {k: sorted(list(v), key=natural_sort_key) for k, v in cache.items()}

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None:
        
        self.invalidate_cache()
        if target_id == "error_matches":
            for fname in filenames:
                record = records.get(fname)
                if record:
                    for face in record.faces:
                        if source_id.isdigit():
                             if str(face.extra_data.get('matched_portrait_cluster_label')) == source_id:
                                 face.extra_data['matched_portrait_cluster_label'] = None
                                 face.extra_data['matched_child_name'] = None
            return

        new_id_val = int(target_id) if target_id.isdigit() else None
        clean_name = self._strip_name_prefix(target_name) if target_name else None
        
        if new_id_val is None: return

        for fname in filenames:
            record = records.get(fname)
            if not record: continue
            
            idx = -1
            if face_selection_map and fname in face_selection_map:
                idx = face_selection_map[fname]
            else:
                for i, f in enumerate(record.faces):
                    if f.extra_data.get('matched_portrait_cluster_label') is None:
                        idx = i; break
            
            if idx != -1 and idx < len(record.faces):
                face = record.faces[idx]
                face.extra_data['matched_portrait_cluster_label'] = new_id_val
                face.extra_data['matched_child_name'] = clean_name
                face.extra_data['match_distance'] = 0.0

    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        self.invalidate_cache()

    def save(self, records: Dict[str, ImageRecord], paths_config: Dict[str, Path]) -> bool:
        json_path = paths_config.get("json_path")
        if not json_path: return False
        
        work_dir = json_path.parent
        matches_path = work_dir / "matches_portrait_to_group.json"
        error_path = work_dir / "error_matches.json"
        
        output_matches = dict()
        clusters = self.get_clusters(records)
        sorted_ids = sorted(clusters.keys(), key=lambda x: int(x) if x.isdigit() else 9999)
        
        total_matches_count = 0

        for cid in sorted_ids:
            if cid in ["-1", "group"]: continue
            files = self.get_files_for_cluster(cid, records)
            cname = clusters[cid][0].effective_name if clusters[cid] else f"Cluster {cid}"
            
            group_photos_data = list()
            for fname in files:
                rec = records[fname]
                min_dist = 0.0
                for f in rec.faces:
                    lbl = f.extra_data.get('matched_portrait_cluster_label')
                    if lbl is not None and str(lbl) == cid:
                        min_dist = f.extra_data.get('match_distance', 0.0)
                        break
                
                group_photos_data.append({
                    "filename": fname,
                    "min_distance": min_dist,
                    "num_faces": 1
                })
            
            total_matches_count += len(group_photos_data)
            output_matches[cid] = {
                "child_name": self._strip_name_prefix(cname),
                "group_photos": group_photos_data
            }

        unmatched_files = list()
        total_errors = 0
        for filename, record in records.items():
            if record.face_count > 1 or record.image_type == 'group':
                unmatched_faces = list()
                for i, face in enumerate(record.faces):
                    if face.extra_data.get('matched_portrait_cluster_label') is None:
                        unmatched_faces.append({
                            "face_index": i, 
                            "nearest_match_distance": face.extra_data.get('match_distance', 1.0)
                        })
                
                if unmatched_faces:
                    total_errors += len(unmatched_faces)
                    unmatched_files.append({
                        "filename": filename,
                        "unmatched_count": len(unmatched_faces),
                        "faces": unmatched_faces
                    })

        try:
            with open(matches_path, 'w', encoding='utf-8') as f:
                json.dump(output_matches, f, ensure_ascii=False, indent=2)
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "description": "Manually updated via Cluster Editor",
                    "unmatched_files": unmatched_files,
                    "total": total_errors
                }, f, indent=2)
            return False 
        except Exception as e:
            logger.error(f"Matches Save Error: {e}")
            return False