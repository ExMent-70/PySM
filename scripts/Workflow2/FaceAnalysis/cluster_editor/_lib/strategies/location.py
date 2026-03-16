# analize/cluster_editor/_lib/strategies/location.py

import logging
import re
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

from ..data_models import ImageRecord, Face
from .base import EditorStrategy

logger = logging.getLogger(__name__)

class LocationModeStrategy(EditorStrategy):
    """
    Стратегия для режима 'location'.
    Группирует файлы по location_cluster.
    Игнорирует индивидуальные лица при группировке (оперирует файлом целиком).
    """

    @property
    def mode_name(self) -> str:
        return "location"

    def get_window_title(self, session_name: str) -> str:
        return f"Редактор ЛОКАЦИЙ - {session_name}"

    def get_clusters(self, records: Dict[str, ImageRecord]) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        
        for record in records.values():
            cid = str(record.location_cluster if record.location_cluster is not None else -1)
            # Имя локации или дефолтное
            cname = record.location_name or f"Loc {cid}"
            
            # Для UI используем первое лицо как "носитель" информации о файле.
            # Если лиц нет, создаем пустой объект Face с привязкой к имени файла.
            face = record.faces[0] if record.faces else Face(bbox=[])
            
            face.effective_name = cname
            face.filename = record.filename
            
            clusters[cid].append(face)
            
        return dict(clusters)

    def get_files_for_cluster(self, cluster_id: str, records: Dict[str, ImageRecord]) -> List[str]:
        result = []
        for filename, record in records.items():
            lid = str(record.location_cluster if record.location_cluster is not None else -1)
            if lid == cluster_id:
                result.append(filename)
        
        # --- ИСПРАВЛЕНИЕ: Натуральная сортировка ---
        def natural_keys(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
            
        return sorted(result, key=natural_keys)

    def move_images(self, source_id: str, target_id: str, filenames: List[str], 
                    records: Dict[str, ImageRecord], 
                    face_selection_map: Optional[Dict[str, int]] = None,
                    target_name: Optional[str] = None) -> None:
        
        new_id_val = int(target_id) if target_id.lstrip('-').isdigit() else None
        
        for filename in filenames:
            record = records.get(filename)
            if record:
                record.location_cluster = new_id_val
                
                # Обновляем имя локации
                if target_id == "-1":
                    record.location_name = None
                elif target_name:
                    record.location_name = target_name

    def rename_cluster(self, cluster_id: str, new_name: str, records: Dict[str, ImageRecord]) -> None:
        files = self.get_files_for_cluster(cluster_id, records)
        for fname in files:
            # Массовое переименование всех файлов в кластере
            records[fname].location_name = new_name

    def save(self, records: Dict[str, ImageRecord], paths_config: Dict[str, Path]) -> bool:
        # Стандартное сохранение
        return True