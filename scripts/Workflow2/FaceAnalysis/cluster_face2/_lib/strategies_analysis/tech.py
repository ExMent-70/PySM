# cluster_face/_lib/strategies_analysis/tech.py

import logging
from argparse import Namespace
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

from ..analysis_manager import AnalysisDataManager
from .base import AnalysisStrategy

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder    
except ImportError as e:
    pass

from _common import (
    icon_ok, 
    icon_warning, 
    icon_error, 
    icon_info,
    icon_save,
    icon_save_warning,
    icon_save_error
)

logger = logging.getLogger(__name__)

class TechnicalStrategy(AnalysisStrategy):
    @property
    def mode_name(self) -> str:
        return "cleaning"

    def run(self, config: Namespace, data_manager: AnalysisDataManager) -> None:
        self.log_header()
        
        # 1. Чтение параметров конфигурации
        min_score = getattr(config, "a_clear_min_score", 0.60)
        min_abs_area = getattr(config, "a_clear_min_abs_area", 2500)
        min_rel_area = getattr(config, "a_clear_min_rel_area", 0.0015)
        
        # Параметры DBSCAN (используем дефолты если не заданы)
        db_eps = getattr(config, "a_sim_threshold", 0.40)
        db_min_samples = getattr(config, "a_clear_min_claster_size", 3)
        metric = getattr(config, "a_metric", "cosine")

        logger.debug(f"Параметры фильтрации: Score>{min_score}, Area>{min_abs_area}px, Rel>{min_rel_area}")

        # 2. Фильтрация (Quality Check)
        trash_stats = {"score": 0, "abs_area": 0, "rel_area": 0, "other": 0}
        ok_count = 0
        
        # Списки для последующей кластеризации
        clustering_indices: List[int] = [] # Глобальные индексы в embeddings
        clustering_map: List[Tuple[str, int]] = [] # (filename, face_index)
        
        # Доступ к сырым данным эмбеддингов
        all_embeddings = data_manager.embeddings
        index_map = data_manager.index_map

        if all_embeddings is None:
            logger.error(f"{icon_error} Эмбеддинги не загружены.")
            return

        logger.info("<br><b><i>Анализ качества лиц...</i></b>")

        for filename, file_data in data_manager.json_data.items():
            faces = file_data.get("faces", [])
            file_indices = index_map.get(filename, [])
            
            # Получаем размеры изображения
            orig_shape = file_data.get("original_shape", [0, 0])
            img_area = orig_shape[0] * orig_shape[1] if len(orig_shape) == 2 else 0

            # Синхронизация индексов
            if len(faces) != len(file_indices):
                continue

            for i, face in enumerate(faces):
                status = "ok"
                reason = "Pass"
                
                # Проверки
                score = face.get("det_score", 0.0)
                bbox = face.get("original_bbox") # [x1, y1, x2, y2]
                
                face_area = 0
                if bbox and len(bbox) == 4:
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    face_area = w * h
                
                if score < min_score:
                    status, reason = "technical_trash", "Low Score"
                    trash_stats["score"] += 1
                elif face_area < min_abs_area:
                    status, reason = "technical_trash", "Small Abs Area"
                    trash_stats["abs_area"] += 1
                elif img_area > 0 and (face_area / img_area) < min_rel_area:
                    status, reason = "technical_trash", "Small Rel Area"
                    trash_stats["rel_area"] += 1
                
                # Обновление записи
                face["quality_status"] = status
                face["temp_cluster_label"] = None
                face["temp_child_name"] = None # Сброс старых имен
                
                if status == "ok":
                    ok_count += 1
                    global_idx = file_indices[i]
                    if global_idx < len(all_embeddings):
                        clustering_indices.append(global_idx)
                        clustering_map.append((filename, i))

        logger.info(f"{icon_info}Отбраковано: Score={trash_stats['score']}, Area={trash_stats['abs_area']}, Rel={trash_stats['rel_area']}")
        logger.info(f"{icon_info}Готовы к кластеризации: {ok_count} лиц.")

        # 3. Кластеризация (DBSCAN)
        if ok_count > 0 and clustering_indices:
            logger.info(f"<br><b><i>Запуск DBSCAN (eps={db_eps}, min={db_min_samples})...</i></b>")
            
            subset_embeddings = all_embeddings[clustering_indices]
            clusterer = DBSCAN(eps=db_eps, min_samples=db_min_samples, metric=metric)
            labels = clusterer.fit_predict(subset_embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            logger.info(f"Результат: Групп={n_clusters}, Шум(Noise)={n_noise}")

            # Применение меток
            for idx, label in enumerate(labels):
                filename, face_idx = clustering_map[idx]
                label_int = int(label)
                
                face_entry = data_manager.json_data[filename]["faces"][face_idx]
                
                if label_int != -1:
                    face_entry["temp_cluster_label"] = label_int
                    face_entry["temp_child_name"] = f"Temp_Cluster_{label_int}"
                else:
                    face_entry["temp_cluster_label"] = -1
                    face_entry["temp_child_name"] = "Noise"
        
        # 4. Сохранение
        data_manager.save_json()
        
        
        output_dir = getattr(config, "a_target_dir")
        
        # 1. Инициализация
        tv_builder = StandardTreeBuilder(icon_size=28)


        # 2. Подготовка данных
        root_node = ResourceNode("Рабочая<br>папка", Path(output_dir), "folder", "Папка с результатами AI-анализа фотографий")
        root_node.children.append(ResourceNode("info_faces.json", Path(output_dir) / "info_faces.json", "code", "Подробная информация о всех лицах обнаруженных на фотографиях текущей фотосессии"))

        tv_builder.add_section("<br>Рабочие папки и файлы", [root_node])


        # 4. Вывод
        pysm_context.log_html(tv_builder.get_html())        