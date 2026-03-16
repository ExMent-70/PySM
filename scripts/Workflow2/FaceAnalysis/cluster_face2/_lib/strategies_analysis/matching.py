# cluster_face/_lib/strategies_analysis/matching.py

import logging
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, NamedTuple
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

from ..analysis_manager import AnalysisDataManager
from .base import AnalysisStrategy
try:
    # Импорт классов API
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder
    # Импорт контекста для вывода
    from pysm_lib.pysm_context import pysm_context
except ImportError:
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

class ClusterProfile(NamedTuple):
    label: int
    vector: np.ndarray
    child_name: str

class MatchingStrategy(AnalysisStrategy):
    @property
    def mode_name(self) -> str:
        return "matches"

    def run(self, config: Namespace, data_manager: AnalysisDataManager) -> None:
        self.log_header()
        logger.info("(<i>идентификация лиц на групповых фотографиях</i>)<br>")
        
        threshold = getattr(config, "a_sim_threshold", 0.45)
        ref_dir_str = getattr(config, "a_ref_dir", None)
        metric = getattr(config, "a_metric", "cosine")

        # 1. Подготовка Target Data (куда пишем)
        target_mgr = data_manager # Это наш основной менеджер
        
        # 2. Подготовка Reference Data (откуда берем эталоны)
        if ref_dir_str:
            ref_path = Path(ref_dir_str)
            if ref_path != target_mgr.data_dir:
                logger.debug(f"Эталонные портретные фотографии загружены из папки: <i>{ref_path}</i>")
                # Создаем временный менеджер для референса
                try:
                    ref_mgr = AnalysisDataManager(ref_path)
                    if not ref_mgr.load_data():
                        logger.error(f"{icon_error} Не удалось загрузить данные эталона.")
                        return
                except Exception as e:
                    logger.error(f"{icon_error} Ошибка инициализации эталона: {e}")
                    return
            else:
                ref_mgr = target_mgr
        else:
            logger.debug("Эталонные портретные фотографии загружены из папки текущей съемки")
            ref_mgr = target_mgr

        # 3. Извлечение эталонов (Centroids)
        centroids = self._calculate_centroids(ref_mgr)
        if not centroids:
            logger.warning("Эталонные кластеры не найдены!")
            return

        # 4. Извлечение целей (Target Faces)
        # Цели - это лица из групповых фото (face_count != 1)
        targets = [] # List of (filename, face_index, embedding)
        
        # Используем хелпер менеджера, но нам нужны детали, поэтому переберем вручную
        count_targets = 0
        if target_mgr.embeddings is not None:
            for fname, info in target_mgr.json_data.items():
                if info.get("face_count") == 1: continue # Пропускаем портреты
                
                indices = target_mgr.index_map.get(fname, [])
                faces = info.get("faces", [])
                
                if len(faces) != len(indices): continue
                
                for i, idx in enumerate(indices):
                    if idx < len(target_mgr.embeddings):
                        targets.append({
                            "filename": fname,
                            "face_index": i,
                            "vector": target_mgr.embeddings[idx]
                        })
                        count_targets += 1
        
        logger.info(f"️{icon_info} Количество лиц требующих идентификации: <b>{count_targets}</b>")
        
        # --- ИЗМЕНЕНИЕ: Убран ранний выход (return), если targets пуст ---
        # Мы должны продолжить, чтобы сгенерировать пустые отчеты
        
        matches_count = 0
        
        # 5. Матчинг (Vectorized) - запускаем только если есть цели
        if targets:
            logger.info(f"<br><b>Запуск процесса идентификации (<i>threshold={threshold})...</i></b>")
            
            target_matrix = np.array([t["vector"] for t in targets])
            
            labels_order = list(centroids.keys())
            centroid_matrix = np.array([centroids[l].vector for l in labels_order])
            
            # cdist
            dists = cdist(target_matrix, centroid_matrix, metric=metric)
            
            min_indices = np.argmin(dists, axis=1)
            min_vals = np.min(dists, axis=1)
            
            # 6. Применение результатов
            for i, target in enumerate(targets):
                dist = float(min_vals[i])
                
                face = target_mgr.json_data[target["filename"]]["faces"][target["face_index"]]
                face["match_distance"] = dist
                
                if dist < threshold:
                    best_label = labels_order[min_indices[i]]
                    profile = centroids[best_label]
                    
                    face["matched_portrait_cluster_label"] = best_label
                    face["matched_child_name"] = profile.child_name
                    matches_count += 1
                else:
                    face["matched_portrait_cluster_label"] = None
                    face["matched_child_name"] = "No Match"

            logger.info(f"<br><b>Результаты идентификации лиц на групповых фотографиях</b>")
            logger.info(f"{icon_ok} идентифицировано: <b>{matches_count}</b>")
            logger.info(f"{icon_error} не идентифицировано: <b>{count_targets-matches_count}</b>")
        else:
            logger.info(f"{icon_warning} Групповые фотографии не найдены (идентификация лиц не требуется)")
        
        
        # 7. Сохранение основного JSON
        target_mgr.save_json()
        
        # 8. Генерация отчетов (matches_portrait_to_group.json и error_matches.json)
        # Выполняется всегда, даже если targets пуст
        self._save_reports(target_mgr, centroids)
        
        target_dir = getattr(config, "a_target_dir", "")
        
        # 1. Инициализация Dashboard
        tv_builder = StandardTreeBuilder(icon_size=28)

        # 2. Подготовка данных
        root_node = ResourceNode("Исходная<br>папка", Path(target_dir), "folder", "Исходная папка с результатами AI-анализа фотографий")
        root_node.children.append(ResourceNode("info_faces.json (таргет)", Path(target_dir) / "info_faces.json", "code", "Подробная информация о всех лицах обнаруженных на фотографиях текущей фотосессии"))
        root_node.children.append(ResourceNode("matches_portrait_to_group.json", Path(target_dir) / "matches_portrait_to_group.json", "code", "Список групповых фотографий для каждого идентифицированного лица"))
        root_node.children.append(ResourceNode("error_matches.json", Path(target_dir) / "error_matches.json", "code", "Список фотографий с не идентифицированными лицами"))

        if ref_dir_str:
            root_node_target = ResourceNode("Эталонная<br>папка", Path(ref_dir_str), "folder", "Эталонная папка с результатами AI-анализа портретных фотографий")
            root_node_target.children.append(ResourceNode("info_faces.json (эталон)", Path(ref_dir_str) / "info_faces.json", "code", "Информация об эталонных портретах"))
            tv_builder.add_section("Используемые ресурсы", [root_node, root_node_target])
        else:
            tv_builder.add_section("Используемые ресурсы", [root_node])

        # 4. Вывод
        pysm_context.log_html(tv_builder.get_html())

    def _calculate_centroids(self, mgr: AnalysisDataManager) -> Dict[int, ClusterProfile]:
        """Считает средние вектора для кластеров из референсного менеджера."""
        ref_vectors = defaultdict(list)
        ref_names = {}
        
        for fname, info in mgr.json_data.items():
            if info.get("face_count") != 1: continue
            
            indices = mgr.index_map.get(fname, [])
            faces = info.get("faces", [])
            
            if not faces or not indices: continue
            
            face = faces[0]
            label = face.get("cluster_label")
            
            if label is not None and label != -1 and mgr.embeddings is not None:
                idx = indices[0]
                if idx < len(mgr.embeddings):
                    ref_vectors[int(label)].append(mgr.embeddings[idx])
                    if int(label) not in ref_names:
                        ref_names[int(label)] = face.get("child_name", f"Unknown_{label}")

        profiles = {}
        for label, vecs in ref_vectors.items():
            arr = np.array(vecs)
            mean = np.mean(arr, axis=0)
            # Normalize
            norm = np.linalg.norm(mean)
            if norm > 1e-6: mean /= norm
            
            profiles[label] = ClusterProfile(label, mean, ref_names[label])
            
        logger.info(f"️{icon_info} Количество эталонов: <b>{len(profiles)}</b>")
        return profiles

    def _save_reports(self, mgr: AnalysisDataManager, centroids: Dict[int, ClusterProfile]):
        """Генерирует дополнительные отчеты."""
        out_dir = mgr.data_dir
        
        # Report 1: Matches
        # Инициализируем отчет ВСЕМИ эталонами, даже если для них нет совпадений
        matches_report = {}
        for lbl, prof in centroids.items():
            matches_report[str(lbl)] = {"child_name": prof.child_name, "group_photos": []}
            
        for fname, info in mgr.json_data.items():
            if info.get("face_count") == 1: continue
            
            # Группируем матчи по файлу
            found_labels = defaultdict(list)
            for face in info.get("faces", []):
                lbl = face.get("matched_portrait_cluster_label")
                dst = face.get("match_distance")
                if lbl is not None:
                    found_labels[str(lbl)].append(dst)
            
            for lbl_str, dists in found_labels.items():
                if lbl_str in matches_report:
                    matches_report[lbl_str]["group_photos"].append({
                        "filename": fname,
                        "min_distance": min(dists),
                        "num_faces": len(dists)
                    })
        
        # --- ИЗМЕНЕНИЕ: Убрана фильтрация пустых записей ---
        # Мы сохраняем matches_report как есть, чтобы видеть пустые списки для ненайденных детей.
        
        with (out_dir / "matches_portrait_to_group.json").open("w", encoding="utf-8") as f:
            json.dump(matches_report, f, indent=2, ensure_ascii=False)
        logger.info(f"{icon_save} файл <i>matches_portrait_to_group.json</i> сохранён")    

        # Report 2: Errors
        errors_report = {"unmatched_files": []}
        total_err = 0
        
        for fname, info in mgr.json_data.items():
            if info.get("face_count") == 1: continue
            
            unmatched = []
            for i, face in enumerate(info.get("faces", [])):
                if face.get("matched_portrait_cluster_label") is None:
                    unmatched.append({
                        "face_index": i,
                        "nearest_match_distance": face.get("match_distance", -1.0)
                    })
            
            if unmatched:
                errors_report["unmatched_files"].append({
                    "filename": fname,
                    "unmatched_count": len(unmatched),
                    "faces": unmatched
                })
                total_err += len(unmatched)
        
        errors_report["total"] = total_err
        
        with (out_dir / "error_matches.json").open("w", encoding="utf-8") as f:
            json.dump(errors_report, f, indent=2, ensure_ascii=False)
        logger.info(f"{icon_save} файл <i>error_matches.json</i> сохранён<br>")