# cluster_face/_lib/strategies_analysis/portraits.py

import logging
import re
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

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

from ..analysis_manager import AnalysisDataManager
from .base import AnalysisStrategy



logger = logging.getLogger(__name__)

class PortraitsStrategy(AnalysisStrategy):
    @property
    def mode_name(self) -> str:
        return "face"

    def run(self, config: Namespace, data_manager: AnalysisDataManager) -> None:
        self.log_header()
        logger.info("(<i>группировка портретных фотографий (по именам и фамилиям))</i><br>")

        # 1. Параметры
        algorithm = getattr(config, "a_algorithm", "dbscan")
        metric = getattr(config, "a_metric", "cosine")
        names_file_path_str = self._resolve_names_file_automatic(data_manager.data_dir)
        
        if not names_file_path_str:
            logger.warning(f"️{icon_warning} Отсутствует файл с именами кластеров. Для имён кластеров будет использован шаблон: <i>Cluster_X</i>")
            children_names = []
        else:
            children_names = self._load_names(Path(names_file_path_str))

        # 2. Фильтрация данных (Только портреты face_count == 1)
        logger.info("<b>Поиск портретных фотографий...</b>")
        
        # Функция-фильтр для менеджера данных
        def portrait_filter(fname, info):
            return info.get("face_count") == 1

        filenames, global_indices, embeddings = data_manager.get_subset_embeddings(portrait_filter)
        
        if not filenames:
            logger.warning(f"️{icon_warning} Портретные фотографии не найдены")
            return
            
        logger.info(f"{icon_ok} Найдено портретов: <b>{len(filenames)}</b>")

        # 3. Кластеризация
        labels = None
        if algorithm == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                logger.error(f"{icon_warning} HDBSCAN не установлен, переключение на DBSCAN.")
                algorithm = "dbscan"
            else:
                min_cluster_size = getattr(config, "a_clear_min_claster_size", 3) # Переиспользуем параметр или свой
                cluster_selection_epsilon = getattr(config, "a_hdb_cluster_selection_epsilon", 0.0)
                # min_samples для hdbscan
                min_samples = getattr(config, "a_hdb_min_samples", None)
                
                logger.info(f"️{icon_info} Настройки алгоритма кластеризации HDBSCAN: min_cluster=<b>{min_cluster_size}</b>, eps=<b>{cluster_selection_epsilon}</b>")
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric=metric,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    allow_single_cluster=False
                )
                labels = clusterer.fit_predict(embeddings)

        if algorithm == "dbscan":
            eps = getattr(config, "a_sim_threshold", 0.25)
            min_samples = getattr(config, "a_clear_min_claster_size", 3)
            logger.info(f"️{icon_info} Настройки алгоритма кластеризации DBSCAN: eps=<b>{eps}</b>, min_samples=<b>{min_samples}</b>")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = clusterer.fit_predict(embeddings)

        if labels is None: 
            return

        # Статистика
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"<br><b>Результаты группировки портретных фотографий:</b>")
        logger.info(f"{icon_ok} создано <b>{n_clusters}</b> портретных кластера(ов)")

        # 4. Сопоставление имен (Хронология)
        name_map = self._resolve_cluster_names(labels, filenames, children_names)

        # 5. Применение изменений
        updates_count = 0
        for i, label in enumerate(labels):
            filename = filenames[i]
            label_int = int(label)
            
            if label_int == -1:
                child_name = "Noise"
                final_label = None # Или -1, зависит от конвенции. Обычно для портретов None или -1.
                # В cluster_portraits.py было: final_label = None if label_int == -1 else label_int
                # Давайте сохраним -1 для явности шума, или None для "не определен".
                # Старый скрипт ставил None для Noise.
                final_label = -1 
            else:
                child_name = name_map.get(label_int, f"Cluster_{label_int}")
                final_label = label_int
            
            # Обновляем JSON
            if filename in data_manager.json_data:
                faces = data_manager.json_data[filename].get("faces", [])
                if faces:
                    faces[0]["cluster_label"] = final_label
                    faces[0]["child_name"] = child_name
                    updates_count += 1
        
        data_manager.save_json()
        logger.info(f"{icon_ok} обновлено записей в файле JSON: <b>{updates_count}</b><br>")


        target_dir = getattr(config, "a_target_dir", "")
        
        # 1. Инициализация
        tv_builder = StandardTreeBuilder(icon_size=28)

        # 2. Подготовка данных
        root_node = ResourceNode("Исходная<br>папка", Path(target_dir), "folder", "Исходная папка с результатами AI-анализа фотографий")
        root_node.children.append(ResourceNode("info_faces.json (таргет)", Path(target_dir) / "info_faces.json", "code", "Подробная информация о всех лицах обнаруженных на фотографиях текущей фотосессии"))
        root_node.children.append(ResourceNode(Path(names_file_path_str).name, Path(names_file_path_str), "txt", "Список (фамилия, имя) используемый в качестве названий портретных кластеров"))
        
        # 3. Добавление секции
        # Можно вызывать несколько раз для разных блоков
        tv_builder.add_section("Используемые ресурсы", [root_node])

        # 4. Вывод
        pysm_context.log_html(tv_builder.get_html())
               


    def _resolve_names_file_automatic(self, target_dir: Path) -> str:
        """
        Автоматический поиск файла имен на основе контекста PySM.
        Приоритет:
        1. {target_dir}/../{photo_session}_{children_file_name} (В папке Output)
        2. {target_dir}/../../{photo_session}_{children_file_name} (В корне сессии)
        3. {target_dir}/../../children.txt (В корне сессии - fallback)
        """
        if not pysm_context:
            logger.warning("Запуск вне контекста PySM. Поиск специфичного файла имен невозможен.")
            # Можно добавить fallback на поиск любого .txt файла рядом, но пока оставим так
        
        # Получаем переменные контекста
        photo_session = pysm_context.get("wf_photo_session", "")
        children_suffix = pysm_context.get("wf_children_file_name", "") 
        
        specific_name = f"{photo_session}_{children_suffix}"
        if not specific_name.endswith(".txt"):
            specific_name += ".txt"

        # target_dir = .../Output/Analysis_XXX
        output_dir = target_dir.parent          # .../Output
        session_dir = target_dir.parent.parent  # .../Session

        potential_paths = [
            output_dir / specific_name,
            session_dir / specific_name,
            session_dir / "children.txt"
        ]

        for p in potential_paths:
            if p.is_file():
                logger.debug(f"ℹ️ Список имён и фамилий загружен из файла: <i>{p.name}</i>")
                return str(p.resolve())

        logger.warning(f"️{icon_warning} Файл с именами кластеров не найден")
        return ""


    def _load_names(self, path: Path) -> List[str]:
        if not path.is_file():
            logger.warning(f"{icon_warning} Файл имен не найден: {path}")
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"{icon_error} Ошибка чтения имен: {e}")
            return []

    def _resolve_cluster_names(self, labels: np.ndarray, filenames: List[str], names: List[str]) -> Dict[int, str]:
        """Распределяет имена по кластерам на основе времени первого фото."""
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                clusters[int(label)].append(filenames[idx])

        # Сортировка по номеру в имени файла (IMG_0001.jpg)
        def get_sort_key(fname: str) -> int:
            match = re.search(r'(\d+)', fname)
            return int(match.group(1)) if match else hash(fname)

        def get_start_time(cid):
            return min(get_sort_key(f) for f in clusters[cid]) if clusters[cid] else float('inf')

        sorted_cids = sorted(clusters.keys(), key=get_start_time)
        
        name_map = {}
        for i, cid in enumerate(sorted_cids):
            if i < len(names):
                name_map[cid] = names[i]
            else:
                name_map[cid] = f"Unknown_{cid}"
        return name_map