# cluster_face/_lib/strategies_analysis/portraits.py

import logging
from argparse import Namespace
from pathlib import Path
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
    pysm_context = None


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
from ..student_ids import (
    StudentIdList,
    build_cluster_student_map,
    build_student_ids_order_context_key,
    load_student_ids_order,
    remove_legacy_name_fields,
)
from .base import AnalysisStrategy



logger = logging.getLogger(__name__)

class PortraitsStrategy(AnalysisStrategy):
    @property
    def mode_name(self) -> str:
        return "face"

    def run(self, config: Namespace, data_manager: AnalysisDataManager) -> None:
        self.log_header()
        logger.info("(<i>группировка портретных фотографий по student_id</i><br>")

        # 1. Параметры
        algorithm = getattr(config, "a_algorithm", "dbscan")
        metric = getattr(config, "a_metric", "cosine")
        student_id_list = self._load_student_ids_order()
        logger.info(
            f"{icon_ok} Загружен список идентификаторов: <b>{len(student_id_list.student_ids)}</b>, "
            f"list_id=<b>{student_id_list.list_id}</b>, "
            f"контекст=<i>{student_id_list.context_key}</i>"
        )

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

        # 4. Сопоставление student_id по хронологии первой фотографии кластера
        student_id_map = build_cluster_student_map(
            labels, filenames, student_id_list.student_ids
        )
        automatic_assignment = n_clusters == len(student_id_list.student_ids)
        if automatic_assignment:
            logger.info(
                f"{icon_ok} Кластеры и student_id сопоставлены: <b>{len(student_id_map)}</b>"
            )
        else:
            logger.warning(
                f"{icon_warning} Количество портретных кластеров ({n_clusters}) "
                f"не совпадает с количеством student_id "
                f"({len(student_id_list.student_ids)})."
            )
            logger.warning(
                f"{icon_warning} Частичное назначение по порядку отключено: "
                "портретные кластеры будут сохранены с student_id=null. "
                "Назначьте учеников и исправьте объединение/разделение "
                "кластеров в cluster_editor (режим face)."
            )

        # 5. Применение изменений
        updates_count = 0
        for i, label in enumerate(labels):
            filename = filenames[i]
            label_int = int(label)
            
            if label_int == -1:
                student_id = None
                final_label = -1
            else:
                student_id = student_id_map.get(label_int)
                final_label = label_int
            
            # Обновляем JSON
            if filename in data_manager.json_data:
                faces = data_manager.json_data[filename].get("faces", [])
                if faces:
                    face = faces[0]
                    face["cluster_label"] = final_label
                    face["student_id"] = student_id
                    remove_legacy_name_fields(face)
                    updates_count += 1
        
        data_manager.save_json()
        logger.info(f"{icon_ok} обновлено записей в файле JSON: <b>{updates_count}</b><br>")
        if not automatic_assignment:
            logger.warning(
                f"{icon_warning} Требуется ручная идентификация портретных "
                "кластеров перед запуском режима matches.<br>"
            )


        target_dir = getattr(config, "a_target_dir", "")
        
        # 1. Инициализация
        tv_builder = StandardTreeBuilder(icon_size=28)

        # 2. Подготовка данных
        root_node = ResourceNode("Исходная<br>папка", Path(target_dir), "folder", "Исходная папка с результатами AI-анализа фотографий")
        root_node.children.append(ResourceNode("info_faces.json (таргет)", Path(target_dir) / "info_faces.json", "code", "Подробная информация о всех лицах обнаруженных на фотографиях текущей фотосессии"))
        
        # 3. Добавление секции
        # Можно вызывать несколько раз для разных блоков
        tv_builder.add_section("Используемые ресурсы", [root_node])

        # 4. Вывод
        pysm_context.log_html(tv_builder.get_html())
               


    def _load_student_ids_order(self) -> StudentIdList:
        """Загружает обязательный порядок student_id из контекста PySM."""

        if not pysm_context:
            raise RuntimeError(
                "Контекст PySM недоступен: невозможно загрузить порядок "
                "student_id текущей фотосессии."
            )

        photo_session = pysm_context.get("wf_photo_session", "")
        context_key = build_student_ids_order_context_key(photo_session)
        raw_value = pysm_context.get_structured(context_key, default=None)
        if raw_value is None:
            raise ValueError(
                f"В контексте отсутствует порядок student_id: {context_key}. "
                "Сохраните список текущей фотосессии в list_create."
            )
        return load_student_ids_order(raw_value, context_key)
