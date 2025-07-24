# fc_lib/fc_report.py

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set  # Добавили Set
from jinja2 import Environment, FileSystemLoader, select_autoescape
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import shutil
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
from tqdm import tqdm


# Используем относительные импорты
from .fc_config import ConfigManager
from .fc_utils import normalize_path
from .fc_messages import get_message
from .fc_json_data_manager import JsonDataManager
from .fc_utils import (
    normalize_path,
    load_embeddings_and_indices,
)  # Добавили load_embeddings_and_indices

logger = logging.getLogger(__name__)

# Путь к шаблонам и скриптам внутри fc_lib
TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "face_report_template.html"
# Имена копируемых файлов
LAZYLOAD_JS_NAME = "lazyload.min.js"
REPORT_CSS_NAME = "report_style.css"
REPORT_JS_NAME = "report_script.js"


# Функция-обертка run_html_report_generation без изменений
def run_html_report_generation(
    config: ConfigManager, json_manager: JsonDataManager
) -> None:
    """
    Запускает генерацию HTML-отчета, если она включена в конфигурации.
    """
    if not config.get("task", "generate_html", default=False):
        logger.info(get_message("INFO_REPORT_DISABLED"))
        return

    # --- ИЗМЕНЕНИЕ: Убираем проверку save_jpeg ---
    # Теперь отчет может генерироваться, даже если JPEG не сохранялись,
    # но превью могут быть недоступны (будет fallback).
    # analyze_raw = config.get("task", "run_image_analysis_and_clustering", default=True)
    # save_jpeg = config.get("processing", "save_jpeg", default=False)
    # if analyze_raw and not save_jpeg:
    #     logger.error(...) # Убираем эту ошибку
    #     return
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    print("")
    print("<b>Запуск генерации HTML-отчета</b>")
    output_path = Path(config.get("paths", "output_path"))
    generator = ReportGenerator(output_path, config, json_manager)

    # Проверяем, загружены ли JSON данные (метаданные лиц)
    if not json_manager.portrait_data and not json_manager.group_data:
        logger.info("Загрузка JSON данных для отчета...")
        if not json_manager.load_data():
            logger.error("Не удалось загрузить JSON данные. Генерация отчета отменена.")
            return
        if not json_manager.portrait_data and not json_manager.group_data:
            logger.warning("Нет JSON данных для генерации отчета.")
            # Решаем генерировать пустой отчет
            # generator.generate_html_report()
            return  # Или можно выйти, если пустой отчет не нужен

    generator.generate_html_report()
    logger.debug("<b>Генерация HTML-отчета завершена</b>")
    print("")


class ReportGenerator:
    """Класс для генерации HTML-отчета и визуализации эмбеддингов."""

    def __init__(
        self, output_path: Path, config: ConfigManager, json_manager: JsonDataManager
    ):
        """
        Инициализирует генератор отчетов.
        """
        self.output_path = output_path
        self.config = config
        self.json_manager = json_manager  # Теперь содержит только метаданные
        self.folder_path = Path(self.config.get("paths", "folder_path"))
        # --- ИЗМЕНЕНИЕ: Путь к эмбеддингам ---
        self.embeddings_dir = self.output_path / "_Embeddings"
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        self.save_jpeg = self.config.get("processing", "save_jpeg", False)
        proc_conf = self.config.get("processing", {})
        self.raw_extensions: Set[str] = set(proc_conf.get("raw_extensions", []))
        self.psd_extensions: Set[str] = set(proc_conf.get("psd_extensions", []))
        self.move_files = self.config.get("task", "move_files_to_claster", False)
        self.extracted_jpg_dir = self.output_path / "_JPG"

        try:
            self.env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=select_autoescape(["html", "xml"]),
            )
            self.env.get_template(TEMPLATE_NAME)  # Проверяем доступность шаблона
            logger.debug(
                f"Jinja2 окружение настроено. Шаблоны ищутся в: {TEMPLATE_DIR.resolve()}"
            )
        except Exception as e:
            logger.error(
                f"Ошибка настройки Jinja2 или загрузки шаблона {TEMPLATE_NAME} из {TEMPLATE_DIR.resolve()}: {e}",
                exc_info=True,
            )
            self.env = None

    # --- _get_report_path без изменений ---
    # (Предполагаем, что FileManager теперь кладет файлы в нужные папки кластеров)
    def _get_report_path(
        self,
        filename_str: str,
        report_base_dir: Path,
        cluster_label: Optional[int],
        child_name: Optional[str],
        is_group: bool = False,
    ) -> Tuple[str, str]:
        original_path = Path(filename_str)
        report_filename = f"{original_path.stem}.jpg"
        folder_name = "00-Noise"
        if is_group:
            folder_name = "00-group_photos"
        elif cluster_label is not None:
            cluster_label_str = f"{cluster_label:02d}"
            name_part = child_name
            if name_part and name_part.startswith(f"{cluster_label_str}-"):
                name_part = name_part[len(cluster_label_str) + 1 :]
            if (
                not name_part
                or name_part in ["Noise", "Unknown"]
                or name_part.startswith("Unknown")
            ):
                name_part = f"Cluster_{cluster_label_str}"
            if name_part.startswith("Unknown_"):
                name_part = name_part.replace("Unknown_", "UK_")
            folder_name = (
                f"{cluster_label_str}-{name_part}" if name_part else cluster_label_str
            )
        elif child_name == "Noise":
            folder_name = "00-Noise"
        elif child_name and child_name.startswith("Unknown"):
            folder_name = f"00-{child_name}"

        moved_cluster_path = self.output_path / folder_name / report_filename
        extracted_jpg_path = self.extracted_jpg_dir / report_filename
        full_path = None

        if self.move_files and moved_cluster_path.exists():
            full_path = moved_cluster_path
            logger.debug(
                f"Найден перемещенный файл '{report_filename}' в: {moved_cluster_path}"
            )
        elif self.save_jpeg and extracted_jpg_path.exists():
            full_path = extracted_jpg_path
            logger.debug(
                f"Найден извлеченный файл '{report_filename}' в: {extracted_jpg_path}"
            )
        elif moved_cluster_path.is_file():  # Проверяем еще раз папку кластера, даже если move_files=false (на случай копирования)
            full_path = moved_cluster_path
            logger.debug(
                f"Найден файл '{report_filename}' в папке кластера (не перемещенный?): {moved_cluster_path}"
            )
        else:
            full_path = extracted_jpg_path  # Fallback на _JPG
            logger.warning(
                f"Файл '{report_filename}' не найден ни в папке кластера '{folder_name}' (move_files={self.move_files}), ни в '{self.extracted_jpg_dir}' (save_jpeg={self.save_jpeg}). Используется путь '{full_path}' как fallback."
            )

        try:
            rel_path = normalize_path(full_path, report_base_dir)
        except ValueError:
            rel_path = full_path.as_uri()
            if not hasattr(self, "_warned_paths"):
                self._warned_paths = set()
            if full_path not in self._warned_paths:
                logger.debug(
                    f"Файл {full_path.name} находится вне папки отчета {report_base_dir}. Используется URI: {rel_path}"
                )
                self._warned_paths.add(full_path)
        except Exception as norm_err:
            logger.error(
                f"Ошибка нормализации пути {full_path} отн. {report_base_dir}: {norm_err}"
            )
            rel_path = report_filename
        return report_filename, rel_path

    def generate_html_report(self) -> None:
        """Генерирует HTML-отчет."""
        if self.env is None:
            logger.error(
                "Jinja2 окружение не инициализировано. Генерация отчета невозможна."
            )
            return

        # Загрузка данных сопоставлений (matches) остается без изменений
        logger.info(get_message("INFO_LOADING_JSON_FOR_REPORT"))
        matches_file = self.output_path / "matches_portrait_to_group.json"
        matches_data = {}
        if matches_file.exists():
            try:
                with matches_file.open("r", encoding="utf-8") as f:
                    matches_data = json.load(f)
                logger.debug(
                    f"Данные сопоставлений загружены из {matches_file.resolve()}"
                )
            except Exception as e:
                logger.error(
                    get_message("ERROR_LOADING_JSON", file_path=matches_file, exc=e)
                )
        else:
            logger.warning(f"Файл сопоставлений {matches_file.resolve()} не найден.")

        report_base_dir = self.output_path
        logger.debug(
            get_message(
                "DEBUG_BASE_DIR_SELECTION",
                base_dir=report_base_dir,
                move_files=self.move_files,
            )
        )

        logger.info("Подготовка данных для HTML-отчета...")
        # Используем данные из JsonDataManager (метаданные)
        portrait_clusters = self._prepare_portrait_clusters(report_base_dir)
        matches = self._prepare_matches(matches_data, report_base_dir)
        # Визуализация теперь будет загружать эмбеддинги сама
        visualization_html = self._visualize_embeddings()

        # Расчет статистики кластеров (без изменений)
        cluster_sizes = []
        small_cluster_threshold = 3
        small_cluster_count = 0
        total_clusters_no_noise = 0
        for label_str, cluster_data in portrait_clusters.items():
            if label_str != "-1":
                size = cluster_data.get("cluster_info", {}).get("num_photos", 0)
                if size > 0:
                    cluster_sizes.append(size)
                    total_clusters_no_noise += 1
                if size < small_cluster_threshold:
                    small_cluster_count += 1
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        median_cluster_size = np.median(cluster_sizes) if cluster_sizes else 0
        logger.info(
            f"Статистика кластеров: Средний размер={avg_cluster_size:.2f}, Медианный={median_cluster_size}, Маленьких ({small_cluster_threshold})={small_cluster_count}"
        )

        # Сбор контекста для шаблона (без изменений)
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_portraits = len(self.json_manager.portrait_data)
        total_group_photos = len(self.json_manager.group_data)
        noise_count = (
            portrait_clusters.get("-1", {}).get("cluster_info", {}).get("num_photos", 0)
        )
        clustering_portrait_config = self.config.get("clustering", "portrait", {})
        matching_config = self.config.get("matching", {})
        report_config = self.config.get("report", {})
        viz_method_config = report_config.get("visualization_method", "t-SNE")
        viz_params_config = self.config.get("report", {}).get(
            viz_method_config.lower(), {}
        )

        method_descriptions = {
            "t-sne": "t-SNE (t-distributed Stochastic Neighbor Embedding): Нелинейный метод снижения размерности, хорошо сохраняющий локальную структуру данных (кластеры). Чувствителен к параметрам.",
            "pca": "PCA (Principal Component Analysis): Линейный метод снижения размерности, сохраняющий максимальную дисперсию данных (глобальную структуру). Менее подходит для визуализации сложных кластеров.",
        }
        param_descriptions = {
            "perplexity": "t-SNE: Примерное количество ближайших соседей. Влияет на баланс локальной и глобальной структуры (типично 5-50).",
            "max_iter": "t-SNE: Максимальное количество итераций оптимизации.",
            "n_iter": "t-SNE: Максимальное количество итераций оптимизации (синоним max_iter).",
            "random_state": "t-SNE/PCA: Сид генератора случайных чисел для воспроизводимости.",
            "n_components": "PCA: Количество главных компонент для сохранения (обычно 2 для 2D-визуализации).",
        }

        full_config_dict = self.config.config.copy()
        config_filename = self.config.config_path.name
        full_config_dict["config_filename"] = config_filename

        context = {
            "report_date": report_date,
            "source_path": str(self.folder_path.resolve()),
            "output_path": str(self.output_path.resolve()),
            "total_portraits": total_portraits,
            "total_group_photos": total_group_photos,
            "total_clusters": total_clusters_no_noise,
            "noise_count": noise_count,
            "total_matches": len(matches),
            "avg_cluster_size": f"{avg_cluster_size:.1f}",
            "median_cluster_size": int(median_cluster_size),
            "small_cluster_count": small_cluster_count,
            "small_cluster_threshold": small_cluster_threshold,
            "portrait_clusters": portrait_clusters,
            "matches": matches,
            "thumbnail_size": report_config.get("thumbnail_size", 150),
            "portrait_algorithm": clustering_portrait_config.get("algorithm", "N/A"),
            "portrait_eps": clustering_portrait_config.get("eps", "N/A"),
            "portrait_min_samples": clustering_portrait_config.get(
                "min_samples", "N/A"
            ),
            "match_threshold": f"{matching_config.get('match_threshold', 'N/A'):.4f}"
            if not matching_config.get("use_auto_threshold")
            else "Auto",
            "metric": clustering_portrait_config.get("metric", "cosine"),
            "visualization_path": visualization_html,
            "visualization_method": viz_method_config.upper(),
            "visualization_params": viz_params_config,
            "method_description": method_descriptions.get(
                viz_method_config.lower(), "Описание не найдено."
            ),
            "param_descriptions": param_descriptions,
            "config": full_config_dict,
        }

        logger.info("Рендеринг HTML-шаблона...")
        try:
            template = self.env.get_template(TEMPLATE_NAME)
            html_content = template.render(context)
        except Exception as e:
            logger.error(
                f"Ошибка рендеринга шаблона {TEMPLATE_NAME}: {e}", exc_info=True
            )
            return

        report_file = self.output_path / "face_clustering_report.html"
        try:  # Копирование ассетов (без изменений)
            files_to_copy = [LAZYLOAD_JS_NAME, REPORT_CSS_NAME, REPORT_JS_NAME]
            logger.debug(
                f"Копирование вспомогательных файлов отчета в {self.output_path.resolve()}..."
            )
            copy_errors = 0
            for filename in files_to_copy:
                source_path = TEMPLATE_DIR / filename
                dest_path = self.output_path / filename
                if source_path.is_file():
                    try:
                        shutil.copy2(source_path, dest_path)
                        logger.debug(f"Файл '{filename}' скопирован.")
                    except Exception as copy_e:
                        logger.error(
                            f"Не удалось скопировать '{filename}' в {self.output_path.resolve()}: {copy_e}"
                        )
                        copy_errors += 1
                else:
                    logger.error(
                        f"Не найден исходный файл '{filename}' для копирования в {source_path.resolve()}"
                    )
                    copy_errors += 1
            if copy_errors > 0:
                logger.warning(
                    f"Возникли ошибки при копировании {copy_errors} вспомогательных файлов отчета."
                )
            else:
                logger.info("Вспомогательные файлы отчета успешно скопированы.")

            report_file.write_text(html_content, encoding="utf-8")
            logger.info(
                get_message("INFO_REPORT_SAVED", output_file=report_file.resolve())
            )
        except Exception as e:
            logger.error(get_message("ERROR_SAVING_REPORT", exc=e), exc_info=True)

    # --- _prepare_portrait_clusters без изменений (использует JsonDataManager) ---
    def _prepare_portrait_clusters(
        self, report_base_dir: Path
    ) -> Dict[str, Dict[str, Any]]:
        logger.debug(get_message("INFO_PREPARING_PORTRAIT_CLUSTERS"))
        portrait_clusters_report: Dict[str, Dict[str, Any]] = {}
        portrait_data = self.json_manager.portrait_data  # Берем метаданные
        temp_clusters: Dict[str, List[Dict]] = {}
        for filename_str, data_dict in tqdm(
            portrait_data.items(), desc="Подготовка портретных кластеров"
        ):
            if (
                not isinstance(data_dict, dict)
                or "faces" not in data_dict
                or not data_dict["faces"]
            ):
                continue
            face_info = data_dict["faces"][0]
            if not isinstance(face_info, dict):
                continue
            label = face_info.get("cluster_label")
            label_str = str(label) if label is not None else "-1"
            if label_str not in temp_clusters:
                temp_clusters[label_str] = []
            gender_onnx = face_info.get("gender_faceonnx")
            age_onnx = face_info.get("age_faceonnx")
            emotion_onnx = face_info.get("emotion_faceonnx")
            beauty_onnx = face_info.get("beauty_faceonnx")
            left_eye_state = face_info.get("left_eye_state")
            right_eye_state = face_info.get("right_eye_state")
            temp_clusters[label_str].append(
                {
                    "filename_orig": filename_str,
                    "det_score": face_info.get("det_score", 0.0),
                    "child_name_orig": face_info.get("child_name"),
                    "gender_onnx": gender_onnx,
                    "age_onnx": age_onnx,
                    "emotion_onnx": emotion_onnx,
                    "beauty_onnx": f"{beauty_onnx:.2f}"
                    if isinstance(beauty_onnx, float)
                    else None,
                    "left_eye_state": left_eye_state,
                    "right_eye_state": right_eye_state,
                }
            )
        sorted_labels = sorted(
            temp_clusters.keys(), key=lambda x: float("inf") if x == "-1" else int(x)
        )
        self._warned_paths = set()
        for label_str in sorted_labels:
            files_in_cluster_temp = temp_clusters[label_str]
            num_photos = len(files_in_cluster_temp)
            if num_photos == 0:
                continue
            ages_onnx = [
                f["age_onnx"]
                for f in files_in_cluster_temp
                if isinstance(f["age_onnx"], int) and f["age_onnx"] >= 0
            ]
            genders_onnx = [
                f["gender_onnx"]
                for f in files_in_cluster_temp
                if isinstance(f["gender_onnx"], str)
            ]
            age_display = "N/A"
            if ages_onnx:
                min_age, max_age = min(ages_onnx), max(ages_onnx)
                age_display = (
                    str(min_age) if min_age == max_age else f"{min_age}-{max_age}"
                )
            gender_display = "N/A"
            if genders_onnx:
                male_count = sum(1 for g in genders_onnx if g == "Male")
                female_count = sum(1 for g in genders_onnx if g == "Female")
                total_defined = male_count + female_count
                if total_defined > 0:
                    if male_count > female_count:
                        gender_display = f"Мужской ({male_count}/{total_defined})"
                    elif female_count > male_count:
                        gender_display = f"Женский ({female_count}/{total_defined})"
                    else:
                        gender_display = (
                            f"М:{male_count}/Ж:{female_count}"
                            if male_count > 0
                            else "N/A"
                        )  # Уточнено

            representative_file_data = files_in_cluster_temp[0]
            _, representative_image_path = self._get_report_path(
                filename_str=representative_file_data["filename_orig"],
                report_base_dir=report_base_dir,
                cluster_label=int(label_str) if label_str != "-1" else None,
                child_name=representative_file_data["child_name_orig"],
                is_group=False,
            )
            final_file_list = []
            for file_data_temp in files_in_cluster_temp:
                report_filename, rel_path = self._get_report_path(
                    filename_str=file_data_temp["filename_orig"],
                    report_base_dir=report_base_dir,
                    cluster_label=int(label_str) if label_str != "-1" else None,
                    child_name=file_data_temp["child_name_orig"],
                    is_group=False,
                )
                left_s = file_data_temp.get("left_eye_state")
                right_s = file_data_temp.get("right_eye_state")
                eye_state_combined = "N/A"
                if left_s == "Closed" and right_s == "Closed":
                    eye_state_combined = "Closed"
                elif left_s == "Open" and right_s == "Open":
                    eye_state_combined = "Open"
                elif left_s == "Closed" and right_s == "Open":
                    eye_state_combined = "Left Closed"
                elif left_s == "Open" and right_s == "Closed":
                    eye_state_combined = "Right Closed"
                elif left_s and not right_s:
                    eye_state_combined = f"Left {left_s}"
                elif not left_s and right_s:
                    eye_state_combined = f"Right {right_s}"
                final_file_list.append(
                    {
                        "filename": report_filename,
                        "rel_path": rel_path,
                        "det_score": f"{file_data_temp['det_score']:.2f}",
                        "child_name": file_data_temp["child_name_orig"],
                        "gender_onnx": file_data_temp.get("gender_onnx"),
                        "age_onnx": file_data_temp.get("age_onnx"),
                        "emotion_onnx": file_data_temp.get("emotion_onnx"),
                        "beauty_onnx": file_data_temp.get("beauty_onnx"),
                        "eye_state_combined": eye_state_combined,
                        "left_eye_state": file_data_temp.get("left_eye_state"),
                        "right_eye_state": file_data_temp.get("right_eye_state"),
                    }
                )
            portrait_clusters_report[label_str] = {
                "cluster_info": {
                    "gender": gender_display,
                    "age": age_display,
                    "num_photos": num_photos,
                },
                "representative_image_path": representative_image_path,
                "child_name": final_file_list[0]["child_name"]
                if final_file_list
                else ("Шум" if label_str == "-1" else "N/A"),
                "files": final_file_list,
            }
        if hasattr(self, "_warned_paths"):
            del self._warned_paths
        logger.info(
            f"Данные для {len(portrait_clusters_report)} портретных кластеров (включая шум) готовы."
        )
        return portrait_clusters_report

    # --- _prepare_matches без изменений (использует JsonDataManager) ---
    def _prepare_matches(
        self, matches_data: Dict[str, Any], report_base_dir: Path
    ) -> Dict[str, Dict]:
        if not matches_data:
            return {}
        logger.debug(get_message("INFO_PREPARING_MATCHES"))
        matches_report = {}
        sorted_labels = sorted(matches_data.keys(), key=lambda x: int(x))
        self._warned_paths = set()
        for label_str in tqdm(sorted_labels, desc="Подготовка сопоставлений"):
            match_info = matches_data[label_str]
            try:
                portrait_label = int(label_str)
            except ValueError:
                continue
            portrait_files_from_match = match_info.get("portrait_files", [])
            processed_portrait_files = []
            child_name_for_cluster = "N/A"
            if portrait_files_from_match:
                first_portrait_filename_orig = portrait_files_from_match[0]
                portrait_file_data = self.json_manager.get_data(
                    first_portrait_filename_orig
                )
                if portrait_file_data and portrait_file_data.get("faces"):
                    child_name_for_cluster = (
                        portrait_file_data["faces"][0].get("child_name") or "N/A"
                    )
                _, rep_path = self._get_report_path(
                    filename_str=first_portrait_filename_orig,
                    report_base_dir=report_base_dir,
                    cluster_label=portrait_label,
                    child_name=child_name_for_cluster,
                    is_group=False,
                )
                processed_portrait_files.append(
                    {"rel_path": rep_path, "child_name": child_name_for_cluster}
                )
            group_files_info_from_match = match_info.get("group_files_matched", [])
            processed_group_photos = []
            for info in group_files_info_from_match:
                filename_str = info.get("group_file")
                if not filename_str:
                    continue
                report_filename, rel_path = self._get_report_path(
                    filename_str=filename_str,
                    report_base_dir=report_base_dir,
                    cluster_label=None,
                    child_name=None,
                    is_group=True,
                )
                num_faces = info.get("num_faces_matched", 0)
                min_dist = info.get("min_distance")
                processed_group_photos.append(
                    {
                        "filename": report_filename,
                        "rel_path": rel_path,
                        "num_faces": num_faces,
                        "confidence": min_dist,
                    }
                )
            processed_group_photos.sort(key=lambda x: x["filename"])
            total_faces_val = match_info.get("total_faces_matched_in_groups", 0)
            avg_conf_val = match_info.get("average_min_distance")
            matches_report[label_str] = {
                "child_name": child_name_for_cluster,
                "portrait_files": processed_portrait_files,
                "group_photos": processed_group_photos,
                "total_faces": total_faces_val,
                "avg_confidence": avg_conf_val,
            }
        if hasattr(self, "_warned_paths"):
            del self._warned_paths
        logger.info(
            f"Данные сопоставлений для {len(matches_report)} портретных кластеров готовы."
        )
        return matches_report

    # --- ИЗМЕНЕНИЕ: _visualize_embeddings загружает эмбеддинги и индекс ---
    def _visualize_embeddings(self) -> Optional[str]:
        """Генерирует интерактивную визуализацию портретных эмбеддингов с помощью Plotly."""
        logger.info(
            get_message("INFO_GENERATING_VISUALIZATION") + " (интерактивная, Plotly)"
        )

        # --- Вызываем функцию из fc_utils ---
        portrait_embeddings_array, portrait_index = load_embeddings_and_indices(
            self.embeddings_dir, "portrait", "ReportGenerator"
        )
        # --- Конец вызова ---

        # Проверяем, что данные загружены
        if portrait_embeddings_array is None or portrait_index is None:
            logger.warning(
                "Портретные эмбеддинги или индекс не найдены/не загружены. Визуализация невозможна."
            )
            return None
        if portrait_embeddings_array.size == 0 or not portrait_index:
            logger.warning(
                "Нет данных в портретных эмбеддингах или индексе для визуализации."
            )
            return None
        if len(portrait_embeddings_array) < 2:
            logger.warning(get_message("WARNING_INSUFFICIENT_DATA"))
            return None
        if portrait_embeddings_array.shape[0] != len(portrait_index):
            logger.error(
                "Несоответствие кол-ва портретных эмбеддингов и записей в индексе. Визуализация может быть некорректной."
            )
            # Продолжаем, но с ошибкой в логе
        # --- Конец загрузки ---

        logger.debug("Извлечение метаданных для визуализации из JSON...")
        labels_list = []
        hover_texts_list = []
        # Создаем обратный индекс для удобства
        index_to_filename = {v: k for k, v in portrait_index.items()}

        # Итерируем по индексам эмбеддингов (0 до N-1)
        for emb_idx in range(portrait_embeddings_array.shape[0]):
            filename = index_to_filename.get(emb_idx)
            if not filename:
                logger.warning(
                    f"Не найден filename для индекса эмбеддинга {emb_idx} при подготовке визуализации."
                )
                # Добавляем плейсхолдеры, чтобы не нарушать порядок
                labels_list.append(-1)  # Помечаем как шум
                hover_texts_list.append(f"Ошибка: Нет данных<br>Индекс: {emb_idx}")
                continue

            # Получаем метаданные из JsonDataManager
            file_data = self.json_manager.get_data(filename)
            if not file_data or "faces" not in file_data or not file_data["faces"]:
                logger.warning(
                    f"Не найдены метаданные для файла {filename} (индекс {emb_idx}) при подготовке визуализации."
                )
                labels_list.append(-1)  # Шум
                hover_texts_list.append(
                    f"Файл: {Path(filename).name}<br>Ошибка: Нет метаданных"
                )
                continue

            face_data = file_data["faces"][0]  # Берем первое лицо для портрета
            if not isinstance(face_data, dict):
                logger.warning(
                    f"Некорректный формат данных лица для файла {filename} (индекс {emb_idx})."
                )
                labels_list.append(-1)  # Шум
                hover_texts_list.append(
                    f"Файл: {Path(filename).name}<br>Ошибка: Некорректные метаданные"
                )
                continue

            cluster_label = face_data.get("cluster_label")
            child_name = face_data.get("child_name", "N/A")
            label_int = int(cluster_label) if cluster_label is not None else -1
            labels_list.append(label_int)

            # Формируем hover текст (как раньше)
            cluster_display = "Шум" if label_int == -1 else label_int
            child_display = (
                child_name
                if child_name and child_name != "Unknown" and child_name != "Noise"
                else f"Кластер {cluster_display}"
            )
            hover_text = f"<b>{child_display}</b><br>Файл: {Path(filename).name}<br>Кластер: {cluster_display}"
            hover_texts_list.append(hover_text)

        if len(labels_list) != portrait_embeddings_array.shape[0]:
            logger.error(
                "Ошибка: Длина списка меток не совпадает с количеством эмбеддингов после сбора метаданных."
            )
            return None  # Не можем продолжать

        logger.info(f"Подготовлено {len(labels_list)} точек для визуализации.")
        labels_array = np.array(labels_list)

        # --- Снижение размерности и генерация Plotly (без изменений) ---
        report_config = self.config.get("report", {})
        method = report_config.get("visualization_method", "t-SNE").lower()
        tsne_config = self.config.get("report", {}).get("tsne", {})
        pca_config = self.config.get("report", {}).get("pca", {})
        embeddings_2d = None
        title = "Интерактивная визуализация портретных эмбеддингов"
        try:
            logger.debug(f"Выполнение снижения размерности методом: {method.upper()}...")
            if method == "pca":
                n_components = pca_config.get("n_components", 2)
                random_state = pca_config.get("random_state", 42)
                reducer = PCA(n_components=n_components, random_state=random_state)
                embeddings_2d = reducer.fit_transform(portrait_embeddings_array)
                title += " (PCA)"
                logger.debug(f"Выполнена PCA с n_components={n_components}.")
            else:  # t-SNE
                perplexity = tsne_config.get("perplexity", 30)
                n_iter = tsne_config.get("max_iter", tsne_config.get("n_iter", 1000))
                random_state = tsne_config.get("random_state", 42)
                n_samples = len(portrait_embeddings_array)
                effective_perplexity = min(perplexity, max(1, n_samples - 2))
                if effective_perplexity != perplexity:
                    logger.warning(
                        f"Perplexity ({perplexity}) >= n_samples ({n_samples}). Уменьшено до {effective_perplexity}."
                    )
                effective_perplexity = max(
                    effective_perplexity, 5
                )  # Убедимся, что не слишком мало
                logger.info(
                    f"Параметры t-SNE: perplexity={effective_perplexity}, n_iter={n_iter}, random_state={random_state}"
                )
                reducer = TSNE(
                    n_components=2,
                    random_state=random_state,
                    perplexity=effective_perplexity,
                    n_iter=n_iter,
                    init="pca",
                    learning_rate="auto",
                )
                embeddings_2d = reducer.fit_transform(portrait_embeddings_array)
                title += f" (t-SNE, perplexity={effective_perplexity})"
                logger.debug("Выполнен t-SNE.")

            if embeddings_2d is None or embeddings_2d.shape[0] != len(labels_list):
                raise ValueError(
                    "Результат снижения размерности имеет неверную форму или отсутствует."
                )
            logger.debug("Снижение размерности завершено.")
        except Exception as e:
            logger.error(
                f"Ошибка при снижении размерности методом {method.upper()}: {e}",
                exc_info=True,
            )
            return None

        logger.info("Создание интерактивного графика Plotly...")
        try:  # Генерация графика Plotly (без изменений)
            df = pd.DataFrame(
                {
                    "x": embeddings_2d[:, 0],
                    "y": embeddings_2d[:, 1],
                    "label": labels_array,
                    "hover_text": hover_texts_list,
                }
            )
            df["label_str"] = df["label"].apply(
                lambda x: "Шум [-1]" if x == -1 else f"Кластер {x}"
            )
            unique_labels = sorted(
                df["label_str"].unique(),
                key=lambda s: (
                    s.startswith("Шум"),
                    int(s.split()[-1].strip("[]"))
                    if "Кластер" in s or "Шум" in s
                    else float("inf"),
                ),
            )
            colors = px.colors.qualitative.Plotly
            color_map = {
                label: colors[i % len(colors)] for i, label in enumerate(unique_labels)
            }
            noise_label_key = next(
                (lbl for lbl in unique_labels if lbl.startswith("Шум")), None
            )
            if noise_label_key:
                color_map[noise_label_key] = "grey"
            fig = go.Figure()
            for label_str in unique_labels:
                df_subset = df[df["label_str"] == label_str]
                fig.add_trace(
                    go.Scattergl(
                        x=df_subset["x"],
                        y=df_subset["y"],
                        mode="markers",
                        marker=dict(
                            color=color_map[label_str],
                            size=7,
                            opacity=0.8,
                            line=dict(width=0.5, color="DarkSlateGrey"),
                        ),
                        name=label_str,
                        text=df_subset["hover_text"],
                        hoverinfo="text",
                    )
                )
            fig.update_layout(
                title=title,
                xaxis_title="Компонента 1",
                yaxis_title="Компонента 2",
                legend_title_text="Кластер",
                hovermode="closest",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                height=750,
                margin=dict(l=40, r=40, t=80, b=40),
            )
            html_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            logger.debug("HTML код визуализации сгенерирован.")
            return html_div
        except Exception as e:
            logger.error(f"Ошибка при создании графика Plotly: {e}", exc_info=True)
            return None


