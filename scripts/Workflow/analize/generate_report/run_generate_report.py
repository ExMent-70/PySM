# analize/generate_report/run_generate_report.py

import argparse
import datetime
import json
import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import jinja2

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from _common.json_data_manager import JsonDataManager
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    JsonDataManager = None
    ConfigResolver = None
    pysm_context = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, data_dir: Path, images_dir: Path, sorted_dir: Optional[Path] = None):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.sorted_dir = sorted_dir
        self.templates_dir = Path(__file__).parent / "templates"
        logger.info(f"Папка с данными отчета (JSON): {self.data_dir}")
        logger.info(f"Папка с JPEG (резерв): {self.images_dir}")
        if self.sorted_dir:
            logger.info(f"Папка с отсортированными фото (приоритет): {self.sorted_dir}")

    def _get_image_rel_path(self, filename: str, cluster_info: Dict) -> str:
        child_name = cluster_info.get("child_name", "97-Unsorted")
        cluster_label = cluster_info.get("cluster_label")
        is_group = cluster_info.get("is_group", False)
        
        target_subfolder = ""
        if is_group:
            target_subfolder = "_Group_Photos"
        elif child_name == "Noise":
            target_subfolder = "99-Noise"
        elif child_name and child_name.startswith("Unknown"):
            target_subfolder = f"98-{child_name}"
        elif child_name and cluster_label is not None:
            clean_name = child_name.split('-', 1)[-1] if '-' in child_name else child_name
            target_subfolder = f"{int(cluster_label):02d}-{clean_name}"
        
        if self.sorted_dir and target_subfolder:
            sorted_path = self.sorted_dir / target_subfolder / filename
            if sorted_path.is_file():
                try:
                    return sorted_path.relative_to(self.data_dir).as_posix()
                except ValueError:
                    return sorted_path.as_posix()

        image_path = self.images_dir / filename
        try:
            return image_path.relative_to(self.data_dir).as_posix()
        except ValueError:
            return image_path.as_posix()

    def _prepare_data(self) -> Optional[Dict[str, Any]]:
        logger.info("Подготовка данных для отчета...")
        try:
            if not JsonDataManager:
                logger.critical("JsonDataManager не был импортирован."); return None

            json_manager = JsonDataManager(
                portrait_json_path=self.data_dir / "info_portrait_faces.json",
                group_json_path=self.data_dir / "info_group_faces.json"
            )
            if not json_manager.load_data(): return None

            matches_path = self.data_dir / "matches_portrait_to_group.json"
            matches_data = {}
            if matches_path.is_file():
                with matches_path.open("r", encoding="utf-8") as f:
                    matches_data = json.load(f)

            portrait_clusters: Dict[str, Dict[str, Any]] = {}
            for filename, data in json_manager.portrait_data.items():
                face = data.get("faces", [{}])[0]
                if not face: continue

                label = str(face.get("cluster_label", -1))
                if label not in portrait_clusters:
                    portrait_clusters[label] = {
                        "child_name": face.get("child_name", "Шум" if label == "-1" else f"Кластер {label}"),
                        "files": []
                    }
                
                cluster_info_for_path = {"child_name": face.get("child_name"), "cluster_label": face.get("cluster_label")}
                rel_path = self._get_image_rel_path(filename, cluster_info_for_path)

                file_info = {
                    "filename": filename,
                    "rel_path": rel_path,
                    "det_score": f"{face.get('det_score', 0.0):.2f}",
                    "gender_onnx": face.get("gender_faceonnx"),
                    "age_onnx": face.get("age_faceonnx"),
                    "emotion_onnx": face.get("emotion_faceonnx"),
                    "beauty_onnx": f"{face.get('beauty_faceonnx', 0.0):.2f}" if face.get('beauty_faceonnx') is not None else "N/A",
                    "left_eye_state": face.get("left_eye_state"),
                    "right_eye_state": face.get("right_eye_state"),
                }
                
                keypoint_analysis = face.get("keypoint_analysis", {})
                file_info["left_eye_state"] = keypoint_analysis.get("eye_states", {}).get("left")
                file_info["right_eye_state"] = keypoint_analysis.get("eye_states", {}).get("right")

                l_eye, r_eye = file_info["left_eye_state"], file_info["right_eye_state"]
                combined = "N/A"
                if l_eye and r_eye: combined = f"L:{l_eye}/R:{r_eye}"
                elif l_eye: combined = f"L:{l_eye}"
                elif r_eye: combined = f"R:{r_eye}"
                file_info["eye_state_combined"] = combined

                portrait_clusters[label]["files"].append(file_info)

            for data in portrait_clusters.values():
                data["files"].sort(key=lambda x: x['filename'])

            prepared_matches = {}
            for label, match_info in matches_data.items():
                group_photos_with_paths = []
                for photo in match_info.get("group_photos", []):
                    rel_path = self._get_image_rel_path(photo["filename"], {"is_group": True})
                    photo["rel_path"] = rel_path
                    photo["confidence"] = photo.pop("min_distance", None)
                    group_photos_with_paths.append(photo)
                
                prepared_matches[label] = {
                    "child_name": match_info.get("child_name", f"Кластер {label}"),
                    "group_photos": group_photos_with_paths
                }

            summary = {
                "total_portraits": len(json_manager.portrait_data),
                "total_group_photos": len(json_manager.group_data),
                "total_clusters": len(portrait_clusters) - (1 if "-1" in portrait_clusters else 0),
                "noise_count": len(portrait_clusters.get("-1", {}).get("files", [])),
                "total_matches": len(prepared_matches),
                "report_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_path": str(self.data_dir.resolve()),
                "images_path": str(self.images_dir.resolve()),
            }

            return {
                "summary": summary, "portrait_clusters": portrait_clusters,
                "matches": prepared_matches, "thumbnail_size": 150,
            }

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для отчета: {e}", exc_info=True)
            return None

    def _copy_assets(self):
        logger.info("Копирование ассетов (css, js)...")
        for asset in ["report_style.css", "report_script.js", "lazyload.min.js"]:
            source = self.templates_dir / asset
            if source.is_file():
                shutil.copy2(source, self.data_dir / asset)

    def run(self):
        context = self._prepare_data()
        if context is None:
            logger.error("Генерация отчета отменена."); return
        try:
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.templates_dir),
                autoescape=jinja2.select_autoescape(['html'])
            )
            template = env.get_template("report_template.html")
            html_content = template.render(context)
            report_path = self.data_dir / "face_clustering_report.html"
            report_path.write_text(html_content, encoding="utf-8")
            self._copy_assets()
            logger.info(f"HTML-отчет успешно сгенерирован: {report_path.resolve()}")
            if IS_MANAGED_RUN:
                pysm_context.log_link(url_or_path=str(report_path), text="<br>Открыть сгенерированный HTML-отчет")
        except Exception as e:
            logger.error(f"Ошибка при рендеринге или сохранении HTML-отчета: {e}", exc_info=True)

def main():
    logger.info("="*10 + " ЗАПУСК ГЕНЕРАЦИИ HTML-ОТЧЕТА " + "="*10)
    
    if not IS_MANAGED_RUN:
        logger.critical("Этот скрипт требует запуска из среды PySM для доступа к контексту.")
        sys.exit(1)

    # 1. Получение путей из контекста
    session_path_str = pysm_context.get("wf_session_path")
    session_name = pysm_context.get("wf_session_name")
    photo_session = pysm_context.get("wf_photo_session")
    
    if not all([session_path_str, session_name, photo_session]):
        logger.critical("Критическая ошибка: Одна или несколько переменных контекста (wf_...) не найдены.")
        sys.exit(1)
        
    # 2. Формирование путей
    base_path = Path(session_path_str) / session_name
    data_dir = base_path / "Output" / f"Analysis_{photo_session}"
    images_dir = data_dir / "JPG"
    sorted_dir = base_path / "Output" / f"Claster_{photo_session}"

    # 3. Валидация путей
    if not data_dir.is_dir():
        logger.critical(f"Папка с данными не найдена: '{data_dir}'")
        sys.exit(1)
    if not images_dir.is_dir():
        logger.critical(f"Папка с JPEG-файлами не найдена: '{images_dir}'")
        sys.exit(1)
    
    sorted_dir_for_gen = sorted_dir if sorted_dir.is_dir() else None
    if not sorted_dir_for_gen:
        logger.warning(f"Папка с отсортированными файлами не найдена: {sorted_dir}. Поиск будет только в основной папке.")

    # 4. Запуск генератора
    generator = ReportGenerator(data_dir, images_dir, sorted_dir_for_gen)
    generator.run()

if __name__ == "__main__":
    main()