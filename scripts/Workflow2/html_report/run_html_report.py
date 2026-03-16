# analize/generate_report/run_generate_report.py

import argparse
import datetime
import json
import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import jinja2

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None
    pysm_context = None

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, target_dir: Path, ref_dir: Path):
        """
        :param target_dir: Папка текущей сессии (где лежит matches.json и куда сохраняем отчет).
        :param ref_dir: Папка эталона (откуда берем портреты и инфо о кластерах).
        """
        self.target_dir = target_dir
        self.ref_dir = ref_dir
        
        # Пути к изображениям
        self.target_images_dir = self.target_dir / "JPG"
        self.ref_images_dir = self.ref_dir / "JPG"
        
        # Папка с шаблонами
        self.templates_dir = Path(__file__).parent / "templates"
        
        # Опционально: папка с отсортированными файлами (только для Target, для совместимости)
        self.sorted_dir = None
        try:
            # Пытаемся угадать путь к Claster_...
            session_name = self.target_dir.parent.parent.name
            photo_session = self.target_dir.name.replace("Analysis_", "")
            potential_sorted = self.target_dir.parent.parent / "Output" / f"Claster_{photo_session}"
            if potential_sorted.exists():
                self.sorted_dir = potential_sorted
        except:
            pass

        logger.info(f"Target Dir: {self.target_dir}")
        logger.info(f"Reference Dir: {self.ref_dir}")

    def _get_rel_path_to_file(self, filename: str, is_reference: bool) -> str:
        """
        Вычисляет относительный путь от папки отчета (target_dir) до файла изображения.
        
        :param is_reference: Если True, ищем файл в ref_dir. Иначе в target_dir.
        """
        # 1. Определяем базовую папку поиска
        search_roots = []
        
        if is_reference:
            # Для эталонов смотрим в Reference/JPG
            search_roots.append(self.ref_images_dir)
        else:
            # Для текущей сессии смотрим сначала в Sorted (если есть), потом в JPG
            if self.sorted_dir:
                # Поиск в подпапках sorted_dir
                found_in_sorted = list(self.sorted_dir.rglob(filename))
                if found_in_sorted:
                    search_roots.append(found_in_sorted[0].parent)
            
            search_roots.append(self.target_images_dir)

        # 2. Ищем файл
        final_path = None
        for root in search_roots:
            candidate = root / filename
            if candidate.exists():
                final_path = candidate
                break
        
        # Если не нашли, формируем путь "как должно быть"
        if not final_path:
            final_path = (self.ref_images_dir if is_reference else self.target_images_dir) / filename

        # 3. Вычисляем относительный путь
        try:
            rel_path = os.path.relpath(final_path, self.target_dir)
            return Path(rel_path).as_posix() # Windows fix: заменяем \ на /
        except ValueError:
            # Если файлы на разных дисках
            return final_path.as_posix()

    def _load_json(self, path: Path) -> Dict:
        if not path.exists(): return {}
        try:
            with path.open("r", encoding="utf-8") as f: return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка чтения {path}: {e}")
            return {}

    def _prepare_data(self) -> Optional[Dict[str, Any]]:
        logger.info("<br>Подготовка данных...")
        
        # 1. Загружаем Reference Data (Портреты)
        ref_json_path = self.ref_dir / "info_faces.json"
        ref_data = self._load_json(ref_json_path)
        if not ref_data and self.ref_dir != self.target_dir:
            logger.warning(f"Не найден info_faces.json в эталонной папке: {self.ref_dir}")

        # 2. Загружаем Target Data (Группы + Matches)
        target_json_path = self.target_dir / "info_faces.json"
        target_data = self._load_json(target_json_path)
        
        matches_path = self.target_dir / "matches_portrait_to_group.json"
        matches_json = self._load_json(matches_path)

        # --- СБОР ПОРТРЕТНЫХ КЛАСТЕРОВ (ИЗ REFERENCE) ---
        portrait_clusters: Dict[str, Dict[str, Any]] = {}
        
        # Итерируемся по эталонным данным
        for filename, data in ref_data.items():
            # Нас интересуют только портреты (face_count == 1)
            if data.get("face_count") != 1: continue
            
            faces = data.get("faces", [])
            if not faces: continue
            face = faces[0]
            
            label = str(face.get("cluster_label", -1))
            if label == "None": label = "-1"
            
            if label not in portrait_clusters:
                child_name = face.get("child_name", f"Кластер {label}")
                if label == "-1": child_name = "Шум (Ref)"
                portrait_clusters[label] = {
                    "child_name": child_name,
                    "files": []
                }
            
            # Формируем данные файла
            file_info = self._extract_face_info(filename, face)
            # Путь к картинке (is_reference=True)
            file_info["rel_path"] = self._get_rel_path_to_file(filename, is_reference=True)
            
            portrait_clusters[label]["files"].append(file_info)

        # Сортировка файлов
        for data in portrait_clusters.values():
            data["files"].sort(key=lambda x: x['filename'])


        # --- СБОР СОВПАДЕНИЙ (ИЗ TARGET) ---
        prepared_matches = {}
        
        # matches_json имеет структуру: { "label": { "child_name": "...", "group_photos": [...] } }
        for label, match_info in matches_json.items():
            group_photos_processed = []
            
            for photo in match_info.get("group_photos", []):
                # photo: {filename, min_distance, num_faces}
                fname = photo.get("filename")
                
                # Путь к картинке (is_reference=False, т.к. это групповое фото из текущей сессии)
                rel_path = self._get_rel_path_to_file(fname, is_reference=False)
                
                photo_entry = photo.copy()
                photo_entry["rel_path"] = rel_path
                photo_entry["confidence"] = photo.pop("min_distance", None)
                group_photos_processed.append(photo_entry)
            
            if group_photos_processed:
                c_name = match_info.get("child_name")
                if not c_name and label in portrait_clusters:
                    c_name = portrait_clusters[label]["child_name"]
                
                prepared_matches[label] = {
                    "child_name": c_name or f"Кластер {label}",
                    "group_photos": group_photos_processed
                }

        # --- СВОДКА ---
        # Считаем статистику для шаблона
        # Сколько всего портретов в Эталоне
        ref_portraits_count = sum(1 for d in ref_data.values() if d.get("face_count") == 1)
        # Сколько групповых в Цели
        tgt_groups_count = sum(1 for d in target_data.values() if d.get("face_count") != 1)

        summary = {
            # Ключи приведены к виду, который ожидает report_template.html
            "total_portraits": ref_portraits_count,
            "total_group_photos": tgt_groups_count,
            "total_clusters": len(portrait_clusters) - (1 if "-1" in portrait_clusters else 0),
            "noise_count": len(portrait_clusters.get("-1", {}).get("files", [])),
            "total_matches": len(prepared_matches),
            "report_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": str(self.target_dir),
            "images_path": str(self.target_images_dir)
        }

        return {
            "summary": summary, 
            "portrait_clusters": portrait_clusters,
            "matches": prepared_matches, 
            "thumbnail_size": 150,
            "is_cross_session": (self.target_dir != self.ref_dir)
        }

    def _extract_face_info(self, filename: str, face: Dict) -> Dict:
        """Извлекает атрибуты лица для отображения."""
        # Keypoints info
        kp = face.get("keypoint_analysis", {})
        l_eye = kp.get("eye_states", {}).get("left")
        r_eye = kp.get("eye_states", {}).get("right")
        eyes_comb = f"L:{l_eye}/R:{r_eye}" if l_eye and r_eye else "N/A"

        return {
            "filename": filename,
            "det_score": f"{face.get('det_score', 0.0):.2f}",
            "gender_onnx": face.get("gender_faceonnx"),
            "age_onnx": face.get("age_faceonnx"),
            "emotion_onnx": face.get("emotion_faceonnx"),
            "beauty_onnx": f"{face.get('beauty_faceonnx', 0.0):.2f}" if face.get('beauty_faceonnx') is not None else "N/A",
            "eye_state_combined": eyes_comb
        }

    def _copy_assets(self):
        logger.info("Копирование ресурсов (css, js)...")
        for asset in ["report_style.css", "report_script.js", "lazyload.min.js"]:
            source = self.templates_dir / asset
            if source.is_file():
                shutil.copy2(source, self.target_dir / asset)

    def run(self):
        context = self._prepare_data()
        if context is None: return

        try:
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.templates_dir),
                autoescape=jinja2.select_autoescape(['html'])
            )
            template = env.get_template("report_template.html")
            html_content = template.render(context)
            
            report_path = self.target_dir / "face_clustering_report.html"
            with report_path.open("w", encoding="utf-8") as f:
                f.write(html_content)
                
            self._copy_assets()
            logger.info(f"HTML-отчет успешно сгенерирован: {report_path.name}")
            
            if IS_MANAGED_RUN:
                pysm_context.log_link(url_or_path=str(report_path), text="<br>Открыть HTML-отчет")
                
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}", exc_info=True)


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация HTML-отчета.")
    p = "a_hr_"
    parser.add_argument(f"--{p}target_dir", type=str, required=True, 
                        help="Папка текущей сессии (Output/Analysis_...).")
    parser.add_argument(f"--{p}ref_dir", type=str, default=None, 
                        help="Папка эталона (Output/Analysis_...). Если нет - используется target.")
    
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()


def main():
    logger.info("<b>Генерация HTML-отчета (Unified)</b>")
    
    if not IS_MANAGED_RUN:
        logger.critical("Требуется PySM.")
        sys.exit(1)

    config = get_config()
    arg_prefix = "a_hr_"
    
    t_dir_str = getattr(config, f"{arg_prefix}target_dir")
    r_dir_str = getattr(config, f"{arg_prefix}ref_dir")
    
    target_dir = Path(t_dir_str)
    ref_dir = Path(r_dir_str) if r_dir_str else target_dir

    if not target_dir.exists():
        logger.critical(f"Target dir не найден: {target_dir}")
        sys.exit(1)

    if target_dir != ref_dir:
        logger.info(f"Режим: <b>Кросс-сессия</b>")
        logger.info(f"Цель (Группы): {target_dir.name}")
        logger.info(f"Эталон (Портреты): {ref_dir.name}")
    else:
        logger.info(f"Режим: <b>Одиночная сессия</b>")

    generator = ReportGenerator(target_dir, ref_dir)
    generator.run()

if __name__ == "__main__":
    main()