"""Генерация HTML-отчёта по портретным кластерам и групповым совпадениям."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import jinja2

from report_data import (
    load_optional_json,
    load_required_json,
    prepare_matches,
    prepare_portrait_clusters,
)
from student_roster import StudentRoster, load_student_roster


try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[3]
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
    """Собирает проверенный контекст и атомарно создаёт HTML-отчёт."""

    def __init__(self, target_dir: Path, ref_dir: Path, roster: StudentRoster):
        self.target_dir = target_dir
        self.ref_dir = ref_dir
        self.roster = roster
        self.target_images_dir = self.target_dir / "JPG"
        self.ref_images_dir = self.ref_dir / "JPG"
        self.templates_dir = Path(__file__).parent / "templates"
        self.sorted_dir = self._find_sorted_dir()

        logger.info(f"Target Dir: {self.target_dir}")
        logger.info(f"Reference Dir: {self.ref_dir}")
        logger.info(f"Список учеников: {self.roster.path}")

    def _find_sorted_dir(self) -> Path | None:
        """Ищет необязательную папку с отсортированными групповыми фото."""

        try:
            photo_session = self.target_dir.name.replace("Analysis_", "")
            potential = (
                self.target_dir.parent.parent
                / "Output"
                / f"Claster_{photo_session}"
            )
            return potential if potential.exists() else None
        except (IndexError, OSError):
            return None

    def _get_rel_path_to_file(self, filename: str, is_reference: bool) -> str:
        """Вычисляет путь к изображению относительно папки отчёта."""

        search_roots: list[Path] = []
        if is_reference:
            search_roots.append(self.ref_images_dir)
        else:
            if self.sorted_dir:
                found_in_sorted = next(self.sorted_dir.rglob(filename), None)
                if found_in_sorted:
                    search_roots.append(found_in_sorted.parent)
            search_roots.append(self.target_images_dir)

        final_path = next(
            (root / filename for root in search_roots if (root / filename).exists()),
            (self.ref_images_dir if is_reference else self.target_images_dir) / filename,
        )
        try:
            return Path(os.path.relpath(final_path, self.target_dir)).as_posix()
        except ValueError:
            return final_path.as_posix()

    def _extract_face_info(self, filename: str, face: dict[str, Any]) -> dict[str, Any]:
        """Извлекает из портретного лица только данные, нужные шаблону."""

        keypoints = face.get("keypoint_analysis", {})
        if not isinstance(keypoints, dict):
            keypoints = {}
        eye_states = keypoints.get("eye_states", {})
        if not isinstance(eye_states, dict):
            eye_states = {}
        left_eye = eye_states.get("left")
        right_eye = eye_states.get("right")
        eyes = f"L:{left_eye}/R:{right_eye}" if left_eye and right_eye else "N/A"
        beauty = face.get("beauty_faceonnx")

        return {
            "filename": filename,
            "rel_path": self._get_rel_path_to_file(filename, is_reference=True),
            "det_score": f"{face.get('det_score', 0.0):.2f}",
            "gender_onnx": face.get("gender_faceonnx"),
            "age_onnx": face.get("age_faceonnx"),
            "emotion_onnx": face.get("emotion_faceonnx"),
            "beauty_onnx": f"{beauty:.2f}" if beauty is not None else "N/A",
            "eye_state_combined": eyes,
        }

    def _prepare_data(self) -> dict[str, Any]:
        logger.info("<br>Подготовка данных...")
        ref_data = load_required_json(self.ref_dir / "info_faces.json")
        target_data = load_required_json(self.target_dir / "info_faces.json")
        matches_data = load_optional_json(
            self.target_dir / "matches_portrait_to_group.json"
        )

        portrait_clusters, used_students = prepare_portrait_clusters(
            ref_data, self.roster, self._extract_face_info
        )
        matches = prepare_matches(
            matches_data,
            target_data,
            portrait_clusters,
            self.roster,
            lambda filename: self._get_rel_path_to_file(filename, is_reference=False),
        )

        total_portraits = sum(
            1
            for photo in ref_data.values()
            if isinstance(photo, dict) and photo.get("face_count") == 1
        )
        total_groups = sum(
            1
            for photo in target_data.values()
            if isinstance(photo, dict) and photo.get("face_count") != 1
        )
        summary = {
            "total_portraits": total_portraits,
            "total_group_photos": total_groups,
            "total_clusters": len(portrait_clusters)
            - (1 if "-1" in portrait_clusters else 0),
            "noise_count": len(portrait_clusters.get("-1", {}).get("files", [])),
            "total_matches": len(matches),
            "identified_students": len(used_students),
            "list_id": self.roster.list_id,
            "unused_students": len(self.roster.students) - len(used_students),
            "report_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": str(self.target_dir),
            "images_path": str(self.target_images_dir),
        }
        return {
            "summary": summary,
            "portrait_clusters": portrait_clusters,
            "matches": matches,
            "thumbnail_size": 150,
            "is_cross_session": self.target_dir != self.ref_dir,
        }

    def _copy_assets(self) -> None:
        logger.info("Копирование ресурсов (css, js)...")
        for asset in ("report_style.css", "report_script.js", "lazyload.min.js"):
            source = self.templates_dir / asset
            if not source.is_file():
                raise FileNotFoundError(f"Ресурс отчёта не найден: {source}")
            shutil.copy2(source, self.target_dir / asset)

    def _atomic_write_report(self, html_content: str) -> Path:
        """Не оставляет частично записанный HTML при ошибке записи."""

        report_path = self.target_dir / "face_clustering_report.html"
        descriptor, temp_name = tempfile.mkstemp(
            dir=self.target_dir,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(html_content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, report_path)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temp_path.unlink(missing_ok=True)
            raise
        return report_path

    def run(self) -> Path:
        context = self._prepare_data()
        environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        html_content = environment.get_template("report_template.html").render(context)
        report_path = self._atomic_write_report(html_content)
        self._copy_assets()
        logger.info(f"HTML-отчёт успешно сгенерирован: {report_path.name}")
        return report_path


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация HTML-отчёта.")
    prefix = "a_hr_"
    parser.add_argument(
        f"--{prefix}target_dir",
        type=str,
        required=True,
        help="Папка текущей сессии (Output/Analysis_...).",
    )
    parser.add_argument(
        f"--{prefix}ref_dir",
        type=str,
        default=None,
        help="Папка эталона. Если не задана, используется target_dir.",
    )
    parser.add_argument(
        f"--{prefix}student_list_file",
        type=str,
        required=True,
        help="Файл *.list — единственный источник ФИО учеников.",
    )
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()


def main() -> int:
    logger.info("<b>Генерация HTML-отчёта (student_id)</b>")
    try:
        config = get_config()
        target_dir = Path(config.a_hr_target_dir)
        ref_dir = Path(config.a_hr_ref_dir) if config.a_hr_ref_dir else target_dir
        list_path = Path(config.a_hr_student_list_file)

        if not target_dir.is_dir():
            raise FileNotFoundError(f"Target dir не найден: {target_dir}")
        if not ref_dir.is_dir():
            raise FileNotFoundError(f"Reference dir не найден: {ref_dir}")

        logger.info(
            "Режим: <b>Кросс-сессия</b>"
            if target_dir != ref_dir
            else "Режим: <b>Одиночная сессия</b>"
        )
        roster = load_student_roster(list_path)
        report_path = ReportGenerator(target_dir, ref_dir, roster).run()
        if IS_MANAGED_RUN:
            pysm_context.log_link(
                url_or_path=str(report_path), text="<br>Открыть HTML-отчёт"
            )
        return 0
    except Exception as exc:
        logger.critical(f"Ошибка генерации отчёта: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
