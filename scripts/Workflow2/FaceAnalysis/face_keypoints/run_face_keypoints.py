# analize/analyze_keypoints/run_analyze_keypoints.py

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import toml
from pydantic import BaseModel, Field

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver = None
    pysm_context = None
    tqdm = lambda x, **kwargs: x

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Pydantic Модели и ConfigManager ---
class HeadPoseThresholds(BaseModel):
    yaw_thresholds: List[float] = Field(default_factory=lambda: [-30.0, -15.0, -5.0, 5.0, 15.0, 30.0])
    pitch_thresholds: List[float] = Field(default_factory=lambda: [-25.0, -10.0, -5.0, 5.0, 10.0, 25.0])
    roll_thresholds: List[float] = Field(default_factory=lambda: [-25.0, -10.0, -5.0, 5.0, 10.0, 25.0])

class AppConfig(BaseModel):
    eye_2d_ratio_thresholds: List[float] = Field(default_factory=lambda: [0.22, 0.264, 0.3])
    eye_z_diff_threshold: float = 0.05
    mouth_2d_ratio_thresholds: List[float] = Field(default_factory=lambda: [0.04, 0.1, 0.22])
    mouth_z_diff_thresholds: List[float] = Field(default_factory=lambda: [0.02, 0.04, 0.06])
    head_pose_thresholds: HeadPoseThresholds = Field(default_factory=HeadPoseThresholds)

class ConfigManager:
    def __init__(self, config_path: Path):
        try:
            config_data = toml.load(config_path)
            self.settings = AppConfig(**config_data)
        except Exception as e:
            logger.warning(f"Ошибка загрузки конфига {config_path}: {e}. Используются значения по умолчанию.")
            self.settings = AppConfig()

# --- KeypointAnalyzer ---
class KeypointAnalyzer:
    def __init__(self, config_manager: ConfigManager):
        self.settings = config_manager.settings

    def analyze_face(self, face_data: Dict) -> Optional[Dict]:
        kps_2d = face_data.get("landmark_2d_106")
        kps_3d = face_data.get("landmark_3d_68")
        pose = face_data.get("pose")

        if not kps_2d or len(kps_2d) < 106:
            return None
        
        kps_2d_np = np.array(kps_2d, dtype=np.float32)
        kps_3d_np = np.array(kps_3d, dtype=np.float32) if kps_3d else None

        # Нормализация по ширине лица
        try:
            face_width = np.linalg.norm(kps_2d_np[9] - kps_2d_np[25])
            if face_width < 1e-6: face_width = 100.0
        except IndexError:
            face_width = 100.0

        eye_left_indices = [35, 39, 42, 41, 37, 36]
        eye_right_indices = [93, 89, 95, 96, 90, 91]
        
        eye_3d_left_slice = kps_3d_np[36:42] if kps_3d_np is not None and len(kps_3d_np) >= 42 else None
        eye_3d_right_slice = kps_3d_np[42:48] if kps_3d_np is not None and len(kps_3d_np) >= 48 else None
        
        eye_left_state = self._get_eye_state(kps_2d_np, eye_left_indices, face_width, eye_3d_left_slice)
        eye_right_state = self._get_eye_state(kps_2d_np, eye_right_indices, face_width, eye_3d_right_slice)

        mouth_3d_slice = kps_3d_np[48:68] if kps_3d_np is not None and len(kps_3d_np) >= 68 else None
        mouth_state = self._get_mouth_state(kps_2d_np, mouth_3d_slice)

        head_pose_state = self._get_head_pose(pose)

        return {
            "eye_states": {"left": eye_left_state, "right": eye_right_state},
            "mouth_state": mouth_state,
            "head_pose": head_pose_state,
        }

    def _get_eye_state(self, kps2d, indices, face_width, kps3d=None):
        result = {"state": "unknown", "ear": None, "normalized_ear": None, "z_diff": None}
        try:
            points = kps2d[indices]
            width = np.linalg.norm(points[0] - points[1])
            if width < 1e-6: return result
            
            h1 = np.linalg.norm(points[2] - points[4])
            h2 = np.linalg.norm(points[3] - points[5])
            ear = (h1 + h2) / (2.0 * width)
            result["ear"] = float(ear)
            result["normalized_ear"] = float(ear * (100.0 / face_width)) if face_width > 0 else float(ear)

            thresholds = self.settings.eye_2d_ratio_thresholds
            if ear < thresholds[0]: result["state"] = "Closed"
            elif ear < thresholds[1]: result["state"] = "Squinting"
            elif ear < thresholds[2]: result["state"] = "Open"
            else: result["state"] = "Wide Open"

            if kps3d is not None and len(kps3d) >= 6:
                z_diff = abs(kps3d[1][2] - kps3d[5][2])
                result["z_diff"] = float(z_diff)
                if z_diff > self.settings.eye_z_diff_threshold and result["state"] == "Closed":
                     result["state"] = "Blinking"

        except Exception as e:
            result["state"] = "error"
        return result

    def _get_mouth_state(self, kps2d, kps3d=None):
        result = {"state": "unknown", "mar": None, "z_diff": None}
        indices = [52, 61, 66, 54, 62, 60, 70, 57]
        try:
            points = kps2d[indices]
            width = np.linalg.norm(points[0] - points[1])
            if width < 1e-6:
                result["state"] = "closed"
                return result

            h1 = np.linalg.norm(points[2] - points[3])
            h2 = np.linalg.norm(points[4] - points[5])
            h3 = np.linalg.norm(points[6] - points[7])
            mar = (h1 + h2 + h3) / (3.0 * width)
            result["mar"] = float(mar)

            thresholds = self.settings.mouth_2d_ratio_thresholds
            if mar < thresholds[0]: result["state"] = "closed"
            elif mar < thresholds[1]: result["state"] = "slightly_open"
            elif mar < thresholds[2]: result["state"] = "open"
            else: result["state"] = "wide_open"

            if kps3d is not None and len(kps3d) >= 20:
                z_diff = abs(kps3d[6][2] - kps3d[14][2])
                result["z_diff"] = float(z_diff)
        
        except Exception as e:
            result["state"] = "error"
        return result

    def _get_head_pose(self, pose: Optional[List[float]]) -> Dict:
        result = {"state": "unknown", "yaw": None, "pitch": None, "roll": None}
        if not pose or len(pose) < 3: return result
        
        yaw, pitch, roll = pose
        result.update({"yaw": yaw, "pitch": pitch, "roll": roll})

        yaw_state = self._classify_angle(yaw, self.settings.head_pose_thresholds.yaw_thresholds,
            ["extreme_left", "left", "slight_left", "frontal", "slight_right", "right", "extreme_right"])
        pitch_state = self._classify_angle(pitch, self.settings.head_pose_thresholds.pitch_thresholds,
            ["extreme_down", "down", "slight_down", "frontal", "slight_up", "up", "extreme_up"])
        roll_state = self._classify_angle(roll, self.settings.head_pose_thresholds.roll_thresholds,
            ["extreme_left_roll", "left_roll", "slight_left_roll", "level", "slight_right_roll", "right_roll", "extreme_right_roll"])

        result["state"] = f"yaw: {yaw_state}, pitch: {pitch_state}, roll: {roll_state}"
        return result

    def _classify_angle(self, angle: float, thresholds: List[float], labels: List[str]) -> str:
        if len(thresholds) != 6: return "invalid_thresholds"
        if angle < thresholds[0]: return labels[0]
        if angle < thresholds[1]: return labels[1]
        if angle < thresholds[2]: return labels[2]
        if angle <= thresholds[3]: return labels[3]
        if angle <= thresholds[4]: return labels[4]
        if angle <= thresholds[5]: return labels[5]
        return labels[6]

    def _compute_overall_statistics(self, metrics: List[Dict]) -> Dict:
        """Считает статистику по всему набору обработанных лиц."""
        def compute_stats(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
            valid = [v for v in values if v is not None and np.isfinite(v)]
            if not valid:
                return {k: None for k in ["mean", "median", "min", "max", "q1", "q3"]}
            return {
                "mean": float(np.mean(valid)), "median": float(np.median(valid)),
                "min": float(np.min(valid)), "max": float(np.max(valid)),
                "q1": float(np.percentile(valid, 25)), "q3": float(np.percentile(valid, 75))
            }

        stats: Dict[str, Any] = {"eyes": {"left": {}, "right": {}}, "mouth": {}, "head_pose": {}}
        
        for side in ["left", "right"]:
            stats["eyes"][side]["ear"] = compute_stats([m["eye_states"][side]["ear"] for m in metrics])
            stats["eyes"][side]["normalized_ear"] = compute_stats([m["eye_states"][side]["normalized_ear"] for m in metrics])
            stats["eyes"][side]["z_diff"] = compute_stats([m["eye_states"][side]["z_diff"] for m in metrics])
        
        stats["mouth"]["mar"] = compute_stats([m["mouth_state"]["mar"] for m in metrics])
        stats["mouth"]["z_diff"] = compute_stats([m["mouth_state"]["z_diff"] for m in metrics])
        
        for axis in ["yaw", "pitch", "roll"]:
            stats["head_pose"][axis] = compute_stats([m["head_pose"][axis] for m in metrics])
        
        return stats
    
    # --- ПОЛНОСТЬЮ ВОССТАНОВЛЕННЫЙ МЕТОД ---
    def _generate_recommendations(self, stats: Dict, data_dir: Path, filename: str):
        """Генерирует файл с рекомендациями по настройке порогов (в полном формате)."""
        recs = [
            "# Рекомендации по настройке порогов в config.toml",
            "# Эти значения основаны на статистическом анализе (медиана, квантили)",
            "# обработанных изображений. Вы можете скопировать предложенные строки",
            "# в ваш config.toml для тонкой настройки.\n"
        ]

        recs.append("=" * 20 + " Глаза " + "=" * 20)
        
        def avg_stat(key):
            left_stats = stats.get("eyes", {}).get("left", {}).get("normalized_ear", {})
            right_stats = stats.get("eyes", {}).get("right", {}).get("normalized_ear", {})
            vals = [s.get(key) for s in [left_stats, right_stats] if s and s.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        ear_q1, ear_median, ear_q3 = avg_stat("q1"), avg_stat("median"), avg_stat("q3")
        
        recs.append("\n# --- eye_2d_ratio_thresholds (на основе нормализованного EAR) ---")
        recs.append("# [порог 'Closed', порог 'Squinting', порог 'Open']")
        current_thresholds = self.settings.eye_2d_ratio_thresholds
        recs.append(f"# Текущие значения: {current_thresholds}")
        
        if all(v is not None for v in [ear_q1, ear_median, ear_q3]):
            recs.append(f"# Статистика (Q1, Median, Q3): [{ear_q1:.3f}, {ear_median:.3f}, {ear_q3:.3f}]")
            recs.append(f"eye_2d_ratio_thresholds = [{ear_q1:.3f}, {ear_median:.3f}, {ear_q3:.3f}]")
        else:
            recs.append("# Недостаточно данных для рекомендации.")

        recs.append("\n" + "=" * 20 + " Рот " + "=" * 20)
        recs.append("\n# --- mouth_2d_ratio_thresholds ---")
        
        mouth_stats = stats.get("mouth", {}).get("mar", {})
        mar_q1, mar_median, mar_q3 = mouth_stats.get("q1"), mouth_stats.get("median"), mouth_stats.get("q3")
        current_thresholds = self.settings.mouth_2d_ratio_thresholds
        recs.append(f"# Текущие значения: {current_thresholds}")
        
        if all(v is not None for v in [mar_q1, mar_median, mar_q3]):
            recs.append(f"# Статистика (Q1, Median, Q3): [{mar_q1:.3f}, {mar_median:.3f}, {mar_q3:.3f}]")
            recs.append(f"mouth_2d_ratio_thresholds = [{mar_q1:.3f}, {mar_median:.3f}, {mar_q3:.3f}]")
        else:
            recs.append("# Недостаточно данных для рекомендации.")
            
        recs.append("\n" + "=" * 20 + " Поза головы (статистика для информации) " + "=" * 20)
        
        yaw_stats = stats.get("head_pose", {}).get("yaw", {})
        pitch_stats = stats.get("head_pose", {}).get("pitch", {})
        roll_stats = stats.get("head_pose", {}).get("roll", {})
        
        if yaw_stats.get("median") is not None:
             recs.append(f"# Yaw (горизонталь): Min={yaw_stats.get('min'):.2f}, Median={yaw_stats.get('median'):.2f}, Max={yaw_stats.get('max'):.2f}")
        if pitch_stats.get("median") is not None:
             recs.append(f"# Pitch (вертикаль): Min={pitch_stats.get('min'):.2f}, Median={pitch_stats.get('median'):.2f}, Max={pitch_stats.get('max'):.2f}")
        if roll_stats.get("median") is not None:
             recs.append(f"# Roll (наклон): Min={roll_stats.get('min'):.2f}, Median={roll_stats.get('median'):.2f}, Max={roll_stats.get('max'):.2f}")
        
        rec_path = data_dir / filename
        try:
            rec_path.write_text("\n".join(recs), encoding="utf-8")
            if IS_MANAGED_RUN and pysm_context:
                pysm_context.log_link(str(rec_path), f"Открыть {filename}")
        except Exception as e:
            logger.error(f"Не удалось сохранить рекомендации: {e}")


def get_cli_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ ключевых точек лиц.")
    default_config_path = Path(__file__).parent / "config.toml"
    
    parser.add_argument(
        "--a_ak_data_dir", 
        type=str, 
        required=True, 
        help="Путь к папке с результатами анализа (содержащей info_faces.json)."
    )
    parser.add_argument("--a_ak_config_file", default=str(default_config_path))
    
    return ConfigResolver(parser).resolve_all() if IS_MANAGED_RUN else parser.parse_args()

def main():
    logger.info("<b>Анализ ключевых точек лица</b>")
    cli_config = get_cli_config()
    
    data_dir = Path(cli_config.a_ak_data_dir)
    config_path = Path(cli_config.a_ak_config_file)
    
    if not data_dir.is_dir():
        logger.critical(f"Папка с данными не найдена: {data_dir}")
        sys.exit(1)
        
    config_manager = ConfigManager(config_path)
    analyzer = KeypointAnalyzer(config_manager)
    
    # 1. Загрузка данных (Unified Storage)
    faces_json_path = data_dir / "info_faces.json"
    landmarks_json_path = data_dir / "info_faces_landmarks.json"
    
    if not faces_json_path.exists():
        logger.critical(f"Файл {faces_json_path.name} не найден.")
        sys.exit(1)

    try:
        with open(faces_json_path, 'r', encoding='utf-8') as f:
            faces_data = json.load(f)
        
        landmarks_data = {}
        if landmarks_json_path.exists():
            with open(landmarks_json_path, 'r', encoding='utf-8') as f:
                landmarks_data = json.load(f)
            logger.info(f"Загружены ландмарки для {len(landmarks_data)} файлов.")
        else:
            logger.warning("Файл ландмарков не найден. Анализ будет неполным (только по базовым точкам, если есть).")

    except Exception as e:
        logger.critical(f"Ошибка чтения JSON: {e}")
        sys.exit(1)

    # 2. Анализ
    detailed_metrics = []
    updates_count = 0
    
    all_filenames = list(faces_data.keys())
    for filename in tqdm(all_filenames, desc="Анализ лиц"):
        file_entry = faces_data[filename]
        land_entry = landmarks_data.get(filename, {})
        
        main_faces = file_entry.get("faces", [])
        land_faces = land_entry.get("faces", [])
        
        for i, face in enumerate(main_faces):
            # Слияние с ландмарками
            face_merged = face.copy()
            if i < len(land_faces):
                face_merged.update(land_faces[i])
            
            # Анализ
            results = analyzer.analyze_face(face_merged)
            
            if results:
                # Обновление данных в памяти
                face["keypoint_analysis"] = {
                    "eye_states": {
                        "left": results["eye_states"]["left"]["state"],
                        "right": results["eye_states"]["right"]["state"],
                    },
                    "mouth_state": results["mouth_state"]["state"],
                    "head_pose": results["head_pose"]["state"],
                }
                detailed_metrics.append(results)
                updates_count += 1

    # 3. Сохранение
    logger.info(f"Обновлено записей: {updates_count}")
    try:
        with open(faces_json_path, 'w', encoding='utf-8') as f:
            json.dump(faces_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {faces_json_path.name}")
    except Exception as e:
        logger.critical(f"Ошибка сохранения JSON: {e}")
        sys.exit(1)

    # 4. Статистика
    if detailed_metrics:
        stats = analyzer._compute_overall_statistics(detailed_metrics)
        try:
            stats_path = data_dir / "statistics_keypoints.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            analyzer._generate_recommendations(stats, data_dir, "recommendations_keypoints.txt")
        except Exception as e:
            logger.error(f"Ошибка сохранения статистики: {e}")

    logger.info("\n")

if __name__ == "__main__":
    main()