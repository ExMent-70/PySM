# fc_lib/fc_keypoints.py
# --- ИЗМЕНЕНИЙ НЕ ТРЕБУЕТСЯ ---

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional  # Добавили Optional
from tqdm import tqdm
import psutil
import numpy as np

# --- Обновленные импорты ---
from .fc_config import ConfigManager
from .fc_messages import get_message
from .fc_json_data_manager import JsonDataManager

# --- Конец обновленных импортов ---

logger = logging.getLogger(__name__)


# Новая функция-обертка
def run_keypoint_analysis(config: ConfigManager, json_manager: JsonDataManager) -> None:
    """
    Запускает анализ ключевых точек, если он включен в конфигурации.
    """
    if not config.get("task", "keypoint_analysis", default=False):
        logger.info(
            "Анализ ключевых точек отключен в конфигурации ('task.keypoint_analysis' = false)."
        )
        return

    print(f"")
    print("<b>Запуск анализа ключевых точек</b>")
    analyzer = KeypointAnalyzer(config)
    output_path = Path(config.get("paths", "output_path"))

    # Загружаем данные, если они еще не загружены (на всякий случай)
    if not json_manager.portrait_data and not json_manager.group_data:
        logger.info("Загрузка JSON данных для анализа ключевых точек...")
        if not json_manager.load_data():
            logger.error(
                "Не удалось загрузить JSON данные. Анализ ключевых точек отменен."
            )
            return

    # Анализируем оба типа данных, если они есть
    if json_manager.portrait_data:
        logger.info(
            f"Анализ ключевых точек для портретных данных ({json_manager.portrait_json_path.name})"
        )
        analyzer.analyze_and_update_json(
            json_manager=json_manager, data_type="portrait", output_path=output_path
        )
    else:
        logger.info("Нет портретных данных для анализа ключевых точек.")

    if json_manager.group_data:
        logger.info(
            f"Анализ ключевых точек для групповых данных ({json_manager.group_json_path.name})"
        )
        analyzer.analyze_and_update_json(
            json_manager=json_manager, data_type="group", output_path=output_path
        )
    else:
        logger.info("Нет групповых данных для анализа ключевых точек.")

    logger.debug("=" * 10 + " Анализ ключевых точек завершен " + "=" * 10)


class KeypointAnalyzer:
    """Класс для анализа ключевых точек лиц."""

    def __init__(self, config: ConfigManager):
        """Инициализирует анализатор ключевых точек."""
        self.config = config
        report_config = config.get("report", {})
        # Настройки анализа берем из report.keypoint_analysis
        keypoint_config = report_config.get("keypoint_analysis", {})
        # Если секции keypoint_analysis нет, используем пустой словарь
        if keypoint_config is None:
            keypoint_config = {}

        self.eye_2d_thresholds = keypoint_config.get(
            "eye_2d_ratio_thresholds", [0.15, 0.25, 0.35]
        )
        self.eye_z_threshold = keypoint_config.get("eye_z_diff_threshold", 0.1)
        self.mouth_2d_thresholds = keypoint_config.get(
            "mouth_2d_ratio_thresholds", [0.2, 0.5, 0.7]
        )
        self.mouth_z_thresholds = keypoint_config.get(
            "mouth_z_diff_thresholds", [1.5, 2.5, 3.5]
        )

        head_pose_thresholds_dict = keypoint_config.get("head_pose_thresholds", {})
        # Убедимся, что head_pose_thresholds_dict действительно словарь
        if head_pose_thresholds_dict is None:
            head_pose_thresholds_dict = {}
        default_head_pose = {
            "yaw_thresholds": [-30.0, -15.0, -5.0, 5.0, 15.0, 30.0],
            "pitch_thresholds": [-25.0, -10.0, -5.0, 5.0, 10.0, 25.0],
            "roll_thresholds": [-25.0, -10.0, -5.0, 5.0, 10.0, 25.0],
        }
        self.head_pose_thresholds = {
            "yaw_thresholds": head_pose_thresholds_dict.get(
                "yaw_thresholds", default_head_pose["yaw_thresholds"]
            ),
            "pitch_thresholds": head_pose_thresholds_dict.get(
                "pitch_thresholds", default_head_pose["pitch_thresholds"]
            ),
            "roll_thresholds": head_pose_thresholds_dict.get(
                "roll_thresholds", default_head_pose["roll_thresholds"]
            ),
        }
        logger.debug(
            f"KeypointAnalyzer инициализирован с порогами: Глаза={self.eye_2d_thresholds}, Рот={self.mouth_2d_thresholds}, Голова={self.head_pose_thresholds}"
        )

    # --- Методы анализа (_analyze_keypoints, _get_eye_state, etc.) без изменений ---
    def _analyze_keypoints(
        self,
        keypoints_2d: List[List[float]],
        keypoints_3d: Optional[List[List[float]]] = None,
        pose: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        eye_states = {"left": {"state": "unknown"}, "right": {"state": "unknown"}}
        mouth_state = {"state": "unknown"}
        head_pose = {"state": "unknown"}
        try:
            if keypoints_2d and len(keypoints_2d) >= 106:
                left_eye_indices = {
                    "outer_corner": 35,
                    "inner_corner": 39,
                    "upper_lid_1": 42,
                    "upper_lid_2": 41,
                    "lower_lid_1": 37,
                    "lower_lid_2": 36,
                }
                right_eye_indices = {
                    "outer_corner": 93,
                    "inner_corner": 89,
                    "upper_lid_1": 95,
                    "upper_lid_2": 96,
                    "lower_lid_1": 90,
                    "lower_lid_2": 91,
                }
                face_width = (
                    abs(keypoints_2d[9][0] - keypoints_2d[25][0])
                    if len(keypoints_2d[9]) >= 1 and len(keypoints_2d[25]) >= 1
                    else None
                )
                if face_width is None or face_width < 1:
                    face_width = 100.0  # Fallback width
                eye_3d_left = (
                    keypoints_3d[36:42]
                    if keypoints_3d and len(keypoints_3d) >= 42
                    else None
                )
                eye_3d_right = (
                    keypoints_3d[42:48]
                    if keypoints_3d and len(keypoints_3d) >= 48
                    else None
                )
                eye_states["left"] = self._get_eye_state(
                    keypoints_2d, left_eye_indices, eye_3d_left, face_width
                )
                eye_states["right"] = self._get_eye_state(
                    keypoints_2d, right_eye_indices, eye_3d_right, face_width
                )
                mouth_3d = (
                    keypoints_3d[48:68]
                    if keypoints_3d and len(keypoints_3d) >= 68
                    else None
                )
                mouth_state = self._get_mouth_state(keypoints_2d, mouth_3d)
            if pose:
                head_pose = self._get_head_pose(pose)
            elif keypoints_3d and len(keypoints_3d) >= 68:
                head_pose = self._get_head_pose_from_3d(keypoints_3d)
        except Exception as e:
            logger.error(get_message("ERROR_KEYPOINT_ANALYSIS_FAILED", exc=e))
        return {
            "eye_states": eye_states,
            "mouth_state": mouth_state,
            "head_pose": head_pose,
        }

    def _get_eye_state(
        self,
        keypoints_2d: List[List[float]],
        eye_indices: Dict[str, int],
        eye_points_3d: Optional[List[List[float]]] = None,
        face_width: float = 100.0,
    ) -> Dict[str, Any]:
        result = {
            "state": "unknown",
            "ear": None,
            "normalized_ear": None,
            "z_diff": None,
        }
        required_indices = list(eye_indices.values())
        if not all(
            0 <= idx < len(keypoints_2d) and len(keypoints_2d[idx]) >= 2
            for idx in required_indices
        ):
            logger.debug(
                f"Некорректные или отсутствующие точки глаза: {required_indices}"
            )
            return result
        try:
            width = abs(
                keypoints_2d[eye_indices["outer_corner"]][0]
                - keypoints_2d[eye_indices["inner_corner"]][0]
            )
            if width < 1:
                ear = 0.0
            else:
                height1 = abs(
                    keypoints_2d[eye_indices["upper_lid_1"]][1]
                    - keypoints_2d[eye_indices["lower_lid_1"]][1]
                )
                height2 = abs(
                    keypoints_2d[eye_indices["upper_lid_2"]][1]
                    - keypoints_2d[eye_indices["lower_lid_2"]][1]
                )
                ear = (height1 + height2) / (2.0 * width)
            if not (0 <= ear <= 1):
                ear = 0.0  # Ограничиваем аномальные значения
            result["ear"] = ear
            result["normalized_ear"] = (
                ear * (100.0 / face_width) if face_width > 50 else ear
            )  # Нормализация для статистики
            z_diff = None
            if (
                eye_points_3d
                and len(eye_points_3d) >= 6
                and all(len(p) >= 3 for p in eye_points_3d)
            ):
                z_diff = abs(eye_points_3d[1][2] - eye_points_3d[5][2])
                result["z_diff"] = z_diff
            logger.debug(
                get_message(
                    "DEBUG_EYE_STATE_ANALYSIS",
                    ear=ear,
                    normalized_ear=result["normalized_ear"],
                    z_diff=z_diff,
                )
            )
            logger.debug(
                f"Eye thresholds: {self.eye_2d_thresholds}, z_diff threshold: {self.eye_z_threshold}"
            )
            if ear < self.eye_2d_thresholds[0]:
                result["state"] = "Closed"  # Изменено на Closed/Open
            elif ear < self.eye_2d_thresholds[1]:
                result["state"] = "Squinting"  # Прищурен
            elif ear < self.eye_2d_thresholds[2]:
                result["state"] = "Open"
            else:
                result["state"] = "Wide Open"
        except Exception as e:
            logger.error(f"Ошибка в _get_eye_state: {e}")
            result["state"] = "error"
        return result

    def _get_mouth_state(
        self,
        keypoints_2d: List[List[float]],
        mouth_points_3d: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        result = {"state": "unknown", "mar": None, "z_diff": None}
        mouth_indices = {
            "left_corner": 52,
            "right_corner": 61,
            "upper_lip_inner_1": 66,
            "upper_lip_inner_2": 62,
            "upper_lip_inner_3": 70,
            "lower_lip_inner_1": 54,
            "lower_lip_inner_2": 60,
            "lower_lip_inner_3": 57,
        }
        required_indices = list(mouth_indices.values())
        if not all(
            0 <= idx < len(keypoints_2d) and len(keypoints_2d[idx]) >= 2
            for idx in required_indices
        ):
            logger.debug(
                f"Некорректные или отсутствующие точки рта: {required_indices}"
            )
            return result
        try:
            width = abs(
                keypoints_2d[mouth_indices["left_corner"]][0]
                - keypoints_2d[mouth_indices["right_corner"]][0]
            )
            if width < 1:
                mar = 0.0
            else:
                h1 = abs(
                    keypoints_2d[mouth_indices["upper_lip_inner_1"]][1]
                    - keypoints_2d[mouth_indices["lower_lip_inner_1"]][1]
                )
                h2 = abs(
                    keypoints_2d[mouth_indices["upper_lip_inner_2"]][1]
                    - keypoints_2d[mouth_indices["lower_lip_inner_2"]][1]
                )
                h3 = abs(
                    keypoints_2d[mouth_indices["upper_lip_inner_3"]][1]
                    - keypoints_2d[mouth_indices["lower_lip_inner_3"]][1]
                )
                mar = (h1 + h2 + h3) / (3.0 * width)
            result["mar"] = mar
            z_diff = None
            if (
                mouth_points_3d
                and len(mouth_points_3d) >= 20
                and all(len(p) >= 3 for p in mouth_points_3d)
            ):
                z_diff = abs(mouth_points_3d[6][2] - mouth_points_3d[14][2])
                result["z_diff"] = z_diff
            logger.debug(f"MAR: {mar}, Z-diff: {z_diff}")
            logger.debug(
                f"Mouth thresholds: {self.mouth_2d_thresholds}, z_diff thresholds: {self.mouth_z_thresholds}"
            )
            if mar < self.mouth_2d_thresholds[0]:
                result["state"] = "closed"
            elif mar < self.mouth_2d_thresholds[1]:
                result["state"] = "slightly_open"
            elif mar < self.mouth_2d_thresholds[2]:
                result["state"] = "open"
            else:
                result["state"] = "wide_open"
            logger.debug(f"Mouth state: {result['state']} (MAR: {mar})")
        except Exception as e:
            logger.error(f"Ошибка в _get_mouth_state: {e}")
            result["state"] = "error"
        return result

    def _get_head_pose(self, pose: List[float]) -> Dict[str, Any]:
        result = {"state": "unknown", "yaw": None, "pitch": None, "roll": None}
        if not pose or len(pose) < 3:
            logger.warning(get_message("WARNING_NO_POSE_DATA"))
            return result
        try:
            yaw, pitch, roll = pose
            result["yaw"] = yaw
            result["pitch"] = pitch
            result["roll"] = roll
            yaw_state = self._classify_angle(
                yaw, self.head_pose_thresholds["yaw_thresholds"], "yaw"
            )
            pitch_state = self._classify_angle(
                pitch, self.head_pose_thresholds["pitch_thresholds"], "pitch"
            )
            roll_state = self._classify_angle(
                roll, self.head_pose_thresholds["roll_thresholds"], "roll"
            )
            state = f"yaw: {yaw_state}, pitch: {pitch_state}, roll: {roll_state}"
            result["state"] = state
            logger.debug(
                get_message(
                    "DEBUG_HEAD_POSE_ANALYSIS",
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    result=state,
                )
            )
        except Exception as e:
            logger.error(f"Ошибка в _get_head_pose: {e}")
            result["state"] = "error"
        return result

    def _get_head_pose_from_3d(self, keypoints_3d: List[List[float]]) -> Dict[str, Any]:
        result = {"state": "unknown", "yaw": None, "pitch": None, "roll": None}
        if len(keypoints_3d) < 68 or not all(
            len(p) >= 3 for p in [keypoints_3d[0], keypoints_3d[16], keypoints_3d[30]]
        ):
            return result
        try:
            nose_tip = keypoints_3d[30]
            yaw = nose_tip[0]
            pitch = nose_tip[1]
            roll = keypoints_3d[0][2] - keypoints_3d[16][2]
            result["yaw"] = yaw
            result["pitch"] = pitch
            result["roll"] = roll
            yaw_state = self._classify_angle(
                yaw, self.head_pose_thresholds["yaw_thresholds"], "yaw"
            )
            pitch_state = self._classify_angle(
                pitch, self.head_pose_thresholds["pitch_thresholds"], "pitch"
            )
            roll_state = self._classify_angle(
                roll, self.head_pose_thresholds["roll_thresholds"], "roll"
            )
            state = f"yaw: {yaw_state}, pitch: {pitch_state}, roll: {roll_state}"
            result["state"] = state
        except Exception as e:
            logger.error(f"Ошибка в _get_head_pose_from_3d: {e}")
            result["state"] = "error"
        return result

    def _classify_angle(
        self, angle: float, thresholds: List[float], axis_name: str
    ) -> str:
        labels_map = {
            "yaw": (
                "extreme_left",
                "left",
                "slight_left",
                "frontal",
                "slight_right",
                "right",
                "extreme_right",
            ),
            "pitch": (
                "extreme_down",
                "down",
                "slight_down",
                "frontal",
                "slight_up",
                "up",
                "extreme_up",
            ),
            "roll": (
                "extreme_left_roll",
                "left_roll",
                "slight_left_roll",
                "level",
                "slight_right_roll",
                "right_roll",
                "extreme_right_roll",
            ),
        }
        labels = labels_map.get(
            axis_name, labels_map["yaw"]
        )  # Default to yaw labels if axis unknown
        if len(thresholds) != 6:
            return "invalid_thresholds"
        if angle < thresholds[0]:
            return labels[0]
        elif angle < thresholds[1]:
            return labels[1]
        elif angle < thresholds[2]:
            return labels[2]
        elif angle <= thresholds[3]:
            return labels[3]
        elif angle <= thresholds[4]:
            return labels[4]
        elif angle <= thresholds[5]:
            return labels[5]
        else:
            return labels[6]

    def compute_statistics(
        self, values: List[Optional[float]]
    ) -> Dict[str, Optional[float]]:
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return {
                "mean": None,
                "median": None,
                "q1": None,
                "q3": None,
                "min": None,
                "max": None,
            }
        try:
            stats = {
                "mean": float(np.mean(valid_values)),
                "median": float(np.median(valid_values)),
                "q1": float(np.percentile(valid_values, 25)),
                "q3": float(np.percentile(valid_values, 75)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
            }
            logger.debug(f"Computed statistics for {len(valid_values)} values: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Ошибка вычисления статистики: {e}")
            return {
                "mean": None,
                "median": None,
                "q1": None,
                "q3": None,
                "min": None,
                "max": None,
            }

    def analyze_metrics(self, detailed_metrics: List[Dict]) -> Dict[str, Any]:
        analysis = {"eyes": {"left": {}, "right": {}}, "mouth": {}, "head_pose": {}}
        left_ear = [m["eyes"]["left"]["ear"] for m in detailed_metrics]
        left_normalized_ear = [
            m["eyes"]["left"]["normalized_ear"] for m in detailed_metrics
        ]
        left_z_diff = [m["eyes"]["left"]["z_diff"] for m in detailed_metrics]
        right_ear = [m["eyes"]["right"]["ear"] for m in detailed_metrics]
        right_normalized_ear = [
            m["eyes"]["right"]["normalized_ear"] for m in detailed_metrics
        ]
        right_z_diff = [m["eyes"]["right"]["z_diff"] for m in detailed_metrics]
        mar = [m["mouth"]["mar"] for m in detailed_metrics]
        mouth_z_diff = [m["mouth"]["z_diff"] for m in detailed_metrics]
        yaw = [m["head_pose"]["yaw"] for m in detailed_metrics]
        pitch = [m["head_pose"]["pitch"] for m in detailed_metrics]
        roll = [m["head_pose"]["roll"] for m in detailed_metrics]
        analysis["eyes"]["left"]["ear"] = self.compute_statistics(left_ear)
        analysis["eyes"]["left"]["normalized_ear"] = self.compute_statistics(
            left_normalized_ear
        )
        analysis["eyes"]["left"]["z_diff"] = self.compute_statistics(left_z_diff)
        analysis["eyes"]["right"]["ear"] = self.compute_statistics(right_ear)
        analysis["eyes"]["right"]["normalized_ear"] = self.compute_statistics(
            right_normalized_ear
        )
        analysis["eyes"]["right"]["z_diff"] = self.compute_statistics(right_z_diff)
        analysis["mouth"]["mar"] = self.compute_statistics(mar)
        analysis["mouth"]["z_diff"] = self.compute_statistics(mouth_z_diff)
        analysis["head_pose"]["yaw"] = self.compute_statistics(yaw)
        analysis["head_pose"]["pitch"] = self.compute_statistics(pitch)
        analysis["head_pose"]["roll"] = self.compute_statistics(roll)
        return analysis

    def generate_recommendations(
        self, analysis: Dict[str, Any], output_path: Path, json_path: Path
    ) -> None:
        recommendations = []  # Логика генерации рекомендаций (без изменений)
        # ... (код генерации рекомендаций как был) ...
        eye_recommendations = []
        for side in ["left", "right"]:
            eye_stats = analysis["eyes"][side]
            current_thresholds = self.eye_2d_thresholds
            ear_median = eye_stats.get("ear", {}).get("median")
            ear_q1 = eye_stats.get("ear", {}).get("q1")
            ear_q3 = eye_stats.get("ear", {}).get("q3")
            eye_recommendations.append(f"Глаза ({side}):")
            eye_recommendations.append(
                f"  Текущие пороги eye_2d_ratio_thresholds: {current_thresholds}"
            )
            if ear_median is not None:
                eye_recommendations.append(
                    f"  Медиана ear: {ear_median:.3f}, Q1: {ear_q1:.3f}, Q3: {ear_q3:.3f}"
                )
            else:
                eye_recommendations.append("  Статистика EAR недоступна.")
            if ear_q1 is not None and ear_q1 < current_thresholds[0]:
                eye_recommendations.append(
                    f"  - Порог 'Closed' ({current_thresholds[0]}) может быть высок. Рекомендация: ~{ear_q1:.3f}."
                )
            if ear_median is not None and ear_median < current_thresholds[1]:
                eye_recommendations.append(
                    f"  - Порог 'Squinting' ({current_thresholds[1]}) может быть высок. Рекомендация: ~{ear_median:.3f}."
                )
            if ear_q3 is not None and ear_q3 < current_thresholds[2]:
                eye_recommendations.append(
                    f"  - Порог 'Open' ({current_thresholds[2]}) может быть высок. Рекомендация: ~{ear_q3:.3f}."
                )
            z_diff_median = eye_stats.get("z_diff", {}).get("median")
            current_z_threshold = self.eye_z_threshold
            if z_diff_median is not None:
                eye_recommendations.append(
                    f"  Текущий порог eye_z_diff_threshold: {current_z_threshold}"
                )
                eye_recommendations.append(f"  Медиана z_diff: {z_diff_median:.3f}")
            if z_diff_median is not None and z_diff_median < current_z_threshold:
                eye_recommendations.append(
                    f"  - Порог z_diff ({current_z_threshold}) может быть высок. Рекомендация: ~{z_diff_median:.3f}."
                )
        recommendations.extend(eye_recommendations)
        recommendations.append("")
        mouth_stats = analysis["mouth"]
        current_mouth_thresholds = self.mouth_2d_thresholds
        mar_median = mouth_stats.get("mar", {}).get("median")
        mar_q1 = mouth_stats.get("mar", {}).get("q1")
        mar_q3 = mouth_stats.get("mar", {}).get("q3")
        recommendations.append("Рот:")
        recommendations.append(
            f"  Текущие пороги mouth_2d_ratio_thresholds: {current_mouth_thresholds}"
        )
        if mar_median is not None:
            recommendations.append(
                f"  Медиана mar: {mar_median:.3f}, Q1: {mar_q1:.3f}, Q3: {mar_q3:.3f}"
            )
        else:
            recommendations.append("  Статистика MAR недоступна.")
        if mar_q1 is not None and mar_q1 < current_mouth_thresholds[0]:
            recommendations.append(
                f"  - Порог 'closed' ({current_mouth_thresholds[0]}) может быть высок. Рекомендация: ~{mar_q1:.3f}."
            )
        if mar_median is not None and mar_median < current_mouth_thresholds[1]:
            recommendations.append(
                f"  - Порог 'slightly_open' ({current_mouth_thresholds[1]}) может быть высок. Рекомендация: ~{mar_median:.3f}."
            )
        if mar_q3 is not None and mar_q3 < current_mouth_thresholds[2]:
            recommendations.append(
                f"  - Порог 'open' ({current_mouth_thresholds[2]}) может быть высок. Рекомендация: ~{mar_q3:.3f}."
            )
        mouth_z_diff_median = mouth_stats.get("z_diff", {}).get("median")
        current_mouth_z_thresholds = self.mouth_z_thresholds
        if mouth_z_diff_median is not None:
            recommendations.append(
                f"  Текущие пороги mouth_z_diff_thresholds: {current_mouth_z_thresholds}"
            )
            recommendations.append(f"  Медиана z_diff: {mouth_z_diff_median:.3f}")
        if (
            mouth_z_diff_median is not None
            and mouth_z_diff_median < current_mouth_z_thresholds[0]
        ):
            recommendations.append(
                f"  - Порог z_diff 'closed' ({current_mouth_z_thresholds[0]}) может быть высок. Рекомендация: ~{mouth_z_diff_median:.3f}."
            )
        elif (
            mouth_z_diff_median is not None
            and mouth_z_diff_median < current_mouth_z_thresholds[1]
        ):
            recommendations.append(
                f"  - Порог z_diff 'slightly_open' ({current_mouth_z_thresholds[1]}) может быть высок. Рекомендация: ~{mouth_z_diff_median:.3f}."
            )
        recommendations.append("")
        head_pose_stats = analysis["head_pose"]
        current_head_pose_thresholds = self.head_pose_thresholds
        yaw_median = head_pose_stats.get("yaw", {}).get("median")
        pitch_median = head_pose_stats.get("pitch", {}).get("median")
        roll_median = head_pose_stats.get("roll", {}).get("median")
        recommendations.append("Поза головы:")
        recommendations.append(
            f"  Текущие пороги head_pose_thresholds: {current_head_pose_thresholds}"
        )
        if yaw_median is not None:
            recommendations.append(
                f"  Медиана yaw: {yaw_median:.3f}, pitch: {pitch_median:.3f}, roll: {roll_median:.3f}"
            )
        else:
            recommendations.append("  Статистика позы головы недоступна.")
        # ... (логика рекомендаций по позе головы) ...

        recommendations_path = output_path / f"recommendations_{json_path.stem}.txt"
        try:
            with recommendations_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(recommendations))
            logger.info(f"Рекомендации сохранены в {recommendations_path}")
        except Exception as e:
            logger.error(
                f"Ошибка сохранения файла рекомендаций {recommendations_path}: {e}"
            )

    def analyze_and_update_json(
        self, json_manager: JsonDataManager, data_type: str, output_path: Path
    ) -> None:
        """Анализирует ключевые точки из данных JsonDataManager и обновляет их."""
        json_path = (
            json_manager.portrait_json_path
            if data_type == "portrait"
            else json_manager.group_json_path
        )
        logger.debug(
            f"Начало анализа ключевых точек для {data_type} данных ({json_path})."
        )
        data_to_process = (
            json_manager.portrait_data
            if data_type == "portrait"
            else json_manager.group_data
        )
        if not data_to_process:
            logger.info(f"Нет {data_type} данных для анализа ключевых точек.")
            return

        detailed_metrics = []
        faces_processed_count = 0
        update_errors = 0
        try:
            memory = psutil.virtual_memory()
            logger.debug(
                get_message(
                    "INFO_MEMORY_USAGE",
                    percent=memory.percent,
                    available=memory.available / (1024 * 1024),
                    total=memory.total / (1024 * 1024),
                )
            )
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о памяти: {e}")

        logger.debug(f"Обработка {len(data_to_process)} файлов типа '{data_type}'...")
        for filename, value in tqdm(
            data_to_process.items(), desc=f"Анализ {data_type} лиц"
        ):
            if not isinstance(value, dict) or "faces" not in value:
                logger.warning(
                    f"Некорректный формат данных для файла {filename} в {data_type}. Пропуск."
                )
                continue
            faces = value["faces"]
            for face_idx, face in enumerate(faces):
                if not isinstance(face, dict):
                    logger.error(
                        get_message(
                            "ERROR_INVALID_FACE_TYPE",
                            key=filename,
                            type=type(face),
                            face=face,
                        )
                    )
                    continue
                keypoints_2d = face.get("landmark_2d_106")
                keypoints_3d = face.get("landmark_3d_68")
                pose = face.get("pose")
                if not keypoints_2d or not isinstance(keypoints_2d, list):
                    logger.debug(
                        get_message(
                            "WARNING_NO_2D_KEYPOINTS",
                            item=f"лицо {face_idx} файла {filename}",
                        )
                    )
                    continue  # Debug, т.к. может быть много
                analysis_results = self._analyze_keypoints(
                    keypoints_2d, keypoints_3d, pose
                )
                update_payload = {
                    "keypoint_analysis": {
                        "eye_states": {
                            "left": analysis_results["eye_states"]["left"]["state"],
                            "right": analysis_results["eye_states"]["right"]["state"],
                        },
                        "mouth_state": analysis_results["mouth_state"]["state"],
                        "head_pose": analysis_results["head_pose"]["state"],
                    }
                }
                if not json_manager.update_face(filename, face_idx, update_payload):
                    update_errors += 1
                faces_processed_count += 1
                detailed_metrics.append(
                    {  # Собираем подробные метрики
                        "face_id": f"{filename}_{face_idx}",
                        "eyes": {
                            "left": analysis_results["eye_states"]["left"],
                            "right": analysis_results["eye_states"]["right"],
                        },
                        "mouth": analysis_results["mouth_state"],
                        "head_pose": analysis_results["head_pose"],
                    }
                )

        if not json_manager.save_data(data_type=data_type):
            logger.error(
                f"Ошибка сохранения {data_type} JSON после анализа ключевых точек."
            )
        logger.info(
            f"Анализ {faces_processed_count} лиц типа '{data_type}' завершен. Ошибок обновления JSON: {update_errors}. Данные сохранены"
        )

        if detailed_metrics:
            analysis = self.analyze_metrics(detailed_metrics)
            stats_path = output_path / f"statistics_{json_path.stem}.json"
            full_stats_data = {
                "summary": {"total_faces": faces_processed_count},
                "detailed_metrics": detailed_metrics,
                "analysis": analysis,
            }  # Упростил summary
            try:
                with stats_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        full_stats_data,
                        f,
                        ensure_ascii=False,
                        indent=4,
                        default=lambda x: None
                        if isinstance(x, float) and (np.isnan(x) or np.isinf(x))
                        else x,
                    )
                logger.info(
                    f"Статистика анализа ключевых точек сохранена в {stats_path.resolve()}"
                )
            except Exception as e:
                logger.error(
                    f"Ошибка сохранения файла статистики {stats_path.resolve()}: {e}"
                )
            self.generate_recommendations(analysis, output_path, json_path)
        else:
            logger.info(
                f"Нет данных для расчета статистики и рекомендаций для {data_type}."
            )
