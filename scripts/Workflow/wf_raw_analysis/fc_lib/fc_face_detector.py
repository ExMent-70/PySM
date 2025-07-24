# fc_lib/fc_face_detector.py

from pathlib import Path
import logging
import cv2
import numpy as np
from typing import List, Optional, Tuple, Any, Dict
from insightface.app import FaceAnalysis

# --- Обновленные импорты ---
from .fc_config import ConfigManager
from .fc_onnx_manager import ONNXModelManager
from .fc_messages import get_message
from .face_data_processor_interface import FaceDataProcessorInterface

# --- Конец обновленных импортов ---

logger = logging.getLogger(__name__)


# === Функция пересчета координат (без изменений) ===
def _recalculate_coords_to_original(
    face_object: Any,
    x1_padded: int,
    y1_padded: int,
    x_offset: int,
    y_offset: int,
    scale_crop_x: float,
    scale_crop_y: float,
    logger_instance: logging.Logger,
) -> Dict[str, Optional[np.ndarray]]:
    recalculated_coords: Dict[str, Optional[np.ndarray]] = {
        "kps": None,
        "bbox": None,
        "landmark_2d_106": None,
        "landmark_3d_68": None,
    }
    kps_relative = getattr(face_object, "kps", None)
    if kps_relative is not None:
        try:
            kps_orig = np.array(
                [
                    (
                        x1_padded + (kp[0] - x_offset) * scale_crop_x,
                        y1_padded + (kp[1] - y_offset) * scale_crop_y,
                    )
                    for kp in kps_relative
                ],
                dtype=np.float32,
            )
            recalculated_coords["kps"] = kps_orig
        except (IndexError, TypeError) as e:
            logger_instance.warning(f"Ошибка пересчета kps: {e}")
        except Exception as e:
            logger_instance.error(
                f"Неожиданная ошибка пересчета kps: {e}", exc_info=True
            )
    bbox_relative = getattr(face_object, "bbox", None)
    if bbox_relative is not None:
        try:
            bbox_orig = np.array(
                [
                    x1_padded + (bbox_relative[0] - x_offset) * scale_crop_x,
                    y1_padded + (bbox_relative[1] - y_offset) * scale_crop_y,
                    x1_padded + (bbox_relative[2] - x_offset) * scale_crop_x,
                    y1_padded + (bbox_relative[3] - y_offset) * scale_crop_y,
                ],
                dtype=np.float32,
            )
            recalculated_coords["bbox"] = bbox_orig
        except (IndexError, TypeError) as e:
            logger_instance.warning(f"Ошибка пересчета bbox: {e}")
        except Exception as e:
            logger_instance.error(
                f"Неожиданная ошибка пересчета bbox: {e}", exc_info=True
            )
    landmarks_2d_relative = getattr(face_object, "landmark_2d_106", None)
    if landmarks_2d_relative is not None:
        try:
            landmarks_2d_orig = np.array(
                [
                    [
                        x1_padded + (kp[0] - x_offset) * scale_crop_x,
                        y1_padded + (kp[1] - y_offset) * scale_crop_y,
                    ]
                    for kp in landmarks_2d_relative
                ],
                dtype=np.float32,
            )
            recalculated_coords["landmark_2d_106"] = landmarks_2d_orig
        except (IndexError, TypeError) as e:
            logger_instance.warning(f"Ошибка пересчета landmark_2d_106: {e}")
        except Exception as e:
            logger_instance.error(
                f"Неожиданная ошибка пересчета landmark_2d_106: {e}", exc_info=True
            )
    landmarks_3d_relative = getattr(face_object, "landmark_3d_68", None)
    if landmarks_3d_relative is not None:
        try:
            landmarks_3d_orig = np.array(
                [
                    [
                        x1_padded + (kp[0] - x_offset) * scale_crop_x,
                        y1_padded + (kp[1] - y_offset) * scale_crop_y,
                        kp[2],
                    ]
                    for kp in landmarks_3d_relative
                ],
                dtype=np.float32,
            )
            recalculated_coords["landmark_3d_68"] = landmarks_3d_orig
        except (IndexError, TypeError) as e:
            logger_instance.warning(f"Ошибка пересчета landmark_3d_68: {e}")
        except Exception as e:
            logger_instance.error(
                f"Неожиданная ошибка пересчета landmark_3d_68: {e}", exc_info=True
            )
    return recalculated_coords


# === Конец функции ===


class FaceDetector:
    """Класс для обнаружения и анализа лиц на изображениях."""

    # Блок 1: Изменение конструктора __init__
    def __init__(
        self,
        config: ConfigManager,
        face_data_processors: List[FaceDataProcessorInterface],
        # --- НОВЫЙ АРГУМЕНТ ---
        onnx_manager: ONNXModelManager,
    ):
        """Инициализирует детектор лиц."""
        self.config = config
        self.face_data_processors = face_data_processors
        # --- НОВЫЙ АТРИБУТ ---
        self.onnx_manager = onnx_manager

        # Основной анализатор insightface инициализируется как и раньше,
        # так как это высокоуровневая обертка со своей логикой.
        self.analyzer = self._initialize_face_analysis()

        self.det_thresh = self.config.get("model", "det_thresh", 0.5)
        self.target_size = tuple(
            self.config.get("processing", "target_size", [640, 640])
        )
        self.save_debug_kps_images = self.config.get(
            "debug", "save_analyzed_kps_images", False
        )
        if self.save_debug_kps_images:
            logger.info("Включено сохранение отладочных изображений с ключевыми точками.")

    # Блок 2: Изменение метода _initialize_face_analysis
    def _initialize_face_analysis(self) -> FaceAnalysis:
        """
        Инициализирует основной анализатор insightface, используя провайдер
        из ONNXModelManager.
        """
        # --- ИЗМЕНЕНИЕ: Берем провайдер и опции из менеджера ---
        provider = self.onnx_manager.provider_name
        options = self.onnx_manager.provider_options
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        logger.info(f"Инициализация FaceAnalysis с провайдером '{provider}'...")
        try:
            model_root_path = self.config.get("paths", "model_root", "../BIN")
            app = FaceAnalysis(
                name=self.config.get("model", "name", "antelopev2"),
                providers=[provider],
                provider_options=options,
                root=str(model_root_path),
            )
            logger.info("Подготовка модели FaceAnalysis...")
            ctx_id = 0 if "ExecutionProvider" in provider and "CPU" not in provider else -1
            app.prepare(
                ctx_id=ctx_id, det_thresh=self.config.get("model", "det_thresh", 0.5)
            )
            logger.info(get_message("INFO_FACE_ANALYSIS_INITIALIZED") + f" Провайдер: {provider}.")
            return app
        except Exception as e:
            logger.error(
                get_message("ERROR_PROCESSING_IMAGE", file_path="FaceAnalysis init", exc=e),
                exc_info=True,
            )
            raise



    def detect_and_analyze(
        self, file_path: Path, img: np.ndarray
    ) -> Tuple[
        Optional[List[Dict[str, Any]]], Optional[List[np.ndarray]], Tuple[int, int]
    ]:
        """
        Обрабатывает изображение: обнаруживает лица, анализирует их и возвращает данные.
        Возвращает:
        - Список словарей данных лиц (без эмбеддингов)
        - Список эмбеддингов (numpy arrays) в том же порядке
        - Форму ИСХОДНОГО изображения (до детекции).
        """
        
        try:
            original_shape = img.shape[:2]
            logger.debug(
                f"Размер изображения {file_path.name} для детекции: {original_shape}"
            )
            det_size = tuple(self.config.get("model", "det_size", default=[1280, 1280]))
            height, width = original_shape
            if width <= 0 or height <= 0:
                logger.error(
                    f"Нулевой размер изображения {file_path.name}: ({height}, {width})"
                )
                return None, None, (0, 0)  # None для лиц и эмбеддингов
            scale = min(
                det_size[0] / width if width > 0 else 1,
                det_size[1] / height if height > 0 else 1,
            )
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            if abs(scale - 1.0) > 1e-6:
                img_resized = cv2.resize(
                    img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
                logger.debug(
                    f"Изображение {file_path.name} масштабировано до: ({new_height}, {new_width}) для детекции"
                )
            else:
                img_resized = img
                logger.debug(f"Масштабирование не требуется для {file_path.name}")

            faces_initial = self.analyzer.get(img_resized)
            logger.debug(
                f"Обнаружено лиц до фильтрации в {file_path.name}: {len(faces_initial)}"
            )
            faces_initial = [
                face
                for face in faces_initial
                if hasattr(face, "det_score") and face.det_score >= self.det_thresh
            ]
            logger.debug(
                f"Обнаружено лиц после фильтрации в {file_path.name}: {len(faces_initial)}"
            )
            if not faces_initial:
                return None, None, original_shape  # None для лиц и эмбеддингов

            valid_faces_initial = []
            scale_x = (
                width / new_width if new_width > 0 else 1.0
            )  # Защита от деления на ноль
            scale_y = (
                height / new_height if new_height > 0 else 1.0
            )  # Защита от деления на ноль
            for face in faces_initial:
                if not hasattr(face, "bbox") or face.bbox is None:
                    continue
                try:
                    face.original_bbox = (
                        face.bbox * np.array([scale_x, scale_y, scale_x, scale_y])
                    ).tolist()
                    valid_faces_initial.append(face)
                except Exception as e:
                    logger.warning(
                        f"Ошибка пересчета initial bbox для лица в {file_path.name}: {e}"
                    )

            if not valid_faces_initial:
                return None, None, original_shape  # None для лиц и эмбеддингов

            # Получаем оба списка из _analyze_cropped_faces
            analyzed_face_data_list, analyzed_face_embeddings_list = (
                self._analyze_cropped_faces(
                    valid_faces_initial,
                    img,
                    file_path,
                    img.copy() if self.save_debug_kps_images else None,
                )
            )

            return (
                analyzed_face_data_list if analyzed_face_data_list else None,
                analyzed_face_embeddings_list
                if analyzed_face_embeddings_list
                else None,
                original_shape,
            )

        except Exception as e:
            logger.error(
                get_message("ERROR_PROCESSING_IMAGE", file_path=file_path, exc=e),
                exc_info=True,
            )
            return None, None, (0, 0)  # None для лиц и эмбеддингов при ошибке

    def _convert_face_to_dict(
        self,
        face: Any,
        original_coords: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Преобразует объект insightface.Face в базовый словарь данных (БЕЗ эмбеддинга).
        """
        # Проверяем хотя бы базовые атрибуты, например bbox или det_score
        if not hasattr(face, "det_score") and not hasattr(face, "bbox"):
            logger.warning(
                "Попытка преобразовать объект лица без базовых атрибутов (det_score/bbox)."
            )
            return None  # Считаем объект невалидным

        coords_source = original_coords if original_coords else {}
        data = {
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0,
            "gender_insight": int(face.gender)
            if hasattr(face, "gender") and face.gender is not None
            else None,
            "age_insight": int(face.age)
            if hasattr(face, "age") and face.age is not None
            else None,
            # Эмбеддинг здесь НЕ добавляется
            "cluster_label": None,
            "child_name": None,
            "matched_portrait_cluster_label": None,
            "matched_child_name": None,
            "match_distance": None,
            "keypoint_analysis": None,
        }
        kps_final = (
            coords_source.get("kps")
            if coords_source.get("kps") is not None
            else getattr(face, "kps", None)
        )
        data["kps"] = kps_final.tolist() if kps_final is not None else None
        bbox_final = (
            coords_source.get("bbox")
            if coords_source.get("bbox") is not None
            else getattr(face, "bbox", None)
        )
        data["bbox"] = bbox_final.tolist() if bbox_final is not None else None
        lmk3d_final = (
            coords_source.get("landmark_3d_68")
            if coords_source.get("landmark_3d_68") is not None
            else getattr(face, "landmark_3d_68", None)
        )
        data["landmark_3d_68"] = (
            lmk3d_final.tolist() if lmk3d_final is not None else None
        )
        lmk2d_final = (
            coords_source.get("landmark_2d_106")
            if coords_source.get("landmark_2d_106") is not None
            else getattr(face, "landmark_2d_106", None)
        )
        data["landmark_2d_106"] = (
            lmk2d_final.tolist() if lmk2d_final is not None else None
        )
        pose_final = getattr(face, "pose", None)
        data["pose"] = pose_final.tolist() if pose_final is not None else None
        original_bbox_val = getattr(face, "original_bbox", None)
        if isinstance(original_bbox_val, np.ndarray):
            data["original_bbox"] = original_bbox_val.tolist()
        elif isinstance(original_bbox_val, (list, tuple)):
            data["original_bbox"] = list(original_bbox_val)
        else:
            data["original_bbox"] = None
        return data

    # Внутри класса FaceDetector в fc_lib/fc_face_detector.py

    # Внутри класса FaceDetector в fc_lib/fc_face_detector.py

    def _analyze_cropped_faces(
        self,
        faces: List[Any],  # Список face_initial
        img: np.ndarray,  # Оригинал
        file_path: Path,
        img_for_debug_kps: Optional[np.ndarray] = None,  # Копия для отладки
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """
        Анализирует кропы лиц, вызывает обработчики, возвращает словари данных и эмбеддинги.
        """
        target_height, target_width = self.target_size
        padding_factor = 0.45
        processed_face_data_list: List[Dict[str, Any]] = []
        processed_face_embeddings_list: List[np.ndarray] = []
        img_h, img_w = img.shape[:2]
        save_debug_images_flag = self.save_debug_kps_images  # Флаг из __init__

        logger.debug(
            f"Анализ кропов для {file_path.name}. Флаг save_debug_kps_images={save_debug_images_flag}. img_for_debug_kps is None: {img_for_debug_kps is None}"
        )
        for idx, face_initial in enumerate(faces):
            # --- Получение кропа (cropped_face_initial) ---
            bbox = face_initial.original_bbox
            if bbox is None or len(bbox) != 4:
                logger.warning(
                    f"Лицо {idx} в {file_path.name}: некорректный original_bbox. Пропуск."
                )
                continue
            x1, y1, x2, y2 = map(int, bbox)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width <= 0 or bbox_height <= 0:
                logger.warning(
                    f"Лицо {idx} в {file_path.name}: некорректный bbox [{x1},{y1},{x2},{y2}]. Пропуск."
                )
                continue
            padding_x = int(bbox_width * padding_factor)
            padding_y = int(bbox_height * padding_factor)
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(img_w, x2 + padding_x)
            y2_padded = min(img_h, y2 + padding_y)
            if x2_padded <= x1_padded or y2_padded <= y1_padded:
                logger.warning(
                    f"Лицо {idx} в {file_path.name}: некорректные коорд. после паддинга. Пропуск."
                )
                continue
            cropped_face_initial = img[y1_padded:y2_padded, x1_padded:x2_padded]
            if cropped_face_initial.size == 0:
                logger.warning(f"Лицо {idx} в {file_path.name}: пустой кроп. Пропуск.")
                continue

            # --- Масштабирование кропа и подготовка к анализу ---
            orig_height, orig_width = cropped_face_initial.shape[:2]
            aspect_ratio = orig_width / orig_height if orig_height > 0 else 1
            new_width, new_height = target_width, target_height
            if aspect_ratio > 1:
                new_width, new_height = (
                    target_width,
                    max(1, int(target_width / aspect_ratio)),
                )
            else:
                new_height, new_width = (
                    target_height,
                    max(1, int(target_height * aspect_ratio)),
                )
            resized_face = cv2.resize(
                cropped_face_initial,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            padded_face_for_analysis = np.zeros(
                (target_height, target_width, 3), dtype=np.uint8
            )
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            padded_face_for_analysis[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = resized_face
            if padded_face_for_analysis.shape[2] == 3:
                cropped_face_rgb = cv2.cvtColor(
                    padded_face_for_analysis, cv2.COLOR_BGR2RGB
                )
            else:
                cropped_face_rgb = cv2.cvtColor(
                    padded_face_for_analysis, cv2.COLOR_GRAY2RGB
                )

            # --- Вторичный анализ кропа ---
            try:
                analyzed_face_list = self.analyzer.get(cropped_face_rgb)
            except Exception as analysis_err:
                logger.error(
                    f"Ошибка insightface.get() для кропа лица {idx} файла {file_path.name}: {analysis_err}",
                    exc_info=True,
                )
                continue  # Пропускаем это лицо

            if not analyzed_face_list:
                logger.debug(
                    f"Повторный анализ кропа лица {idx} не дал результатов в {file_path.name}. Пропуск."
                )
                continue

            # --- ИСПРАВЛЕНИЕ: Получаем face_final ДО извлечения эмбеддинга ---
            face_final = analyzed_face_list[0]
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            # Извлекаем эмбеддинг
            face_embedding_final = getattr(face_final, "embedding", None)
            if face_embedding_final is None:
                logger.warning(
                    f"face_final не имеет эмбеддинга для {file_path.name}, лицо {idx}. Пропуск."
                )
                continue

            # --- Пересчет координат ---
            scale_crop_x = (x2_padded - x1_padded) / new_width if new_width > 0 else 1
            scale_crop_y = (y2_padded - y1_padded) / new_height if new_height > 0 else 1
            original_coords_dict = _recalculate_coords_to_original(
                face_object=face_final,
                x1_padded=x1_padded,
                y1_padded=y1_padded,
                x_offset=x_offset,
                y_offset=y_offset,
                scale_crop_x=scale_crop_x,
                scale_crop_y=scale_crop_y,
                logger_instance=logger,
            )

            # --- Создание словаря данных (без эмбеддинга) ---
            face_final.original_bbox = (
                face_initial.original_bbox
            )  # Переносим исходный bbox
            face_data_dict = self._convert_face_to_dict(
                face_final, original_coords_dict
            )
            if face_data_dict is None:
                logger.warning(
                    f"Не удалось создать базовый словарь для лица {idx} файла {file_path.name}. Пропуск."
                )
                continue

            face_data_bundle = {
                "full_image": img,
                "face_crop_initial": cropped_face_initial,
                "landmarks_2d_106": original_coords_dict.get("landmark_2d_106"),
                "landmarks_3d_68": original_coords_dict.get("landmark_3d_68"),
                "pose": face_final.pose if hasattr(face_final, "pose") and face_final.pose is not None else None,
                "filename": file_path.name,
                "face_index": idx,
                "config": self.config,
                "embedding": face_embedding_final,
                # --- НОВЫЙ КЛЮЧ ---
                "onnx_manager": self.onnx_manager,
            }

                      
            # --- Вызов обработчиков (код остается тот же, но теперь они получат менеджер) ---
            if self.face_data_processors:
                for processor in self.face_data_processors:
                    processor_name = type(processor).__name__
                    try:
                        # Теперь face_data_bundle содержит onnx_manager
                        processor.process_face_data(face_data_bundle, face_data_dict)
                        logger.debug(f"Обработчик {processor_name} завершил работу.")
                    except Exception as process_err:
                        logger.error(f"Ошибка при вызове обработчика {processor_name}: {process_err}", exc_info=True)
                        face_data_dict[f"error_{processor_name}"] = str(process_err)                        
                        
                        

            # --- Сохранение отладочного изображения (с добавленным логированием) ---
            if save_debug_images_flag and img_for_debug_kps is not None:
                logger.debug(
                    f"Попытка сохранения отладочного KPS для {file_path.name}, лицо {idx}"
                )
                kps_list = face_data_dict.get("kps")
                if kps_list and isinstance(kps_list, list):
                    if not all(
                        isinstance(p, (list, tuple)) and len(p) >= 2 for p in kps_list
                    ):
                        logger.warning(
                            f"Отладка KPS {file_path.name}/{idx}: неверный формат kps_list."
                        )
                    else:
                        debug_image_copy = img_for_debug_kps.copy()
                        points_drawn = 0
                        for kp_idx, kp in enumerate(kps_list):
                            try:
                                if len(kp) < 2:
                                    continue
                                kp_x_float, kp_y_float = float(kp[0]), float(kp[1])
                                if not (
                                    np.isfinite(kp_x_float) and np.isfinite(kp_y_float)
                                ):
                                    continue
                                kp_x, kp_y = int(kp_x_float), int(kp_y_float)
                                if (
                                    0 <= kp_x < debug_image_copy.shape[1]
                                    and 0 <= kp_y < debug_image_copy.shape[0]
                                ):
                                    cv2.circle(
                                        debug_image_copy,
                                        (kp_x, kp_y),
                                        5,
                                        (0, 0, 255),
                                        -1,
                                    )
                                    points_drawn += 1
                                # else: logger.warning(f"Отладка KPS {file_path.name}/{idx}: точка {kp_idx} вне границ.")
                            except (ValueError, TypeError, IndexError) as draw_err:
                                logger.error(
                                    f"Отладка KPS {file_path.name}/{idx}: ошибка точки {kp_idx} {kp}: {draw_err}"
                                )

                        if points_drawn > 0:
                            output_path_path = Path(
                                self.config.get("paths", "output_path")
                            )
                            debug_dir = output_path_path / "debug_kps"
                            try:
                                debug_dir.mkdir(parents=True, exist_ok=True)
                            except OSError as mkdir_err:
                                logger.error(
                                    f"Не удалось создать папку {debug_dir.resolve()}: {mkdir_err}"
                                )
                                continue
                            debug_filename = str(
                                debug_dir
                                / f"{file_path.stem}_face{idx}_analyzed_kps_orig.jpg"
                            )
                            try:
                                success_write = cv2.imwrite(
                                    debug_filename, debug_image_copy
                                )
                                if success_write:
                                    logger.info(
                                        f"Отладочное изображение СОХРАНЕНО: {debug_filename}"
                                    )
                                else:
                                    logger.error(
                                        f"cv2.imwrite ВЕРНУЛ FALSE: {debug_filename}"
                                    )
                            except Exception as imwrite_err:
                                logger.error(
                                    f"ИСКЛЮЧЕНИЕ при сохранении отладки {debug_filename}: {imwrite_err}",
                                    exc_info=True,
                                )
                        else:
                            logger.warning(
                                f"Отладка KPS {file_path.name}/{idx}: точки были, но ни одна не нарисована."
                            )
                else:
                    logger.warning(
                        f"Отладка KPS {file_path.name}/{idx}: ключ 'kps' отсутствует или не список."
                    )
            elif save_debug_images_flag and img_for_debug_kps is None:
                logger.warning(
                    f"Отладка KPS {file_path.name}/{idx}: флаг включен, но img_for_debug_kps is None!"
                )

            # Добавляем данные в списки
            processed_face_data_list.append(face_data_dict)
            processed_face_embeddings_list.append(face_embedding_final)
            # --- Конец цикла по лицам ---

        return processed_face_data_list, processed_face_embeddings_list
