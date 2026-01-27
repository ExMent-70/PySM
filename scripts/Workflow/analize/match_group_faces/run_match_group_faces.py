# analize/match_group_faces/run_match_group_faces.py
"""
Идентифицирует лица на групповых фотографиях путем их сопоставления
с эталонными портретными кластерами.

Особенности реализации:
- Использует косинусное расстояние (Cosine Distance) для сравнения.
- Центроиды кластеров вычисляются с учетом L2-нормализации.
- Работает с аргументами командной строки через PySM ConfigResolver.
- Генерирует JSON-отчет о совпадениях (matches_portrait_to_group.json).
- Генерирует JSON-отчет о нераспознанных лицах (error_matches.json).
"""

# --- Блок 1: Импорты и настройка окружения ---
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from scipy.spatial.distance import cdist

# Настройка системного пути
try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    pass

# Импорт модулей проекта
from _common.json_data_manager import JsonDataManager
from _common._shared import EmbeddingLoader
from pysm_lib.pysm_context import ConfigResolver
from pysm_lib.pysm_progress_reporter import tqdm


# --- Блок 2: Константы и Логирование ---
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

CONST_METRIC = "cosine"
CONST_DEFAULT_THRESHOLD = 0.45


# --- Блок 3: Структуры данных (DTO) ---
class ClusterProfile(NamedTuple):
    """
    Хранит данные об эталонном кластере (портрете).
    """
    label: int
    vector: np.ndarray  # Нормализованный усредненный вектор
    child_name: str


class MatchResult(NamedTuple):
    """
    Результат сопоставления для конкретного лица с группового фото.
    """
    group_face_index: int      # Индекс лица в массиве эмбеддингов
    best_cluster_label: int    # ID наиболее подходящего кластера
    distance: float            # Косинусное расстояние до этого кластера
    is_match: bool             # Удовлетворяет ли расстояние порогу


# --- Блок 4: Компоненты бизнес-логики ---

class CentroidCalculator:
    """
    Сервис для создания 'Галереи эталонов'.
    Преобразует разрозненные портретные эмбеддинги в усредненные векторы кластеров.
    """

    @staticmethod
    def calculate(
        p_embeds: np.ndarray,
        p_index: Dict[str, int],
        manager: JsonDataManager
    ) -> Dict[int, ClusterProfile]:
        """
        Вычисляет центроиды для каждого кластера.
        
        Args:
            p_embeds: Массив всех портретных эмбеддингов.
            p_index: Словарь {filename: index} для портретов.
            manager: Менеджер данных с информацией о кластерах.
            
        Returns:
            Словарь {cluster_label: ClusterProfile}.
        """
        # 1. Сбор индексов векторов по кластерам
        cluster_indices = defaultdict(list)
        cluster_names = {}

        for filename, data in manager.portrait_data.items():
            faces = data.get("faces", [])
            if not faces:
                continue
            
            face = faces[0]
            label = face.get("cluster_label")

            # Игнорируем шумовые (-1) и неразмеченные кластеры
            if label is not None and label != -1:
                emb_idx = p_index.get(filename)
                if emb_idx is not None:
                    cluster_indices[label].append(emb_idx)
                    # Сохраняем имя (если еще не сохранили)
                    if label not in cluster_names:
                        cluster_names[label] = face.get("child_name", f"Unknown_{label}")

        # 2. Математическое усреднение и нормализация
        profiles = {}
        for label, indices in cluster_indices.items():
            if not indices:
                continue
                
            vectors = p_embeds[indices]
            mean_vector = vectors.mean(axis=0)

            # L2-нормализация обязательна для корректного сравнения через Cosine Distance
            norm = np.linalg.norm(mean_vector)
            normalized_vector = mean_vector / norm if norm > 1e-6 else mean_vector

            profiles[label] = ClusterProfile(
                label=label,
                vector=normalized_vector,
                child_name=cluster_names.get(label, "Unknown")
            )

        logger.info(f"Сформировано эталонных портретов: <b>{len(profiles)}</b>")
        return profiles


class CosineFaceMatcher:
    """
    Ядро сопоставления. Отвечает только за математические операции.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        logger.info(f"Метрика: <b>{CONST_METRIC}</b>")
        logger.info(f"Порог схожести: <b>{self.threshold}</b><br>")

    def match_batch(
        self,
        group_embeds: np.ndarray,
        centroids_map: Dict[int, ClusterProfile]
    ) -> List[MatchResult]:
        """
        Выполняет массовое сопоставление векторов.
        """
        if not centroids_map:
            return []

        # Подготовка данных для cdist
        # Гарантируем порядок лейблов
        labels = list(centroids_map.keys())
        # Матрица эталонов (M x D)
        centroid_matrix = np.array([p.vector for p in centroids_map.values()])

        # Вычисление матрицы расстояний (N x M)
        # N - кол-во групповых лиц, M - кол-во эталонов
        distances = cdist(group_embeds, centroid_matrix, metric=CONST_METRIC)

        # Находим минимальное расстояние для каждого лица
        min_indices = np.argmin(distances, axis=1)
        min_vals = np.min(distances, axis=1)

        results = []
        for i in range(len(group_embeds)):
            dist = float(min_vals[i])
            best_idx = min_indices[i]
            label = labels[best_idx]
            
            # Принимаем решение на основе порога
            is_match = dist < self.threshold

            results.append(MatchResult(
                group_face_index=i,
                best_cluster_label=label,
                distance=dist,
                is_match=is_match
            ))
        
        return results


class ResultHandler:
    """
    Отвечает за применение результатов к данным проекта и сохранение отчетов.
    """

    def __init__(self, manager: JsonDataManager, centroids: Dict[int, ClusterProfile]):
        self.manager = manager
        self.centroids = centroids

    def apply_matches(self, results: List[MatchResult], g_index_map: Dict[str, int]):
        """
        Обновляет метаданные в JsonDataManager на основе результатов матчинга.
        """
        # Инвертируем карту индексов {index -> filename::face_idx} для быстрого доступа
        index_to_key = {v: k for k, v in g_index_map.items()}

        for res in tqdm(results, desc="Обновление метаданных"):
            key = index_to_key.get(res.group_face_index)
            if not key:
                continue

            try:
                filename, face_idx_str = key.split("::")
                face_idx = int(face_idx_str)
            except ValueError:
                logger.warning(f"Некорректный формат ключа индекса: {key}")
                continue

            # Подготовка данных для обновления
            update_data: Dict[str, Any] = {
                "match_distance": res.distance
            }

            if res.is_match:
                profile = self.centroids[res.best_cluster_label]
                update_data["matched_portrait_cluster_label"] = int(profile.label)
                update_data["matched_child_name"] = profile.child_name
            else:
                update_data["matched_portrait_cluster_label"] = None
                update_data["matched_child_name"] = "No Match"

            self.manager.update_face(filename, face_idx, update_data, data_type="group")

    def save_json_report(self, output_dir: Path):
        """
        Формирует и сохраняет matches_portrait_to_group.json (только совпадения).
        """
        report = {}

        # 1. Инициализация структуры
        for label, profile in self.centroids.items():
            report[str(label)] = {
                "child_name": profile.child_name,
                "group_photos": []
            }

        # 2. Агрегация данных
        for filename, data in self.manager.group_data.items():
            file_matches = defaultdict(list)
            
            for face in data.get("faces", []):
                label = face.get("matched_portrait_cluster_label")
                dist = face.get("match_distance")
                
                if label is not None and dist is not None:
                    file_matches[str(label)].append(dist)

            for label_str, dists in file_matches.items():
                if label_str in report:
                    report[label_str]["group_photos"].append({
                        "filename": filename,
                        "min_distance": min(dists),
                        "num_faces": len(dists)
                    })

        # 3. Сортировка
        for label_data in report.values():
            label_data["group_photos"].sort(
                key=lambda x: x.get("min_distance", float("inf"))
            )

        # 4. Запись
        output_path = output_dir / "matches_portrait_to_group.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ список сопоставления: <i>{output_path.name}</i>")

    def save_error_report(self, output_dir: Path):
        """
        Формирует и сохраняет error_matches.json (нераспознанные лица).
        """
        report = {
            "description": "Список групповых фотографий с лицами, не сопоставленными ни с одним кластером.",
            "unmatched_files": []
        }

        total_unmatched_faces = 0

        for filename, data in self.manager.group_data.items():
            unmatched_faces = []
            
            # Проходим по всем лицам на фото
            for i, face in enumerate(data.get("faces", [])):
                label = face.get("matched_portrait_cluster_label")
                
                # Если label is None или явно "No Match" - это ошибка сопоставления
                if label is None:
                    # Добавляем информацию о дистанции (насколько мы были далеки от ближайшего кандидата)
                    # Это полезно для настройки Threshold.
                    dist = face.get("match_distance")
                    unmatched_faces.append({
                        "face_index": i,
                        "nearest_match_distance": dist if dist is not None else -1.0
                    })

            if unmatched_faces:
                report["unmatched_files"].append({
                    "filename": filename,
                    "unmatched_count": len(unmatched_faces),
                    "faces": unmatched_faces
                })
                total_unmatched_faces += len(unmatched_faces)

        # Сортируем файлы по количеству нераспознанных лиц (от большего к меньшему)
        report["unmatched_files"].sort(key=lambda x: x["unmatched_count"], reverse=True)
        report["total_unmatched_faces"] = total_unmatched_faces

        output_path = output_dir / "error_matches.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(
            f"✅ отчет об ошибках (<b>{total_unmatched_faces}</b> неидентифицированных лиц): <i>{output_path.name}</i><br>"
        )


# --- Блок 5: Обработка аргументов ---
def get_config() -> argparse.Namespace:
    """
    Определяет и парсит аргументы скрипта с помощью PySM ConfigResolver.
    """
    parser = argparse.ArgumentParser(
        description="Сопоставление лиц на групповых фото с эталонными портретами."
    )
    arg_prefix = "a_mg_"

    parser.add_argument(
        f"--{arg_prefix}portrait_json",
        type=str,
        required=True,
        help="Путь к info_portrait_faces.json (эталон)",
    )
    parser.add_argument(
        f"--{arg_prefix}group_json",
        type=str,
        required=True,
        help="Путь к info_group_faces.json для анализа",
    )
    parser.add_argument(
        f"--{arg_prefix}threshold",
        type=float,
        default=CONST_DEFAULT_THRESHOLD,
        help=f"Порог схожести (cosine distance). По умолчанию {CONST_DEFAULT_THRESHOLD}",
    )

    return ConfigResolver(parser).resolve_all()


# --- Блок 6: Точка входа ---
def main():
    logger.info("<b>Сопоставление лиц на групповых фотографиях с эталонными портретами</b><br>")
    try:
        # 1. Конфигурация
        cli_config = get_config()
        arg_prefix = "a_mg_"
        
        p_json_path = Path(getattr(cli_config, f"{arg_prefix}portrait_json"))
        g_json_path = Path(getattr(cli_config, f"{arg_prefix}group_json"))
        threshold = getattr(cli_config, f"{arg_prefix}threshold")
        
        output_dir = g_json_path.parent
        
        # Динамическое определение путей к эмбеддингам
        p_emb_dir = p_json_path.parent / "_Embeddings"
        g_emb_dir = g_json_path.parent / "_Embeddings"

        logger.debug(f"Портреты: {p_json_path}")
        logger.debug(f"Группы: {g_json_path}")
        logger.debug(f"Папка эмбеддингов портретов: {p_emb_dir}")

        # 2. Загрузка данных
        # Эмбеддинги
        p_loader = EmbeddingLoader(p_emb_dir)
        p_embeds, p_index = p_loader.load("portrait")
        
        g_loader = EmbeddingLoader(g_emb_dir)
        g_embeds, g_index = g_loader.load("group")

        if p_embeds is None or p_index is None:
            logger.error("❌ Критические данные отсутствуют: нет портретных эмбеддингов")
            sys.exit(1)

        # JSON метаданные
        json_manager = JsonDataManager(
            portrait_json_path=p_json_path, 
            group_json_path=g_json_path
        )
        if not json_manager.load_data():
            logger.error("❌ Ошибка загрузки JSON файлов")
            sys.exit(1)

        # 3. Обработка (Пайплайн)
        
        # Шаг A: Подготовка эталонов
        centroids = CentroidCalculator.calculate(p_embeds, p_index, json_manager)
        if not centroids:
            logger.warning("Не найдено валидных кластеров портретов. Сопоставление невозможно.")
            # Пустые отчеты
            handler = ResultHandler(json_manager, {})
            handler.save_json_report(output_dir)
            handler.save_error_report(output_dir)
            sys.exit(0)
        # Проверка наличия групповых данных
        if g_embeds is None or g_index is None:
            logger.warning("Групповые эмбеддинги не найдены.")
            handler = ResultHandler(json_manager, centroids)
            handler.save_json_report(output_dir)
            handler.save_error_report(output_dir)
            sys.exit(0)

        # Шаг B: Математическое сопоставление
        matcher = CosineFaceMatcher(threshold=threshold)
        match_results = matcher.match_batch(g_embeds, centroids)
        # Шаг C: Применение результатов
        handler = ResultHandler(json_manager, centroids)
        handler.apply_matches(match_results, g_index)

        # 4. Сохранение результатов
        # Сохраняем только group_json
        logger.info(f"<br>")        
        json_manager.portrait_json_path = None
        json_manager.save_data()
        
        # Генерация отчетов
        handler.save_json_report(output_dir)
        handler.save_error_report(output_dir)

        logger.debug("Процесс завершен успешно.")

    except Exception as e:
        logger.critical(f"Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()